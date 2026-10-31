"""Per-account SQLite authority, immutable revisions, and atomic artifact publication."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

try:
    from .grading_document import GradingError, document, entry_body
except ImportError:
    from grading_document import GradingError, document, entry_body


def owner_key(platform, sender):
    if not platform or not sender or str(sender).lower() in {"unknown", "none"}:
        raise GradingError("无法确认当前用户身份，已停止访问作答本")
    return hashlib.sha256(json.dumps([str(platform), str(sender)], ensure_ascii=False).encode()).hexdigest()


class GradingStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "records.sqlite3"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS owners(owner TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
              book_dir TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS records(owner TEXT NOT NULL, number INTEGER NOT NULL,
              revision INTEGER NOT NULL, original_json TEXT NOT NULL, current_json TEXT NOT NULL,
              artifact_dir TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
              PRIMARY KEY(owner,number));
            CREATE TABLE IF NOT EXISTS revisions(owner TEXT NOT NULL, number INTEGER NOT NULL,
              revision INTEGER NOT NULL, record_json TEXT NOT NULL, instruction TEXT NOT NULL,
              artifact_dir TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY(owner,number,revision));
            CREATE TABLE IF NOT EXISTS requests(owner TEXT NOT NULL, request_key TEXT NOT NULL,
              number INTEGER NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(owner,request_key));
            CREATE TABLE IF NOT EXISTS vectors(owner TEXT NOT NULL, model_key TEXT NOT NULL,
              content_hash TEXT NOT NULL, vector_json TEXT NOT NULL,
              PRIMARY KEY(owner,model_key,content_hash));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _user(self, owner):
        if not isinstance(owner, str) or len(owner) != 64 or any(c not in '0123456789abcdef' for c in owner):
            raise GradingError("用户标识无效")
        return self.root / "users" / owner

    def rows(self, owner):
        self._user(owner)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM records WHERE owner=? ORDER BY number", (owner,)).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        value = dict(row)
        value["original"] = json.loads(value.pop("original_json"))
        value["record"] = json.loads(value.pop("current_json"))
        return value

    def get(self, owner, number):
        self._user(owner)
        with self.connect() as db:
            row = db.execute("SELECT * FROM records WHERE owner=? AND number=?", (owner, number)).fetchone()
        if row is None:
            raise GradingError(f"你的作答本中没有第 {number} 题")
        return self._decode(row)

    def version(self, owner):
        with self.connect() as db:
            row = db.execute("SELECT version,book_dir FROM owners WHERE owner=?", (owner,)).fetchone()
        return (row["version"], row["book_dir"]) if row else (0, "")

    def request(self, owner, key):
        if not key:
            return None
        with self.connect() as db:
            row = db.execute("SELECT number,revision FROM requests WHERE owner=? AND request_key=?", (owner, key)).fetchone()
        return dict(row) if row else None

    def artifact(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise GradingError("作答本文件路径无效")
        return path

    def publish(self, owner, number, revision, record, pdf_bytes, request_key="", instruction=""):
        """Prepare all artifacts first. SQLite is the single atomic visibility point."""
        import fitz
        rows = self.rows(owner)
        version, _ = self.version(owner)
        old = next((r for r in rows if r["number"] == number), None)
        expected = old["revision"] + 1 if old else 1
        next_number = max((r["number"] for r in rows), default=0) + 1
        if revision != expected or (old is None and number != next_number):
            raise GradingError("作答本已发生变化，请重新执行")
        identifier = uuid.uuid4().hex
        entry = self._user(owner) / "entries" / f"{number:06d}" / f"r{revision}-{identifier}"
        book = self._user(owner) / "books" / f"v{version+1}-{identifier}"
        entry.mkdir(parents=True, mode=0o700)
        book.mkdir(parents=True, mode=0o700)
        committed = False
        try:
            (entry / "answer.tex").write_text(document(entry_body(number, revision, record)), encoding="utf-8")
            (entry / "answer.pdf").write_bytes(pdf_bytes)
            relative = str(entry.relative_to(self.root))
            candidate = {"number": number, "revision": revision, "record": record, "artifact_dir": relative}
            combined = sorted([r for r in rows if r["number"] != number] + [candidate], key=lambda r: r["number"])
            bodies, toc = [], []
            with fitz.open() as merged:
                for row in combined:
                    bodies.append(entry_body(row["number"], row["revision"], row["record"]))
                    toc.append([1, f"第{row['number']}题：{row['record']['title']}", len(merged) + 1])
                    with fitz.open(self.artifact(row["artifact_dir"]) / "answer.pdf") as source:
                        if not len(source):
                            raise GradingError("生成了空白 PDF，未保存本次修改")
                        merged.insert_pdf(source)
                for i, page in enumerate(merged):
                    # Consistent global pagination after joining independent questions.
                    box = fitz.Rect(0, page.rect.height - 35, page.rect.width, page.rect.height)
                    page.draw_rect(box, color=None, fill=(1, 1, 1), overlay=True)
                    page.insert_textbox(box, str(i + 1), fontsize=9, align=1, color=(0.4, 0.4, 0.4))
                merged.set_toc(toc)
                merged.set_metadata({"title": "个人数学作答本", "author": "AstrBot mathsolve"})
                merged.save(book / "book.pdf", garbage=4, deflate=True)
            (book / "book.tex").write_text(document("\n\\newpage\n".join(bodies)), encoding="utf-8")
            # Flush artifacts before committing the DB pointers (crash consistency).
            for path in [entry / "answer.tex", entry / "answer.pdf", book / "book.pdf", book / "book.tex"]:
                with path.open("rb") as file:
                    os.fsync(file.fileno())
            for path in (entry, entry.parent, entry.parent.parent, book, book.parent, self._user(owner)):
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            now = datetime.now(timezone.utc).isoformat()
            text = json.dumps(record, ensure_ascii=False)
            original = json.dumps(old["original"], ensure_ascii=False) if old else text
            book_relative = str(book.relative_to(self.root))
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("INSERT OR IGNORE INTO owners(owner) VALUES(?)", (owner,))
                actual = db.execute("SELECT version FROM owners WHERE owner=?", (owner,)).fetchone()[0]
                if actual != version:
                    raise GradingError("作答本正在更新，请重试；原记录未改变")
                db.execute("""INSERT INTO records VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(owner,number) DO UPDATE SET revision=excluded.revision,
                    current_json=excluded.current_json,artifact_dir=excluded.artifact_dir,updated=excluded.updated""",
                           (owner, number, revision, original, text, relative, old["created"] if old else now, now))
                db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?,?)",
                           (owner, number, revision, text, instruction, relative, now))
                if request_key:
                    db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_key, number, revision))
                db.execute("UPDATE owners SET version=?,book_dir=? WHERE owner=?", (version+1, book_relative, owner))
            committed = True
            try:
                self._prune_books(owner, book)
            except OSError:
                pass  # Optional cleanup must not turn a committed write into failure.
            return self.get(owner, number)
        finally:
            if not committed:
                shutil.rmtree(entry, ignore_errors=True)
                shutil.rmtree(book, ignore_errors=True)

    def _prune_books(self, owner, current):
        # Keep all entry revisions, but not quadratic copies of old full books.
        # A full-day grace period protects concurrent sends of an older export.
        for path in (self._user(owner) / "books").iterdir():
            if path != current and path.is_dir() and time.time() - path.stat().st_mtime > 86400:
                shutil.rmtree(path, ignore_errors=True)

    def export(self, owner):
        _, relative = self.version(owner)
        if not relative:
            raise GradingError("你的作答本还是空的，请先提交一次批改")
        path = self.artifact(relative) / "book.pdf"
        if not path.is_file():
            raise GradingError("作答本 PDF 文件缺失，请联系管理员；题目记录仍保留")
        return path

    def history(self, owner, number):
        self.get(owner, number)
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT revision,instruction,created FROM revisions WHERE owner=? AND number=? ORDER BY revision", (owner, number))]

    def owners(self):
        with self.connect() as db:
            return [r[0] for r in db.execute("SELECT owner FROM owners")]

    def vectors(self, owner, model_key):
        with self.connect() as db:
            return {r[0]: json.loads(r[1]) for r in db.execute("SELECT content_hash,vector_json FROM vectors WHERE owner=? AND model_key=?", (owner, model_key))}

    def save_vector(self, owner, model_key, content_hash, vector):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?)", (owner, model_key, content_hash, json.dumps(vector)))
