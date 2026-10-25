"""Stable PDF theme catalogue and durable, platform/user-scoped preferences."""
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time


# Public numbers are permanent. Never derive them from sorting or renumber them.
THEMES = (
    ("1", "original", "原版", "现有原版"),
    ("2", "a", "马卡龙圆角", "原 A"),
    ("3", "g", "青瓷书签", "原 G"),
    ("4", "h", "靛蓝分栏", "原 H"),
    ("5", "jade", "青玉浮签", "旧预览 2"),
    ("6", "mono_double", "黑白双框", "替换候选 2"),
    ("7", "jewel", "宝石深色", "替换候选 3"),
    ("8", "mist", "雾紫双描", "旧预览 7"),
    ("9", "lake", "湖蓝色脊", "旧预览 8"),
)
BY_NUMBER = {row[0]: row for row in THEMES}
BY_STYLE = {row[1]: row for row in THEMES}
ALL_STYLES = tuple(BY_STYLE)
DEFAULT_STYLES = ("original", "a")
VALID_CHOICES = frozenset((*ALL_STYLES, "random", "default"))


def user_key(event):
    """One account across groups/private chats; never use a group/session ID."""
    if event is None:
        return None
    try:
        uid = str(event.get_sender_id() or "").strip()
    except Exception:
        return None
    if not uid or uid.lower() in ("unknown", "none", "0"):
        return None
    platform = ""
    for name in ("get_platform_name", "get_platform_id"):
        try:
            platform = str(getattr(event, name)() or "").strip()
        except Exception:
            continue
        if platform:
            break
    if not platform:
        platform = str(getattr(event, "unified_msg_origin", "") or "").split(":")[0]
    if not platform:
        return None
    # Avoid storing plain account identifiers in the preference database.
    identity = json.dumps([platform, uid], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


class ThemePreferences:
    """Short independent SQLite transactions survive reloads and concurrent writes."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / "pdf_theme_preferences.sqlite3"

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute(
            "CREATE TABLE IF NOT EXISTS preferences "
            "(user_key TEXT PRIMARY KEY, choice TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        return db

    def get(self, key):
        if not key:
            return "default"
        db = self._connect()
        try:
            row = db.execute("SELECT choice FROM preferences WHERE user_key=?", (key,)).fetchone()
            return row[0] if row and row[0] in VALID_CHOICES else "default"
        finally:
            db.close()

    def set(self, key, choice):
        if not key:
            raise ValueError("无法识别用户账号，未修改设置")
        if choice not in VALID_CHOICES:
            raise ValueError("未知的 PDF 主题")
        db = self._connect()
        try:
            with db:
                if choice == "default":
                    db.execute("DELETE FROM preferences WHERE user_key=?", (key,))
                else:
                    db.execute(
                        "INSERT INTO preferences VALUES (?, ?, ?) "
                        "ON CONFLICT(user_key) DO UPDATE SET choice=excluded.choice, "
                        "updated_at=excluded.updated_at", (key, choice, time.time()),
                    )
        finally:
            db.close()


def choose_style(choice):
    if choice in BY_STYLE:
        return choice
    return secrets.choice(ALL_STYLES if choice == "random" else DEFAULT_STYLES)


def choice_label(choice):
    if choice in BY_STYLE:
        number, _, name, _ = BY_STYLE[choice]
        return f"{number} · {name}"
    if choice == "random":
        return "随机：全部 9 种主题等概率，每份 PDF 抽取一次"
    return "默认：1 原版 / 2 马卡龙圆角，各 50%"


def parse_choice(text):
    value = str(text or "").strip()
    if value in BY_NUMBER:
        return BY_NUMBER[value][1]
    return {"随机": "random", "random": "random",
            "默认": "default", "default": "default", "重置": "default"}.get(value.lower())


def help_text(choice):
    rows = "\n".join(f"{n}. {name}" for n, _, name, _ in THEMES)
    return (
        f"你当前的 PDF 主题：{choice_label(choice)}\n\n{rows}\n\n"
        "/pdf主题 1～9：固定使用指定主题\n"
        "/pdf主题 随机：全部 9 种等概率随机\n"
        "/pdf主题 默认：恢复原版/马卡龙圆角各 50%\n"
        "/pdf主题概览：查看全部主题拼图\n"
        "只影响本人；跨群/私聊保存，重启后保留。"
        "适用于 /pdf、/spdf 及对应 Agent 工具，不影响图片或 MD。"
    )
