"""Per-group /pdf admission totals and live tasks. No prompts/user content stored."""

from datetime import datetime, timedelta, timezone
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import uuid

BEIJING_TZ = timezone(timedelta(hours=8))


def _process_owner():
    # Stable across plugin reloads, different after process/server restart.
    try:
        start = Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return f'{os.getpid()}:{boot}:{start}'
    except OSError:
        return str(os.getpid())


def task_scope(event):
    """Use the actual group ID, never a per-user conversation/session ID."""
    def call(name):
        try:
            value = getattr(event, name)()
            return str(value).strip() if value is not None else ''
        except Exception:
            return ''

    platform = call('get_platform_id') or call('get_platform_name')
    if not platform:
        platform = str(getattr(event, 'unified_msg_origin', '')).split(':')[0]
    group = call('get_group_id')
    if not group:
        group = str(getattr(getattr(event, 'message_obj', None), 'group_id', '') or '').strip()
    if group and group != '0':
        return json.dumps([platform, 'group', group], ensure_ascii=False)
    sender = call('get_sender_id')
    if not sender:
        raise ValueError('Cannot identify PDF task conversation')
    # Private chats must not expose another user's totals or task counts.
    return json.dumps([platform, 'private', sender], ensure_ascii=False)


class PdfTaskStats:
    def __init__(self, path, owner=None):
        self.path = str(path)
        self.owner = owner if owner is not None else _process_owner()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS daily (scope TEXT NOT NULL, day TEXT NOT NULL, total INTEGER NOT NULL, PRIMARY KEY(scope, day))')
            db.execute('CREATE TABLE IF NOT EXISTS active (token TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS active_scope ON active(scope, owner)')
            # AstrBot runs in one process. A restarted process has no surviving jobs.
            db.execute('DELETE FROM active WHERE owner != ?', (self.owner,))
            db.executescript('''
                CREATE TABLE IF NOT EXISTS usage_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage_jobs(token TEXT PRIMARY KEY, scope TEXT NOT NULL, kind TEXT NOT NULL, started REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS usage_jobs_scope_time ON usage_jobs(scope, started);
                CREATE TABLE IF NOT EXISTS usage_calls(id INTEGER PRIMARY KEY, scope TEXT NOT NULL, token TEXT NOT NULL, provider TEXT NOT NULL, started REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS usage_calls_scope_time ON usage_calls(scope, started);
                CREATE TABLE IF NOT EXISTS usage_groups(scope TEXT PRIMARY KEY, target TEXT NOT NULL, group_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS usage_delivery(scope TEXT NOT NULL, day TEXT NOT NULL, state TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(scope, day));
            ''')
            db.execute("INSERT OR IGNORE INTO usage_meta VALUES ('started', ?)", (str(datetime.now(BEIJING_TZ).timestamp()),))

    def _connect(self):
        # Fresh short-lived connection per operation; safe for concurrent callers.
        return sqlite3.connect(self.path, timeout=2)

    def begin(self, scope, now=None):
        now = now or datetime.now(BEIJING_TZ)
        if now.tzinfo is None:
            raise ValueError('A timezone-aware timestamp is required')
        day = now.astimezone(BEIJING_TZ).date().isoformat()
        token = uuid.uuid4().hex
        db = self._connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO daily(scope, day, total) VALUES (?, ?, 1) ON CONFLICT(scope, day) DO UPDATE SET total = total + 1', (scope, day))
                db.execute('INSERT INTO active(token, scope, owner) VALUES (?, ?, ?)', (token, scope, self.owner))
                db.execute('INSERT INTO usage_jobs VALUES (?, ?, ?, ?)', (token, scope, 'pdf', now.timestamp()))
                total = db.execute('SELECT total FROM daily WHERE scope = ? AND day = ?', (scope, day)).fetchone()[0]
                active = db.execute('SELECT COUNT(*) FROM active WHERE scope = ? AND owner = ?', (scope, self.owner)).fetchone()[0]
            return token, total, active
        finally:
            db.close()

    def record_job(self, scope, kind, now=None):
        token = uuid.uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO usage_jobs VALUES (?, ?, ?, ?)', (token, scope, kind, (now or datetime.now(BEIJING_TZ)).timestamp()))
        return token

    def record_call(self, scope, token, provider, now=None):
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO usage_calls(scope, token, provider, started) VALUES (?, ?, ?, ?)',
                       (scope, token, str(provider), (now or datetime.now(BEIJING_TZ)).timestamp()))

    def register_group(self, scope, target, group_id):
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO usage_groups VALUES (?, ?, ?) ON CONFLICT(scope) DO UPDATE SET target=excluded.target, group_id=excluded.group_id', (scope, target, group_id))

    def report(self, scope, day, now=None):
        """Half-open local-day windows; never invent hourly history from old totals."""
        now = now or datetime.now(BEIJING_TZ)
        start = datetime.combine(day, datetime.min.time(), BEIJING_TZ)
        end = start + timedelta(days=1)
        previous = start - timedelta(days=1)
        cutoff = min(end.timestamp(), now.timestamp())
        with closing(self._connect()) as db:
            db.execute('BEGIN')  # Consistent snapshot while model calls arrive.
            tracked_since = float(db.execute("SELECT value FROM usage_meta WHERE key='started'").fetchone()[0])
            jobs = db.execute('SELECT kind, started FROM usage_jobs WHERE scope=? AND started>=? AND started<?', (scope, previous.timestamp(), cutoff)).fetchall()
            models = db.execute('SELECT provider, COUNT(*) FROM usage_calls WHERE scope=? AND started>=? AND started<? GROUP BY provider ORDER BY COUNT(*) DESC, provider', (scope, start.timestamp(), cutoff)).fetchall()
            old_total = db.execute('SELECT total FROM daily WHERE scope=? AND day=?', (scope, day.isoformat())).fetchone()
        def series(origin, stop):
            counts = [None if origin.timestamp()+(h+1)*3600 <= tracked_since or origin.timestamp()+h*3600 >= stop else 0 for h in range(24)]
            for kind, at in jobs:
                if kind == 'pdf' and origin.timestamp() <= at < min(origin.timestamp()+86400, stop):
                    hour = int((at-origin.timestamp())//3600)
                    counts[hour] = (counts[hour] or 0)+1
            return counts
        current = series(start, cutoff)
        yesterday = series(previous, min(start.timestamp(), now.timestamp()))
        deep = sum(kind == 'spdf' and start.timestamp() <= at < cutoff for kind, at in jobs)
        pdf_count = sum(v or 0 for v in current)
        partial = tracked_since > start.timestamp()
        # Existing daily counters can be retained as a total, not converted to a curve.
        total = max(pdf_count, old_total[0] if old_total else 0)
        return dict(day=day.isoformat(), previous_day=previous.date().isoformat(), pdf=current,
                    previous=yesterday, pdf_total=total, timed_pdf_total=pdf_count,
                    deep_total=deep, models=[dict(name=p, count=n) for p, n in models],
                    tracked_since=datetime.fromtimestamp(tracked_since, BEIJING_TZ).strftime('%m-%d %H:%M'),
                    partial=partial, live=cutoff < end.timestamp(),
                    until=datetime.fromtimestamp(cutoff, BEIJING_TZ).strftime('%H:%M') if cutoff < end.timestamp() else '24:00')

    def delivery_groups(self):
        with closing(self._connect()) as db:
            return db.execute('SELECT scope, target, group_id FROM usage_groups ORDER BY scope').fetchall()

    def claim_delivery(self, scope, day, now=None):
        at = (now or datetime.now(BEIJING_TZ)).timestamp()
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state, updated FROM usage_delivery WHERE scope=? AND day=?', (scope, day)).fetchone()
            # Persistent lease avoids reload duplicates; failed sends retry after 10 min.
            if row and (row[0] == 'sent' or at-row[1] < 600):
                return False
            db.execute("INSERT INTO usage_delivery VALUES (?, ?, 'pending', ?) ON CONFLICT(scope, day) DO UPDATE SET state='pending', updated=excluded.updated", (scope, day, at))
        return True

    def mark_delivered(self, scope, day, now=None):
        with closing(self._connect()) as db, db:
            db.execute("UPDATE usage_delivery SET state='sent', updated=? WHERE scope=? AND day=?", ((now or datetime.now(BEIJING_TZ)).timestamp(), scope, day))

    def finish(self, token):
        db = self._connect()
        try:
            with db:
                db.execute('DELETE FROM active WHERE token = ? AND owner = ?', (token, self.owner))
        finally:
            db.close()


def waiting_notice(total, active):
    return f'⏳ 正在生成中，今天第 {total} 个解答任务。\n当前共有 {active} 个任务正在进行，请稍候……'
