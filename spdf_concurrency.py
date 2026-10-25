"""Atomic, per-conversation Deepthink admission; independent of /pdf counters."""
import asyncio
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import logging
from pathlib import Path
import sqlite3
import time
import uuid

try:
    from .pdf_task_stats import task_scope, _process_owner
    from .spdf_schedule import SpdfAccessDenied
except ImportError:
    from pdf_task_stats import task_scope, _process_owner
    from spdf_schedule import SpdfAccessDenied

DEFAULT_LIMIT = 5
_current_slot = ContextVar('mathsolve_spdf_slot', default=None)
_log = logging.getLogger('astrbot')


def configured_limit(plugin):
    value = plugin._cfg('spdf_session_max_tasks', DEFAULT_LIMIT)
    if isinstance(value, bool) or not str(value).strip().isascii() or not str(value).strip().isdigit():
        raise SpdfAccessDenied('⚠️ Deepthink 并发上限配置有误，请在插件 WebUI 中填写大于等于 1 的整数。')
    limit = int(value)
    if limit < 1:
        raise SpdfAccessDenied('⚠️ Deepthink 并发上限配置有误，请在插件 WebUI 中填写大于等于 1 的整数。')
    return limit


class SpdfTaskLimiter:
    def __init__(self, path, owner=None):
        self.path = str(path)
        self.owner = _process_owner() if owner is None else owner
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS active(token TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL, started REAL NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS active_scope ON active(scope, owner)')
            # Same-process plugin reloads keep running jobs. A process restart
            # cannot keep any old jobs, so only those stale reservations expire.
            db.execute('DELETE FROM active WHERE owner != ?', (self.owner,))

    def connect(self):
        return sqlite3.connect(self.path, timeout=2)

    def acquire(self, scope, limit):
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            active = db.execute('SELECT COUNT(*) FROM active WHERE scope=? AND owner=?', (scope, self.owner)).fetchone()[0]
            if active >= limit:
                raise SpdfAccessDenied(f'⏳ 当前共 {active}/{limit} 个 Deepthink 任务进行中，已达到此会话的并发上限。请等待任务结束后再试。')
            token = uuid.uuid4().hex
            db.execute('INSERT INTO active VALUES (?, ?, ?, ?)', (token, scope, self.owner, time.time()))
        return token

    def count(self, scope):
        with closing(self.connect()) as db:
            return db.execute('SELECT COUNT(*) FROM active WHERE scope=? AND owner=?', (scope, self.owner)).fetchone()[0]

    def release(self, token):
        with closing(self.connect()) as db, db:
            db.execute('DELETE FROM active WHERE token=? AND owner=?', (token, self.owner))


def limiter_for(plugin):
    limiter = getattr(plugin, '_spdf_task_limiter', None)
    if limiter is None:
        limiter = SpdfTaskLimiter(Path(plugin.DATA_DIR) / 'spdf_active.sqlite3')
        plugin._spdf_task_limiter = limiter
    return limiter


@dataclass
class Slot:
    plugin: object
    event: object
    task: object
    limiter: object
    scope: str
    token: str
    limit: int
    active: bool = True


@contextmanager
def spdf_slot(plugin, event, admission=None):
    current = _current_slot.get()
    task = asyncio.current_task()
    if (current is not None and current.active and current.plugin is plugin
            and current.event is event and current.task is task):
        # Command -> solver is one job. Child tasks / other events cannot reuse
        # the reservation just because asyncio copied their ContextVars.
        yield current
        return
    plugin._spdf_require_access(event, admission)
    limit = configured_limit(plugin)
    try:
        scope = task_scope(event)
        limiter = limiter_for(plugin)
        token = limiter.acquire(scope, limit)
    except SpdfAccessDenied:
        raise
    except Exception as exc:
        _log.error('[mathsolve] Deepthink admission unavailable: %s', type(exc).__name__)
        raise SpdfAccessDenied('⚠️ Deepthink 任务计数暂不可用，请稍后重试。') from None
    slot = Slot(plugin, event, task, limiter, scope, token, limit)
    context_token = _current_slot.set(slot)
    try:
        yield slot
    finally:
        slot.active = False
        # No await: release still runs when cancellation interrupts a model,
        # compilation, message send, or a suspended command generator.
        try:
            limiter.release(token)
        finally:
            try:
                _current_slot.reset(context_token)
            except ValueError:
                # Async-generator finalization may run in a fresh context.
                # Its original context only retains an inactive, unusable slot.
                if _current_slot.get() is slot:
                    _current_slot.set(None)


def spdf_waiting_notice(plugin, event):
    text = '🧠 Deepthink 正在深度思考，将进行多角色推理与校验，并生成 PDF，请稍候…'
    slot = _current_slot.get()
    if slot is not None and slot.active and slot.plugin is plugin and slot.event is event:
        text += f'\n当前共 {slot.limiter.count(slot.scope)}/{configured_limit(plugin)} 个 Deepthink 任务进行中。'
    return text


def limited_spdf_command(fn):
    @wraps(fn)
    async def run(self, event, *args, **kwargs):
        try:
            with spdf_slot(self, event):
                work = fn(self, event, *args, **kwargs)
                try:
                    async for result in work:
                        yield result
                finally:
                    await work.aclose()
        except SpdfAccessDenied as exc:
            yield event.plain_result(str(exc))
    return run


def limited_spdf_generation(fn):
    @wraps(fn)
    async def run(self, problem_text, event, *args, **kwargs):
        with spdf_slot(self, event, kwargs.get('_admission')):
            return await fn(self, problem_text, event, *args, **kwargs)
    return run


def limited_spdf_tool(get_plugin, extract_context):
    def decorate(fn):
        @wraps(fn)
        async def run(self, context, **kwargs):
            plugin = get_plugin()
            _, event = extract_context(context)
            try:
                with spdf_slot(plugin, event):
                    return await fn(self, context, **kwargs)
            except SpdfAccessDenied as exc:
                return json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False)
        return run
    return decorate
