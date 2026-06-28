# -*- coding: utf-8 -*-
try:
    from .shared import *
except ImportError:
    from shared import *


class MemoryMixin:
    """Session cleanup helpers for the mathsolve plugin."""

    def _start_session_cleaner(self):
        if self._session_cleaner_task and not self._session_cleaner_task.done():
            return
        interval = int(self._cfg("session_cleanup_interval_sec", 3600) or 3600)
        self._session_cleaner_task = asyncio.create_task(self._clean_expired_sessions_loop(interval))

    async def _clean_expired_sessions_loop(self, interval_sec: int):
        ttl = int(self._cfg("session_ttl_sec", 86400) or 86400)
        while True:
            try:
                await asyncio.sleep(max(60, interval_sec))
                now = time.time()
                async with self._state_lock:
                    expired = [
                        k for k, v in MATH_SESSION_STATE.items()
                        if now - float(v.get("last_active_ts", v.get("last_image_ts", 0) or 0)) > ttl
                    ]
                    for k in expired:
                        del MATH_SESSION_STATE[k]
                if expired:
                    logger.info(f"Session 清理完成: 删除 {len(expired)} 个过期会话")
            except Exception as e:
                logger.warning(f"Session 清理任务异常: {e}")
