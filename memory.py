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

    def _start_cache_cleaner(self):
        if not bool(self._cfg("cache_cleanup_enabled", True)):
            return
        if self._cache_cleaner_task and not self._cache_cleaner_task.done():
            return
        interval = self._cache_int_cfg("cache_cleanup_interval_sec", 3600)
        self._cache_cleaner_task = asyncio.create_task(self._clean_cache_files_loop(interval))

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

    async def _clean_cache_files_loop(self, interval_sec: int):
        while True:
            try:
                await asyncio.sleep(max(60, int(interval_sec or 3600)))
                if not bool(self._cfg("cache_cleanup_enabled", True)):
                    continue
                stats = self._clean_cache_files_once()
                deleted = stats.get("deleted", 0)
                if deleted:
                    logger.info(
                        "缓存文件清理完成: "
                        f"deleted={deleted} image_deleted={stats.get('image_deleted', 0)} "
                        f"pdf_deleted={stats.get('pdf_deleted', 0)} bytes={stats.get('bytes', 0)}"
                    )
            except Exception as e:
                logger.warning(f"缓存文件清理任务异常: {e}")

    @staticmethod
    def _cache_int_cfg_value(raw: Any, default: int) -> int:
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _cache_int_cfg(self, key: str, default: int) -> int:
        return self._cache_int_cfg_value(self._cfg(key, default), default)

    @staticmethod
    def _cache_is_within_dir(path: str, base_dir: str) -> bool:
        try:
            p = Path(path).resolve()
            b = Path(base_dir).resolve()
            p.relative_to(b)
            return True
        except Exception:
            return False

    def _cache_scan_top_level_files(self, base_dir: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if not base_dir or not os.path.isdir(base_dir):
                return out
            base_real = str(Path(base_dir).resolve())
            with os.scandir(base_real) as it:
                for ent in it:
                    try:
                        if not ent.is_file(follow_symlinks=False):
                            continue
                        fp = str(Path(ent.path).resolve())
                        if not self._cache_is_within_dir(fp, base_real):
                            continue
                        st = ent.stat(follow_symlinks=False)
                        out.append({"path": fp, "mtime": float(st.st_mtime), "size": int(st.st_size)})
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def _cache_delete_file(self, path: str, base_dir: str) -> Tuple[bool, int]:
        try:
            if not self._cache_is_within_dir(path, base_dir):
                return False, 0
            if not os.path.isfile(path):
                return False, 0
            size = int(os.path.getsize(path))
            os.remove(path)
            return True, size
        except Exception:
            return False, 0

    def _clean_one_cache_dir(self, base_dir: str, ttl_sec: int, max_files: int, protect_recent_sec: int) -> Dict[str, int]:
        now = time.time()
        files = self._cache_scan_top_level_files(base_dir)
        deleted_paths = set()
        deleted = 0
        deleted_bytes = 0

        def can_delete(item: Dict[str, Any]) -> bool:
            try:
                age = now - float(item.get("mtime", 0) or 0)
                return age >= max(0, protect_recent_sec)
            except Exception:
                return False

        if ttl_sec > 0:
            for item in files:
                try:
                    age = now - float(item.get("mtime", 0) or 0)
                    if age <= ttl_sec or not can_delete(item):
                        continue
                    ok, size = self._cache_delete_file(str(item.get("path", "")), base_dir)
                    if ok:
                        deleted_paths.add(str(item.get("path", "")))
                        deleted += 1
                        deleted_bytes += size
                except Exception:
                    continue

        remaining = [x for x in files if str(x.get("path", "")) not in deleted_paths]
        if max_files > 0 and len(remaining) > max_files:
            remaining.sort(key=lambda x: float(x.get("mtime", 0) or 0))
            overflow = len(remaining) - max_files
            for item in remaining:
                if overflow <= 0:
                    break
                if not can_delete(item):
                    continue
                ok, size = self._cache_delete_file(str(item.get("path", "")), base_dir)
                if ok:
                    overflow -= 1
                    deleted += 1
                    deleted_bytes += size

        return {"deleted": deleted, "bytes": deleted_bytes}

    def _clean_cache_files_once(self) -> Dict[str, int]:
        protect_recent_sec = self._cache_int_cfg("cache_cleanup_protect_recent_sec", 300)
        image_stats = self._clean_one_cache_dir(
            self.IMAGE_CACHE_DIR,
            ttl_sec=self._cache_int_cfg("image_cache_ttl_sec", 86400),
            max_files=self._cache_int_cfg("image_cache_max_files", 1000),
            protect_recent_sec=protect_recent_sec,
        )
        pdf_stats = self._clean_one_cache_dir(
            self.PDF_CACHE_DIR,
            ttl_sec=self._cache_int_cfg("pdf_cache_ttl_sec", 604800),
            max_files=self._cache_int_cfg("pdf_cache_max_files", 300),
            protect_recent_sec=protect_recent_sec,
        )
        return {
            "image_deleted": int(image_stats.get("deleted", 0)),
            "pdf_deleted": int(pdf_stats.get("deleted", 0)),
            "deleted": int(image_stats.get("deleted", 0)) + int(pdf_stats.get("deleted", 0)),
            "bytes": int(image_stats.get("bytes", 0)) + int(pdf_stats.get("bytes", 0)),
        }
