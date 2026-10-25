"""Refresh QQ image credentials through the message's own OneBot connection."""

import asyncio
import copy
import os
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class PdfImageInputError(RuntimeError):
    """An input image is unavailable; do not run models on an incomplete question."""


def qq_image_parts(url):
    try:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        if (
            parts.scheme in ("http", "https")
            and parts.hostname == "multimedia.nt.qq.com.cn"
            and parts.port in (None, 80, 443)
            and not parts.username
            and parts.path == "/download"
            and query.get("appid") in ("1406", "1407")
            and query.get("fileid")
        ):
            return parts, query
    except (TypeError, ValueError):
        pass
    return None


def image_cache_identity(url):
    qq = qq_image_parts(url)
    if qq:
        query = qq[1]
        return "qq:" + query["appid"] + ":" + query["fileid"]
    return url


def safe_image_error(error):
    # Exception text can contain signed URLs, so never log it verbatim here.
    code = getattr(error, "code", None)
    return f"HTTP {code}" if isinstance(code, int) else type(error).__name__


class QqImageRecoveryMixin:
    async def _pdf_prepare_agent_images(self, event):
        """Promote recovered QQ images before build_main_agent materializes them.

        The normal agent downloads attachments BEFORE on_llm_request. Keeping
        a PDF-only snapshot does not replace Image.url used by that download.
        """
        changes = []

        async def visit(components, depth=0):
            if depth > 4:
                raise PdfImageInputError("图片读取失败：引用图片的嵌套层数过多，请直接发送图片。")
            for component in components or []:
                kind = component.__class__.__name__.lower()
                if kind == "reply":
                    await visit(getattr(component, "chain", None), depth + 1)
                    continue
                if kind != "image":
                    continue
                ref = str(getattr(component, "url", None) or getattr(component, "file", None) or "")
                if not qq_image_parts(ref):
                    continue
                file_ref = str(getattr(component, "file", None) or "")
                local = self._pdf_ref_to_local_path(file_ref)
                source = file_ref if local and os.path.isfile(local) else ref
                stable = await self._pdf_snapshot_image_urls([source], event=event)
                if len(stable) != 1 or not stable[0].startswith("data:image/"):
                    raise PdfImageInputError("图片读取失败：未能固定 QQ 图片内容。")
                changes.append((component, stable[0]))

        await visit(getattr(getattr(event, "message_obj", None), "message", None))
        # Commit only after all attachments are ready; never proceed on a partial question.
        for component, stable in changes:
            component.url = stable
            component.file = stable
            if hasattr(component, "path"):
                component.path = ""
        return len(changes)

    async def _pdf_prepare_agent_request_images(self, event, req):
        """Cover generated quoted-image refs and image parts from conversation history."""
        normalized = {}

        async def resolve(ref):
            if not isinstance(ref, str) or not qq_image_parts(ref):
                return ref
            key = image_cache_identity(ref)
            if key not in normalized:
                stable = await self._pdf_snapshot_image_urls([ref], event=event)
                if len(stable) != 1 or not stable[0].startswith("data:image/"):
                    raise PdfImageInputError("图片读取失败：未能恢复请求中的 QQ 图片。")
                normalized[key] = stable[0]
            return normalized[key]

        async def normalize_parts(parts):
            if not isinstance(parts, list):
                return
            for part in parts:
                kind = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if kind != "image_url":
                    continue
                value = part.get("image_url") if isinstance(part, dict) else getattr(part, "image_url", None)
                if isinstance(value, dict):
                    value["url"] = await resolve(value.get("url"))
                elif isinstance(value, str):
                    stable = await resolve(value)
                    if isinstance(part, dict):
                        part["image_url"] = stable
                    else:
                        part.image_url = stable
                elif hasattr(value, "url"):
                    value.url = await resolve(value.url)

        images = [await resolve(ref) for ref in (getattr(req, "image_urls", None) or [])]
        contexts = copy.deepcopy(getattr(req, "contexts", None))
        if isinstance(contexts, list):
            for message in contexts:
                if isinstance(message, dict):
                    await normalize_parts(message.get("content"))
        extra = copy.deepcopy(getattr(req, "extra_user_content_parts", None))
        await normalize_parts(extra)
        if normalized:
            req.image_urls = images
            if contexts is not None:
                req.contexts = contexts
            if extra is not None:
                req.extra_user_content_parts = extra
        return len(normalized)

    async def _pdf_refresh_qq_image_url(self, url, event, timeout_sec=8):
        qq = qq_image_parts(url)
        bot = getattr(event, "bot", None)
        call = getattr(bot, "call_action", None)
        if not qq or not callable(call):
            raise PdfImageInputError("当前消息没有可用的 QQ 图片凭证刷新接口")

        parts, query = qq
        kind = "group" if query["appid"] == "1407" else "private"
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        self_id = raw.get("self_id") if hasattr(raw, "get") else None
        routing = {"self_id": self_id} if self_id else {}
        cache_key = (id(bot), str(self_id or ""))

        if not hasattr(self, "_pdf_qq_rkey_lock"):
            self._pdf_qq_rkey_lock = asyncio.Lock()
            self._pdf_qq_rkey_cache = {}
        async with self._pdf_qq_rkey_lock:
            cache = self._pdf_qq_rkey_cache
            now = time.monotonic()
            cached = cache.get(cache_key, {}).get(kind)
            rejected = query.get("rkey", "")
            if not cached or cached[1] <= now or cached[0] == rejected:
                keys = {}
                for action in ("get_rkey", "nc_get_rkey"):
                    try:
                        result = await asyncio.wait_for(
                            call(action, **routing), timeout=max(1, min(8, timeout_sec))
                        )
                        if isinstance(result, dict):
                            if result.get("status") == "failed":
                                continue
                            result = result.get("data", result)
                        if not isinstance(result, list):
                            continue
                        for item in result:
                            if not isinstance(item, dict):
                                continue
                            key_kind = {10: "private", 20: "group", "10": "private", "20": "group"}.get(
                                item.get("type"), item.get("type")
                            )
                            value = str(item.get("rkey") or "").removeprefix("&rkey=")
                            if key_kind not in ("private", "group") or not value:
                                continue
                            # Never trust the protocol client's long-lived URL cache.
                            # Cache our directly fetched credentials for at most 60s.
                            ttl = float(item.get("ttl", 60))
                            created = float(item.get("created_at", item.get("time", time.time())))
                            remaining = max(0, min(60, created + ttl - time.time() - 10))
                            keys[key_kind] = (value, time.monotonic() + remaining)
                        if kind in keys:
                            break
                    except Exception:
                        # The older action name remains a bounded compatibility fallback.
                        continue
                if kind not in keys:
                    raise PdfImageInputError("QQ 图片下载凭证刷新失败")
                if len(cache) >= 32:
                    cache.clear()
                cache[cache_key] = keys
                cached = keys[kind]

            pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "rkey"]
            pairs.append(("rkey", cached[0]))
            return urlunsplit(parts._replace(query=urlencode(pairs)))
