# -*- coding: utf-8 -*-
try:
    from .shared import *
    from .pdf_styles import randomized_pdf_style
    from .pdf_prompt_preferences import PdfPromptPreferences, user_key, preference_context, compose_pdf_prompt
    from .usage_tracking import tracked_generation, record_model_call
    from .token_usage import generate_with_token_usage
    from .qq_image_recovery import (PdfImageInputError, QqImageRecoveryMixin,
                                    image_cache_identity, qq_image_parts, safe_image_error)
except ImportError:
    from shared import *
    from pdf_styles import randomized_pdf_style
    from pdf_prompt_preferences import PdfPromptPreferences, user_key, preference_context, compose_pdf_prompt
    from usage_tracking import tracked_generation, record_model_call
    from token_usage import generate_with_token_usage
    from qq_image_recovery import (PdfImageInputError, QqImageRecoveryMixin,
                                   image_cache_identity, qq_image_parts, safe_image_error)


class PdfMixin(QqImageRecoveryMixin):
    """PDF generation, provider fallback, and TeX compilation helpers."""

    @staticmethod
    def _pdf_parse_timeout_value(value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            s = str(value if value is not None else "").strip()
            if not s:
                return default
            return max(0, int(float(s)))
        except Exception:
            return default

    def _pdf_default_model_timeout_sec(self) -> int:
        timeout_sec = self._pdf_parse_timeout_value(self._cfg("pdf_provider_timeout_sec", 180), 180)
        return 180 if timeout_sec is None else timeout_sec

    def _pdf_parse_timeout_overrides(self) -> Dict[str, int]:
        """解析 provider_id=timeout_sec 形式的超时覆盖配置。"""
        raw = str(self._cfg("pdf_provider_timeout_overrides", "") or "")
        out: Dict[str, int] = {}
        for item in re.split(r"[,;\s]+", raw):
            item = (item or "").strip()
            if not item:
                continue
            sep = "=" if "=" in item else ("|" if "|" in item else "")
            if not sep:
                continue
            pid, timeout_txt = item.split(sep, 1)
            pid = (pid or "").strip()
            timeout_sec = self._pdf_parse_timeout_value(timeout_txt, None)
            if pid and timeout_sec is not None:
                out[pid] = timeout_sec
        return out

    def _pdf_timeout_for_provider(self, provider_id: str) -> int:
        pid = str(provider_id or "").strip()
        overrides = self._pdf_parse_timeout_overrides()
        return overrides.get(pid, self._pdf_default_model_timeout_sec())

    async def _pdf_maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value) or inspect.isawaitable(value):
            return await value
        return value

    def _pdf_extract_model_name_from_obj(self, obj: Any, depth: int = 0) -> str:
        """从 AstrBot provider 对象/配置中尽量取真实模型名。"""
        if obj is None or depth > 3:
            return ""

        model_keys = (
            "model", "model_name", "model_id", "modelName", "modelId",
            "llm_model", "chat_model", "selected_model", "default_model",
        )
        nested_keys = (
            "config", "provider_config", "provider_settings", "settings",
            "options", "kwargs", "extra", "metadata", "meta",
        )

        if isinstance(obj, dict):
            for key in model_keys:
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for key in nested_keys:
                val = obj.get(key)
                found = self._pdf_extract_model_name_from_obj(val, depth + 1)
                if found:
                    return found
            return ""

        for key in model_keys:
            try:
                val = getattr(obj, key, None)
            except Exception:
                val = None
            if isinstance(val, str) and val.strip():
                return val.strip()

        for key in nested_keys:
            try:
                val = getattr(obj, key, None)
            except Exception:
                val = None
            found = self._pdf_extract_model_name_from_obj(val, depth + 1)
            if found:
                return found

        try:
            d = getattr(obj, "__dict__", None)
        except Exception:
            d = None
        if isinstance(d, dict) and d is not obj:
            return self._pdf_extract_model_name_from_obj(d, depth + 1)
        return ""

    async def _pdf_find_provider_obj(self, provider_id: str) -> Any:
        pid = str(provider_id or "").strip()
        if not pid:
            return None

        roots = [getattr(self, "context", None)]
        try:
            pm = getattr(self.context, "provider_manager", None)
            if pm is not None:
                roots.insert(0, pm)
        except Exception:
            pass

        method_names = (
            "get_provider", "get_provider_by_id", "get_provider_by_provider_id",
            "get_chat_provider", "get_chat_provider_by_id", "get_provider_instance",
            "get_provider_inst", "get_provider_by_name", "get_provider_by_key",
        )
        for root in roots:
            if root is None:
                continue
            for name in method_names:
                fn = getattr(root, name, None)
                if not callable(fn):
                    continue
                try:
                    obj = await self._pdf_maybe_await(fn(pid))
                    if obj is not None:
                        return obj
                except Exception:
                    continue

        dict_attrs = (
            "providers", "provider_map", "provider_dict", "provider_insts",
            "provider_instances", "chat_providers", "llm_providers",
            "provider_settings", "configs", "config",
        )
        id_attrs = ("id", "provider_id", "providerId", "name", "key")

        def _matches_provider_id(item: Any) -> bool:
            try:
                if isinstance(item, dict):
                    return any(str(item.get(k, "") or "").strip() == pid for k in id_attrs)
                return any(str(getattr(item, k, "") or "").strip() == pid for k in id_attrs)
            except Exception:
                return False

        for root in roots:
            if root is None:
                continue
            for attr in dict_attrs:
                try:
                    store = getattr(root, attr, None)
                except Exception:
                    store = None
                if isinstance(store, dict):
                    if pid in store:
                        return store.get(pid)
                    for item in store.values():
                        if _matches_provider_id(item):
                            return item
                elif isinstance(store, (list, tuple)):
                    for item in store:
                        if _matches_provider_id(item):
                            return item
        return None

    async def _pdf_model_label_for_filename(self, provider_id: str) -> str:
        model_name = ""
        try:
            provider_obj = await self._pdf_find_provider_obj(provider_id)
            model_name = self._pdf_extract_model_name_from_obj(provider_obj)
        except Exception:
            model_name = ""
        return _sanitize_model_label_for_filename(model_name or provider_id)

    def _pdf_preflight_timeout_sec(self) -> int:
        timeout_sec = self._pdf_parse_timeout_value(self._cfg("pdf_provider_preflight_timeout_sec", 8), 8)
        return 8 if timeout_sec is None else timeout_sec

    def _pdf_preflight_cache_ttl_sec(self) -> int:
        ttl_sec = self._pdf_parse_timeout_value(self._cfg("pdf_provider_preflight_cache_ttl_sec", 300), 300)
        return 300 if ttl_sec is None else ttl_sec

    @staticmethod
    def _pdf_is_data_image_ref(url: str) -> bool:
        return str(url or "").strip().lower().startswith("data:image/")

    @staticmethod
    def _pdf_is_base64_image_ref(url: str) -> bool:
        u = str(url or "").strip()
        return u.startswith("base64://")

    @staticmethod
    def _pdf_describe_image_ref(url: str) -> str:
        u = str(url or "")
        if u.startswith("data:"):
            head, _, payload = u.partition(",")
            return f"data_uri:{head[:40]} payload_len={len(payload)}"
        if u.startswith("base64://"):
            return f"base64_payload_len={len(u) - len('base64://')}"
        if u.lower().startswith(("http://", "https://")):
            try:
                parsed = urllib.parse.urlparse(u)
                return f"{parsed.scheme}://{parsed.netloc}{urllib.parse.unquote(parsed.path)}"
            except Exception:
                return f"http_url_len={len(u)}"
        return str(Path(u).name or u)[:200]

    @staticmethod
    def _pdf_is_http_image_url(url: str) -> bool:
        u = str(url or "").strip()
        return u.lower().startswith(("http://", "https://"))

    @staticmethod
    def _pdf_file_uri_to_path(file_uri: str) -> str:
        parsed = urllib.parse.urlparse(str(file_uri or ""))
        if parsed.scheme.lower() != "file":
            return str(file_uri or "")
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            raw_path = f"//{parsed.netloc}{parsed.path}"
        else:
            raw_path = parsed.path
        path = urllib.request.url2pathname(raw_path)
        if re.match(r"^/[A-Za-z]:[/\\\\]", path):
            path = path[1:]
        return path

    @staticmethod
    def _pdf_ref_to_local_path(url: str) -> str:
        u = str(url or "").strip()
        if not u:
            return ""
        if u.lower().startswith("file://"):
            return PdfMixin._pdf_file_uri_to_path(u)
        if u.lower().startswith(("http://", "https://", "data:", "base64://")):
            return ""
        return u

    def _pdf_image_path_to_data_uri(self, path: str) -> str:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            raise RuntimeError("empty local image file")
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        if not str(mime).startswith("image/"):
            mime = "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")

    def _pdf_copy_local_image_sync(self, url: str) -> str:
        src = self._pdf_ref_to_local_path(url)
        if not src:
            raise RuntimeError("not a local image path")
        src = os.path.normpath(src)
        for idx in range(15):
            if os.path.isfile(src) and os.path.getsize(src) > 0:
                break
            if idx < 14:
                time.sleep(0.2)
        if not os.path.isfile(src):
            raise RuntimeError(f"local image missing: {src}")
        if os.path.getsize(src) <= 0:
            raise RuntimeError(f"local image empty: {src}")
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
            ext = ".jpg"
        os.makedirs(self.IMAGE_CACHE_DIR, exist_ok=True)
        stat = os.stat(src)
        key = hashlib.sha1(f"{src}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8", "ignore")).hexdigest()
        dst = os.path.join(self.IMAGE_CACHE_DIR, f"pdf_input_{key}{ext}")
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        return dst

    async def _pdf_get_event_image_inputs(self, event: AstrMessageEvent,
                                          defer_errors: bool = False) -> List[str]:
        """每个图片组件只保留一份输入；坏链接只能留待命令重试，不能交给模型。"""
        cached = getattr(event, "_mathsolve_pdf_image_inputs", None)
        if cached is not None:
            return list(cached)
        refs: List[str] = []
        all_ready = True
        msg_obj = getattr(event, "message_obj", None)
        for comp in (getattr(msg_obj, "message", None) or []):
            if not (isinstance(comp, Image) or comp.__class__.__name__.lower() == "image"):
                continue
            candidates = []
            file_ref = str(getattr(comp, "file", "") or "").strip()
            url_ref = str(getattr(comp, "url", "") or "").strip()
            for value in (file_ref, url_ref):
                if not value or value in candidates:
                    continue
                local = self._pdf_ref_to_local_path(value)
                if not local or os.path.isfile(local):
                    candidates.append(value)
            # Prefer existing local bytes over another attempt at an expired URL.
            candidates.sort(key=self._pdf_is_http_image_url)
            ready = None
            for value in candidates:
                try:
                    ready = await self._pdf_snapshot_image_urls([value], event=event)
                    break
                except PdfImageInputError:
                    continue
            if ready is None and not candidates and hasattr(comp, "convert_to_file_path"):
                try:
                    path = await asyncio.wait_for(comp.convert_to_file_path(), timeout=15)
                    ready = await self._pdf_snapshot_image_urls([path], event=event)
                except Exception as exc:
                    logger.warning(f"/pdf 图片组件读取失败: {safe_image_error(exc)}")
            if ready is None:
                all_ready = False
                if not defer_errors:
                    raise PdfImageInputError("图片读取失败：已尝试恢复图片输入，但仍无法读取。请重新上传原图后重试；本次未调用模型。")
                # Overwrite the user's previous image with THIS failed input; otherwise
                # a following bare /pdf could silently answer the previous question.
                ready = [url_ref or file_ref or "missing-image-input"]
            refs.extend(value for value in ready if value not in refs)
        if all_ready:
            setattr(event, "_mathsolve_pdf_image_inputs", tuple(refs))
        return refs

    @staticmethod
    def _pdf_guess_image_ext(url: str, content_type: str = "") -> str:
        ct = str(content_type or "").split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(ct) if ct else ""
        if ext:
            return ".jpg" if ext == ".jpe" else ext
        try:
            path = urllib.parse.urlparse(str(url or "")).path
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                return ext
        except Exception:
            pass
        return ".jpg"

    def _pdf_download_image_sync(self, url: str, timeout_sec: int, max_bytes: int) -> str:
        key = hashlib.sha256(image_cache_identity(str(url)).encode()).hexdigest()
        os.makedirs(self.IMAGE_CACHE_DIR, exist_ok=True)
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"):
            cached = os.path.join(self.IMAGE_CACHE_DIR, f"pdf_input_{key}{ext}")
            if os.path.isfile(cached) and 0 < os.path.getsize(cached) <= max_bytes:
                return cached
        req = urllib.request.Request(
            str(url),
            headers={
                "User-Agent": "Mozilla/5.0 AstrBot mathsolve pdf image snapshot",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec or 15))) as resp:
            content_length = str(resp.headers.get("Content-Length") or "").strip()
            if content_length and int(content_length) > max_bytes:
                raise RuntimeError(f"image too large: {content_length} bytes")
            content_type = str(resp.headers.get("Content-Type") or "")
            if content_type.split(";", 1)[0].strip().lower() in ("application/json", "text/html", "text/plain"):
                raise RuntimeError("download response is not an image")
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise RuntimeError(f"image too large: >{max_bytes} bytes")
        if not data:
            raise RuntimeError("empty image response")
        ext = self._pdf_guess_image_ext(url, content_type)
        path = os.path.join(self.IMAGE_CACHE_DIR, f"pdf_input_{key}{ext}")
        fd, pending = tempfile.mkstemp(prefix="pdf_input_", suffix=".part", dir=self.IMAGE_CACHE_DIR)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(pending, path)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
        return path

    async def _pdf_snapshot_image_urls(self, image_urls: Optional[List[str]],
                                      event: Optional[AstrMessageEvent] = None) -> List[str]:
        """在 /pdf 模型链开始前固定图片输入，并转成 data URI 交给视觉模型。"""
        # Grading's input images are transient, including natural-language
        # Agent calls prepared before the grading tool itself runs.
        if event is not None and not getattr(self, "_grading_snapshot_proxy", False):
            try:
                from .grading import grading_intent
            except ImportError:
                from grading import grading_intent
            if grading_intent(event):
                import copy
                with tempfile.TemporaryDirectory(prefix="mathsolve_grade_snapshot_") as temp:
                    proxy = copy.copy(self)
                    proxy.IMAGE_CACHE_DIR = temp
                    proxy._grading_snapshot_proxy = True
                    return await proxy._pdf_snapshot_image_urls(image_urls, event=event)
        urls = [str(u or "").strip() for u in (image_urls or []) if str(u or "").strip()]
        if not urls:
            return urls

        timeout_sec = int(self._cfg("pdf_snapshot_image_timeout_sec", 15) or 15)
        max_bytes = int(self._cfg("pdf_snapshot_image_max_bytes", 25 * 1024 * 1024) or (25 * 1024 * 1024))
        snapshot_enabled = bool(self._cfg("pdf_snapshot_images_before_generation", True))
        loop = asyncio.get_running_loop()
        out: List[str] = []
        for u in dict.fromkeys(urls):
            # QQ's expiring URLs must be resolved even if general URL snapshots are disabled.
            if not snapshot_enabled and self._pdf_is_http_image_url(u) and not qq_image_parts(u):
                out.append(u)
                continue
            if self._pdf_is_data_image_ref(u):
                out.append(u)
                continue
            if self._pdf_is_base64_image_ref(u):
                payload = u[len("base64://"):].strip()
                if payload:
                    out.append("data:image/jpeg;base64," + payload)
                else:
                    raise PdfImageInputError("图片读取失败：收到空图片，请重新上传原图。")
                continue
            try:
                if self._pdf_is_http_image_url(u):
                    try:
                        if getattr(self, "_grading_snapshot_proxy", False):
                            try:
                                from .grading import finish_io
                            except ImportError:
                                from grading import finish_io
                            local_path = await finish_io(self._pdf_download_image_sync, u, timeout_sec, max_bytes)
                        else:
                            local_path = await loop.run_in_executor(
                                None, lambda url=u: self._pdf_download_image_sync(url, timeout_sec, max_bytes),
                            )
                    except Exception as download_error:
                        if getattr(download_error, "code", None) not in (400, 401, 403) or not qq_image_parts(u):
                            raise
                        refreshed = await self._pdf_refresh_qq_image_url(u, event, timeout_sec)
                        if getattr(self, "_grading_snapshot_proxy", False):
                            local_path = await finish_io(self._pdf_download_image_sync, refreshed, timeout_sec, max_bytes)
                        else:
                            local_path = await loop.run_in_executor(
                                None, lambda url=refreshed: self._pdf_download_image_sync(url, timeout_sec, max_bytes),
                            )
                        logger.info("/pdf QQ 图片凭证已自动刷新，图片下载恢复")
                else:
                    if getattr(self, "_grading_snapshot_proxy", False):
                        try:
                            from .grading import finish_io
                        except ImportError:
                            from grading import finish_io
                        local_path = await finish_io(self._pdf_copy_local_image_sync, u)
                    else:
                        local_path = await loop.run_in_executor(
                            None,
                            lambda url=u: self._pdf_copy_local_image_sync(url),
                        )
                data_uri = self._pdf_image_path_to_data_uri(local_path)
                if data_uri not in out:
                    out.append(data_uri)
                logger.info(f"/pdf 图片已快照为 data URI: source={self._pdf_describe_image_ref(u)} local_path={local_path}")
            except Exception as e:
                logger.warning(f"/pdf 图片输入失败，已停止生成: source={self._pdf_describe_image_ref(u)} error={safe_image_error(e)}")
                raise PdfImageInputError("图片读取失败：图片无法下载或读取，已停止生成以免漏题。请重新上传原图后重试；这不是模型超时。") from e
        return out

    def _pdf_parse_provider_entry(self, entry: str) -> Tuple[str, Optional[int]]:
        """解析候补模型条目：provider_id 或 provider_id=timeout_sec。"""
        item = str(entry or "").strip()
        if not item:
            return "", None
        sep = "=" if "=" in item else ("|" if "|" in item else "")
        if not sep:
            return item, None
        pid, timeout_txt = item.split(sep, 1)
        timeout_sec = self._pdf_parse_timeout_value(timeout_txt, None)
        return (pid or "").strip(), timeout_sec

    @staticmethod
    def _pdf_requests_gpt_model(problem_text: str) -> bool:
        """用户输入中出现 gpt（大小写不敏感）时请求 GPT 专用模型。"""
        return re.search(r"gpt", str(problem_text or ""), flags=re.I) is not None

    def _pdf_compact_enabled_for_slot(self, key: str, default: bool = True) -> bool:
        """读取模型槽位的紧凑提示开关，并兼容旧版全局配置。"""
        missing = object()
        value = self._cfg(key, missing)
        if value is missing:
            value = self._cfg("pdf_enable_compact_prompt_rules", default)
        return bool(value)

    def _pdf_build_provider_chain(
            self,
            primary_provider_id: str,
            current_provider_id: str,
            routed_provider_id: str = "",
            routed_timeout_sec: Optional[int] = None,
            routed_compact_prompt_rules: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        构建 /pdf 生成模型链。
        关键词指定模型优先；随后是 pdf_provider_id（留空则当前会话模型）和候补模型槽位。
        """
        default_timeout = self._pdf_default_model_timeout_sec()
        overrides = self._pdf_parse_timeout_overrides()
        chain: List[Dict[str, Any]] = []
        seen = set()

        def add_provider(
                pid: str,
                timeout_sec: Optional[int] = None,
                compact_prompt_rules: Optional[bool] = None,
        ) -> None:
            p = str(pid or "").strip()
            if not p or p in seen:
                return
            seen.add(p)
            final_timeout = timeout_sec
            if final_timeout is None:
                final_timeout = overrides.get(p, default_timeout)
            if compact_prompt_rules is None:
                compact_prompt_rules = bool(self._cfg("pdf_enable_compact_prompt_rules", True))
            chain.append({
                "provider_id": p,
                "timeout_sec": max(0, int(final_timeout or 0)),
                "compact_prompt_rules": bool(compact_prompt_rules),
            })

        add_provider(
            routed_provider_id,
            routed_timeout_sec,
            routed_compact_prompt_rules,
        )

        add_provider(
            str(primary_provider_id or current_provider_id or "").strip(),
            compact_prompt_rules=self._pdf_compact_enabled_for_slot(
                "pdf_provider_enable_compact_prompt_rules",
            ),
        )

        for i in range(1, 6):
            pid = str(self._cfg(f"pdf_fallback_provider_id_{i}", "") or "").strip()
            timeout_sec = self._pdf_parse_timeout_value(
                    self._cfg(f"pdf_fallback_provider_timeout_sec_{i}", None),
                    None,
            )
            add_provider(
                pid,
                timeout_sec,
                self._pdf_compact_enabled_for_slot(
                    f"pdf_fallback_provider_enable_compact_prompt_rules_{i}",
                ),
            )

        # 兼容旧版文本配置：若用户配置里还残留 pdf_fallback_provider_ids，也继续读取。
        fallback_raw = str(self._cfg("pdf_fallback_provider_ids", "") or "")
        for item in re.split(r"[,;\s]+", fallback_raw):
            pid, timeout_sec = self._pdf_parse_provider_entry(item)
            add_provider(pid, timeout_sec)

        return chain

    @staticmethod
    def _pdf_prioritize_provider_chain(
            provider_chain: List[Dict[str, Any]],
            preferred_provider_id: str,
    ) -> List[Dict[str, Any]]:
        preferred = str(preferred_provider_id or "").strip()
        if not preferred:
            return list(provider_chain or [])
        head = [x for x in (provider_chain or []) if str(x.get("provider_id", "")).strip() == preferred]
        tail = [x for x in (provider_chain or []) if str(x.get("provider_id", "")).strip() != preferred]
        return head + tail

    def _pdf_set_preflight_cache(self, provider_id: str, ok: bool, error: str = "") -> None:
        pid = str(provider_id or "").strip()
        if not pid:
            return
        try:
            if not ok:
                # 失败只影响当前尝试，不做负缓存；否则临时抖动会让模型在 TTL 内持续被跳过。
                self._pdf_provider_preflight_cache.pop(pid, None)
                return
            self._pdf_provider_preflight_cache[pid] = {
                "ok": True,
                "ts": time.time(),
                "error": "",
            }
        except Exception:
            pass

    def _pdf_get_preflight_cache(self, provider_id: str) -> Optional[Dict[str, Any]]:
        pid = str(provider_id or "").strip()
        if not pid:
            return None
        ttl_sec = self._pdf_preflight_cache_ttl_sec()
        if ttl_sec <= 0:
            return None
        try:
            item = self._pdf_provider_preflight_cache.get(pid)
            if not isinstance(item, dict):
                return None
            if not bool(item.get("ok", False)):
                # 兼容旧版本可能留下的失败缓存，避免继续误跳过该模型。
                self._pdf_provider_preflight_cache.pop(pid, None)
                return None
            ts = float(item.get("ts", 0) or 0)
            if (time.time() - ts) > ttl_sec:
                self._pdf_provider_preflight_cache.pop(pid, None)
                return None
            return item
        except Exception:
            return None

    async def _pdf_preflight_provider(self, provider_id: str, purpose: str = "") -> Tuple[bool, str]:
        """正式生成前用极短 prompt 检查 provider 连通性；结果短时间缓存。"""
        pid = str(provider_id or "").strip()
        if not pid or (not bool(self._cfg("pdf_enable_provider_preflight", True))):
            return True, ""

        cached = self._pdf_get_preflight_cache(pid)
        if cached is not None:
            ok = bool(cached.get("ok", False))
            return ok, str(cached.get("error", "") or "")

        timeout_sec = self._pdf_preflight_timeout_sec()
        prompt = "ping\n请只回复 OK，不要输出其它内容。"
        try:
            call = generate_with_token_usage(self,
                chat_provider_id=pid,
                prompt=prompt,
                image_urls=[],
            )
            resp = await asyncio.wait_for(call, timeout=timeout_sec) if timeout_sec > 0 else await call
            txt = (
                    getattr(resp, "completion_text", None)
                    or getattr(resp, "completion", None)
                    or getattr(resp, "text", None)
                    or ""
            )
            if str(txt or "").strip():
                self._pdf_set_preflight_cache(pid, True)
                return True, ""
            err = "preflight_empty_completion"
            self._pdf_set_preflight_cache(pid, False, err)
            return False, err
        except asyncio.TimeoutError:
            err = f"preflight timeout after {timeout_sec}s"
            self._pdf_set_preflight_cache(pid, False, err)
            logger.warning(f"/pdf 模型预检超时，尝试下一个: purpose={purpose or '-'} provider_id={pid} timeout={timeout_sec}s")
            return False, err
        except Exception as e:
            err = str(e)
            self._pdf_set_preflight_cache(pid, False, err)
            logger.warning(f"/pdf 模型预检失败，尝试下一个: purpose={purpose or '-'} provider_id={pid}: {err}")
            return False, err

    async def _pdf_llm_generate_with_fallback(
            self,
            provider_chain: List[Dict[str, Any]],
            prompt: Any,
            image_urls: Optional[List[str]] = None,
            purpose: str = "",
    ) -> Tuple[str, str, List[str]]:
        """按顺序尝试 /pdf 模型链；单个模型超时/失败/空输出时自动切到下一个。"""
        if not provider_chain:
            raise RuntimeError("未能获取 chat_provider_id")

        failures: List[Tuple[str, str]] = []
        attempted_provider_ids: List[str] = []
        image_error_count = 0
        generation_failure_count = 0
        last_exc: Optional[BaseException] = None

        for idx, spec in enumerate(provider_chain):
            pid = str(spec.get("provider_id", "") or "").strip()
            if not pid:
                continue
            attempted_provider_ids.append(pid)
            timeout_sec = self._pdf_parse_timeout_value(spec.get("timeout_sec"), self._pdf_default_model_timeout_sec())
            timeout_sec = 0 if timeout_sec is None else timeout_sec
            preflight_ok, preflight_err = await self._pdf_preflight_provider(pid, purpose=purpose)
            if not preflight_ok:
                last_exc = RuntimeError(f"provider_id={pid} preflight failed: {preflight_err}")
                failures.append((pid, str(last_exc)))
                continue
            try:
                model_prompt = prompt(spec) if callable(prompt) else prompt
                record_model_call(pid)
                call = generate_with_token_usage(self,
                    chat_provider_id=pid,
                    prompt=str(model_prompt or ""),
                    image_urls=image_urls if image_urls else [],
                )
                llm_resp = await asyncio.wait_for(call, timeout=timeout_sec) if timeout_sec > 0 else await call
                raw = (
                        getattr(llm_resp, "completion_text", None)
                        or getattr(llm_resp, "completion", None)
                        or getattr(llm_resp, "text", None)
                        or ""
                )
                raw = (raw or "").strip()
                if raw:
                    self._pdf_set_preflight_cache(pid, True)
                    if idx > 0:
                        logger.info(f"/pdf 模型回退成功: purpose={purpose or '-'} provider_id={pid}")
                    return raw, pid, attempted_provider_ids
                raise RuntimeError("empty_completion")
            except asyncio.TimeoutError as e:
                generation_failure_count += 1
                last_exc = RuntimeError(f"provider_id={pid} timeout after {timeout_sec}s")
                self._pdf_set_preflight_cache(pid, False, str(last_exc))
                failures.append((pid, str(last_exc)))
                logger.warning(f"/pdf 模型调用超时，尝试下一个: purpose={purpose or '-'} provider_id={pid} timeout={timeout_sec}s")
            except Exception as e:
                last_exc = e
                err_str = str(e)
                generation_failure_count += 1
                if "failed to download file from" in err_str.lower():
                    raise PdfImageInputError("图片读取失败：模型调用前下载图片失败，已停止轮换模型。请重新上传原图后重试。") from e
                if any(marker in err_str.lower() for marker in ("process input image", "image invalid", "invalid image")):
                    image_error_count += 1
                else:
                    self._pdf_set_preflight_cache(pid, False, err_str)
                failures.append((pid, err_str))
                logger.warning(f"/pdf 模型调用失败，尝试下一个: purpose={purpose or '-'} provider_id={pid}: {err_str}")

        if image_error_count and image_error_count == generation_failure_count:
            raise PdfImageInputError("图片读取失败：可用模型均无法处理该图片，请重新上传原图。候补模型配置错误不会改变这个图片错误。")

        summary = "; ".join(f"{pid}: {err}" for pid, err in failures[-3:])
        if last_exc:
            raise RuntimeError(f"PDF 模型调用全部失败({purpose or '-'}): {summary or last_exc}") from last_exc
        raise RuntimeError("PDF 模型调用全部失败")

    @tracked_generation('pdf')
    @randomized_pdf_style
    async def _solve_math_to_pdf(
            self,
            problem_text: str,
            event: AstrMessageEvent,
            image_urls: Optional[List[str]] = None,
            ref_pdf_latex: str = "",
    ):
        """LLM -> LaTeX -> PDF（仅本地 xelatex 编译）"""
        # One immutable snapshot per PDF job; never affect shared builders used by /spdf.
        personal_prompt = await asyncio.to_thread(PdfPromptPreferences(self.DATA_DIR).get, user_key(event))
        personal_context = preference_context(
            personal_prompt, str(getattr(event, "message_str", "") or ""), problem_text,
        )
        def with_preferences(builder):
            return lambda spec: compose_pdf_prompt(builder(spec), personal_context)

        primary_provider_id = str(self._cfg("pdf_provider_id", "") or "").strip()
        gpt_requested = self._pdf_requests_gpt_model(problem_text)
        gpt_provider_id = str(self._cfg("pdf_gpt_provider_id", "") or "").strip() if gpt_requested else ""
        if gpt_requested and not gpt_provider_id:
            logger.warning("/pdf 检测到 GPT 关键词，但未配置 pdf_gpt_provider_id；继续使用常规模型链")

        # 当前会话使用的 provider
        current_provider_id = ""
        try:
            current_provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        except Exception:
            current_provider_id = ""

        original_image_count = len(image_urls or [])
        if image_urls:
            image_urls = await self._pdf_snapshot_image_urls(image_urls, event=event)
        has_image = bool(image_urls)
        if original_image_count and (not has_image) and (not str(problem_text or "").strip()):
            raise RuntimeError("Upstream Error: Image invalid")
        ref_pdf_latex = (ref_pdf_latex or "").strip()

        gpt_timeout_sec = self._pdf_parse_timeout_value(
            self._cfg("pdf_gpt_provider_timeout_sec", None),
            self._pdf_default_model_timeout_sec(),
        )
        provider_chain = self._pdf_build_provider_chain(
            primary_provider_id,
            current_provider_id,
            routed_provider_id=gpt_provider_id,
            routed_timeout_sec=gpt_timeout_sec,
            routed_compact_prompt_rules=(
                self._pdf_compact_enabled_for_slot("pdf_gpt_provider_enable_compact_prompt_rules")
                if gpt_provider_id else None
            ),
        )

        if not provider_chain:
            raise RuntimeError("未能获取 chat_provider_id")
        initial_provider_chain = list(provider_chain)
        provider_id = str(provider_chain[0].get("provider_id", "") or "").strip()

        # 生成提示词：普通新题 / 追问（携带线程上下文；追问也可能带截图等新图）
        def prompt(spec: Dict[str, Any]) -> str:
            compact_prompt_rules = bool(spec.get("compact_prompt_rules", True))
            if ref_pdf_latex:
                return self._build_pdf_followup_prompt(
                    question_text=problem_text,
                    ref_pdf_latex=ref_pdf_latex,
                    has_image=has_image,
                    compact_prompt_rules=compact_prompt_rules,
                )
            return self._build_pdf_latex_prompt(
                problem_text,
                has_image=has_image,
                compact_prompt_rules=compact_prompt_rules,
            )

        raw, provider_id, _initial_attempted_provider_ids = await self._pdf_llm_generate_with_fallback(
                provider_chain=provider_chain,
                prompt=with_preferences(prompt),
                image_urls=image_urls,
                purpose="initial",
        )
        provider_chain = self._pdf_prioritize_provider_chain(provider_chain, provider_id)
        parsed = _pdf_raw_to_latex_parts(raw, problem_text=problem_text, has_image=has_image, preserve_empty_theorems=bool(personal_prompt))
        raw = str(parsed.get("raw", "") or "")
        problem_tex = str(parsed.get("problem_tex", "") or "")
        theorems_tex = str(parsed.get("theorems_tex", "") or "")
        solution_tex = str(parsed.get("solution_tex", "") or "")
        norm_items = parsed.get("norm_items") or []
        is_multi = bool(parsed.get("is_multi", False))
        tex_src = str(parsed.get("tex_src", "") or "")
        # 编译（仅本地）
        # 可选：失败后自动把编译日志回传给“生成 LaTeX 的模型”进行修复，并用“守门模型”判定是否通过
        pdf_bytes: Optional[bytes] = None
        compile_guard_enabled = bool(self._cfg("pdf_enable_compile_guard", False))
        completeness_guard_enabled = bool(self._cfg("pdf_enable_completeness_guard", False))
        if 'is_multi' in locals() and is_multi:
            completeness_guard_enabled = False


        compile_max_rounds = int(self._cfg("pdf_guard_max_rounds", 3) or 3)
        completeness_max_rounds = int(self._cfg("pdf_completeness_guard_max_rounds", 2) or 2)

        overall_max_rounds = max(
            compile_max_rounds if compile_guard_enabled else 0,
            completeness_max_rounds if completeness_guard_enabled else 0,
        )

        if compile_guard_enabled or completeness_guard_enabled:
            for round_idx in range(overall_max_rounds + 1):
                pdf_bytes = await self._compile_tex_to_pdf(tex_src)
                compile_ok = bool(pdf_bytes)

                # 1) 编译错误守门：编译失败时优先修复
                if compile_guard_enabled:
                    judge_pass = await self._pdf_guard_judge(
                        compile_ok=compile_ok,
                        log_text=(self._last_texlive_log or ""),
                        tex_src=tex_src,
                        preferred_provider_id=str(self._cfg("pdf_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if (not compile_ok) or (not judge_pass):
                        if round_idx >= overall_max_rounds:
                            break

                        # 编译失败：把日志发回给“生成 LaTeX 的模型”让其修复重写
                        def repair_prompt(spec: Dict[str, Any]) -> str:
                            return self._build_pdf_latex_repair_prompt(
                                problem_text=problem_text,
                                has_image=has_image,
                                compile_log=(self._last_texlive_log or ""),
                                prev_tex_src=tex_src,
                                prev_raw=raw,
                                compact_prompt_rules=bool(spec.get("compact_prompt_rules", True)),
                            )

                        raw, provider_id, _repair_attempted_provider_ids = await self._pdf_llm_generate_with_fallback(
                                provider_chain=provider_chain,
                                prompt=with_preferences(repair_prompt),
                                image_urls=image_urls,
                                purpose="compile_repair",
                        )
                        provider_chain = self._pdf_prioritize_provider_chain(provider_chain, provider_id)
                        parsed = _pdf_raw_to_latex_parts(raw, problem_text=problem_text, has_image=has_image, preserve_empty_theorems=bool(personal_prompt))
                        raw = str(parsed.get("raw", "") or "")
                        problem_tex = str(parsed.get("problem_tex", "") or "")
                        theorems_tex = str(parsed.get("theorems_tex", "") or "")
                        solution_tex = str(parsed.get("solution_tex", "") or "")
                        norm_items = parsed.get("norm_items") or []
                        is_multi = bool(parsed.get("is_multi", False))
                        tex_src = str(parsed.get("tex_src", "") or "")
                        # 下一轮继续：先编译，再判错/判完整性
                        continue

                # 若编译失败且没有启用编译守门，则跳出交给后续“纯文本兜底”
                if not compile_ok and not compile_guard_enabled:
                    break

                # 2) 完整性守门：编译通过后检查“是否漏答/未写完”
                if compile_ok and completeness_guard_enabled:
                    comp_pass, comp_issues = await self._pdf_completeness_guard_judge(
                        problem_text=problem_text,
                        problem_tex=problem_tex,
                        solution_tex=solution_tex,
                        preference_scope=personal_context,
                        preferred_provider_id=str(self._cfg("pdf_completeness_guard_provider_id", "") or "").strip(),
                        fallback_provider_id=(current_provider_id or provider_id or "").strip(),
                    )

                    if comp_pass:
                        break

                    if round_idx >= overall_max_rounds:
                        break

                    # 不完整：把缺失点回传给“生成 LaTeX 的模型”让其补全重写
                    def repair_prompt(spec: Dict[str, Any]) -> str:
                        return self._build_pdf_incomplete_repair_prompt(
                            problem_text=problem_text,
                            has_image=has_image,
                            issues=comp_issues,
                            prev_tex_src=tex_src,
                            prev_raw=raw,
                            compact_prompt_rules=bool(spec.get("compact_prompt_rules", True)),
                        )

                    raw, provider_id, _repair_attempted_provider_ids = await self._pdf_llm_generate_with_fallback(
                            provider_chain=provider_chain,
                            prompt=with_preferences(repair_prompt),
                            image_urls=image_urls,
                            purpose="completeness_repair",
                    )
                    provider_chain = self._pdf_prioritize_provider_chain(provider_chain, provider_id)
                    parsed = _pdf_raw_to_latex_parts(raw, problem_text=problem_text, has_image=has_image, preserve_empty_theorems=bool(personal_prompt))
                    raw = str(parsed.get("raw", "") or "")
                    problem_tex = str(parsed.get("problem_tex", "") or "")
                    theorems_tex = str(parsed.get("theorems_tex", "") or "")
                    solution_tex = str(parsed.get("solution_tex", "") or "")
                    norm_items = parsed.get("norm_items") or []
                    is_multi = bool(parsed.get("is_multi", False))
                    tex_src = str(parsed.get("tex_src", "") or "")
                    continue

                # 编译通过，且不需要/已通过完整性守门
                if compile_ok:
                    break
        else:
            pdf_bytes = await self._compile_tex_to_pdf(tex_src)

        # 二次兜底：把主要内容按“纯文本”强转义
        if not pdf_bytes:
            compile_log_before_safe_retry = self._last_texlive_log or ""
            safe_problem = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(problem_tex or problem_text))
            safe_solution = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(solution_tex or raw))
            safe_problem = _escape_text_preserve_dollar_math(safe_problem)
            safe_solution = _escape_text_preserve_dollar_math(safe_solution)

            safe_theorems = (
                r"\begin{theoremBox}{说明}" + "\n"
                r"（由于原始 LaTeX 片段含有不兼容字符，已自动转为安全文本显示。数学公式若以 $...$ 包裹则仍保持公式渲染。）" + "\n"
                r"\end{theoremBox}"
            )
            if 'is_multi' in locals() and is_multi and 'norm_items' in locals() and norm_items:
                safe_items = []
                for _p, _t, _s in norm_items:
                    sp = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_p)))
                    ss = _escape_text_preserve_dollar_math(_ensure_balanced_dollar_math(_strip_known_xml_like_tags(_s)))
                    # theorems 统一放在开头：这里用 safe_theorems 作为占位，后续在 multi builder 中会自动去重
                    safe_items.append((sp, safe_theorems, ss))
                tex_src_retry = _build_pdf_latex_document_multi(safe_items)
            else:
                tex_src_retry = _build_pdf_latex_document(safe_problem, safe_theorems, safe_solution)
            pdf_bytes = await self._compile_tex_to_pdf(tex_src_retry)
            if pdf_bytes:
                tex_src = tex_src_retry
            elif (not (self._last_texlive_log or "").strip()) and compile_log_before_safe_retry.strip():
                self._last_texlive_log = compile_log_before_safe_retry

        # 三次兜底：当前候补模型“能返回文本”但 LaTeX 编译失败时，继续换后续模型重新生成。
        if not pdf_bytes and bool(self._cfg("pdf_fallback_on_compile_error", True)):
            failed_compile_logs: List[Tuple[str, str]] = []
            first_compile_log = (self._last_texlive_log or "").strip()
            if first_compile_log:
                failed_compile_logs.append((provider_id, first_compile_log))

            used_pid = str(provider_id or "").strip()
            tried_compile_pids = {used_pid} if used_pid else set()
            remaining_specs: List[Dict[str, Any]] = []
            seen_used = False
            for spec in initial_provider_chain:
                pid = str(spec.get("provider_id", "") or "").strip()
                if not pid or pid in tried_compile_pids:
                    if pid == used_pid:
                        seen_used = True
                    continue
                if used_pid and not seen_used:
                    continue
                remaining_specs.append(spec)

            for spec in remaining_specs:
                next_pid = str(spec.get("provider_id", "") or "").strip()
                if not next_pid:
                    continue
                tried_compile_pids.add(next_pid)
                logger.warning(
                    f"/pdf 当前模型输出编译失败，尝试下一个模型重新生成: "
                    f"failed_provider_id={provider_id} next_provider_id={next_pid}"
                )
                try:
                    raw_next, provider_next, _compile_fallback_attempted_provider_ids = await self._pdf_llm_generate_with_fallback(
                        provider_chain=[spec],
                        prompt=with_preferences(prompt),
                        image_urls=image_urls,
                        purpose="compile_fallback",
                    )
                    parsed_next = _pdf_raw_to_latex_parts(raw_next, problem_text=problem_text, has_image=has_image, preserve_empty_theorems=bool(personal_prompt))
                    candidate_tex_src = str(parsed_next.get("tex_src", "") or "")
                    candidate_pdf = await self._compile_tex_to_pdf(candidate_tex_src)

                    if not candidate_pdf:
                        compile_log_before_safe_retry = self._last_texlive_log or ""
                        candidate_problem = str(parsed_next.get("problem_tex", "") or "")
                        candidate_solution = str(parsed_next.get("solution_tex", "") or parsed_next.get("raw", "") or "")
                        safe_problem = _ensure_balanced_dollar_math(
                            _strip_known_xml_like_tags(candidate_problem or problem_text)
                        )
                        safe_solution = _ensure_balanced_dollar_math(_strip_known_xml_like_tags(candidate_solution))
                        safe_problem = _escape_text_preserve_dollar_math(safe_problem)
                        safe_solution = _escape_text_preserve_dollar_math(safe_solution)
                        safe_theorems = (
                            r"\begin{theoremBox}{说明}" + "\n"
                            r"（由于原始 LaTeX 片段含有不兼容字符，已自动转为安全文本显示。数学公式若以 $...$ 包裹则仍保持公式渲染。）" + "\n"
                            r"\end{theoremBox}"
                        )
                        candidate_norm_items = parsed_next.get("norm_items") or []
                        if bool(parsed_next.get("is_multi", False)) and candidate_norm_items:
                            safe_items = []
                            for _p, _t, _s in candidate_norm_items:
                                sp = _escape_text_preserve_dollar_math(
                                    _ensure_balanced_dollar_math(_strip_known_xml_like_tags(_p))
                                )
                                ss = _escape_text_preserve_dollar_math(
                                    _ensure_balanced_dollar_math(_strip_known_xml_like_tags(_s))
                                )
                                safe_items.append((sp, safe_theorems, ss))
                            tex_src_retry = _build_pdf_latex_document_multi(safe_items)
                        else:
                            tex_src_retry = _build_pdf_latex_document(safe_problem, safe_theorems, safe_solution)
                        candidate_pdf = await self._compile_tex_to_pdf(tex_src_retry)
                        if candidate_pdf:
                            candidate_tex_src = tex_src_retry
                        elif (not (self._last_texlive_log or "").strip()) and compile_log_before_safe_retry.strip():
                            self._last_texlive_log = compile_log_before_safe_retry

                    if candidate_pdf:
                        pdf_bytes = candidate_pdf
                        raw = str(parsed_next.get("raw", "") or raw_next or "")
                        provider_id = str(provider_next or next_pid)
                        provider_chain = self._pdf_prioritize_provider_chain(provider_chain, provider_id)
                        problem_tex = str(parsed_next.get("problem_tex", "") or "")
                        theorems_tex = str(parsed_next.get("theorems_tex", "") or "")
                        solution_tex = str(parsed_next.get("solution_tex", "") or "")
                        norm_items = parsed_next.get("norm_items") or []
                        is_multi = bool(parsed_next.get("is_multi", False))
                        tex_src = candidate_tex_src
                        logger.info(f"/pdf 编译失败后模型回退成功: provider_id={provider_id}")
                        break

                    failed_log = (self._last_texlive_log or "").strip()
                    if failed_log:
                        failed_compile_logs.append((str(provider_next or next_pid), failed_log))
                except Exception as e:
                    failed_compile_logs.append((next_pid, str(e)))
                    logger.warning(f"/pdf 编译失败后回退模型也未成功: provider_id={next_pid}: {e}")

            if not pdf_bytes and failed_compile_logs and not (self._last_texlive_log or "").strip():
                last_pid, last_log = failed_compile_logs[-1]
                self._last_texlive_log = f"{last_pid}: {last_log}"

        if not pdf_bytes:
            log_snip = (self._last_texlive_log or "").strip()[:1000]
            raise RuntimeError(f"Compile failed. Log: {log_snip}")

        user_name = _get_sender_display_name(event)
        # 文件名必须唯一：原来只精确到分钟，用户在同一分钟内多次 /pdf（尤其并发）会互相覆盖，导致 PDF 被截断/乱码
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:8]
        model_label = await self._pdf_model_label_for_filename(provider_id)
        fname = _sanitize_filename(f"{model_label}-{user_name}-{ts}-{rand}.pdf")

        pdf_path = os.path.join(self.PDF_CACHE_DIR, fname)
        tmp_path = pdf_path + f".tmp-{uuid.uuid4().hex}"
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)
        # 原子替换，避免发送时读到“半截文件”
        os.replace(tmp_path, pdf_path)

        return pdf_path, fname, tex_src

    def _pdf_compact_prompt_rules(
            self,
            first_prefix: str = "   - ",
            next_prefix: Optional[str] = None,
            enabled: Optional[bool] = None,
    ) -> str:
        if enabled is None:
            enabled = bool(self._cfg("pdf_enable_compact_prompt_rules", True))
        if not enabled:
            return ""
        return (
            f"{first_prefix}只有必要的行间公式才使用 $$...$$ 语法，其余公式都使用行内公式 $...$\n"
        )

    def _build_pdf_latex_prompt(
            self,
            problem_text: str,
            has_image: bool = False,
            compact_prompt_rules: Optional[bool] = None,
    ) -> str:
        """/pdf 新题提示词（仅生成可编译的 LaTeX 片段）
        包含：TikZ绘图支持、知识点总结模式、完整性强制约束
        """
        intro_text = ""

        # --- 1. 场景判断逻辑 (保持不变) ---
        if has_image:
            if problem_text:
                # 【情况A：图文共存】强制模型以图片为核心，文字为追问
                intro_text = (
                    "用户上传了图片作为**关键的解题背景或证明过程**，同时提供了具体的**文字追问**。\n"
                    "**最高优先级指令**：\n"
                    "1. 你的回答必须**深度结合图片内容**。用户提到的关键点（如系数来源、特定步骤、为何是1/4等）都在图片中，脱离图片回答是无效的。\n"
                    "2. 请先识别图片中的数学推导、定理引用或手写批注，定位到用户文字提问所指的具体步骤。\n"
                    "3. 在 <problem> 标签中，请综合图片里的题面/证明目标以及用户的文字追问来描述。\n"
                    f"【用户文字追问】：{problem_text}\n"
                )
            else:
                # 【情况B：仅有图片】常规识别
                intro_text = (
                    "用户上传了一张包含数学题目的图片。\n"
                    "你的首要任务是识别图片中的题目内容，并将其转化为 LaTeX 格式放在 <problem> 标签中。\n"
                )
        else:
            # 【情况C：仅有文字】
            intro_text = f"题目/需求如下：\n{problem_text}\n"

        # --- 2. 构建核心提示词 ---
        return (
            "你是数学题标准解答的排版与写作助手。\n"
            "你的任务：把题目与标准解答写成可直接编译的 LaTeX 正文片段（中文可用）。\n"
            f"{intro_text}"
            "要求：\n"
            "1) **纯净输出**：不要输出 \\\\documentclass、\\\\usepackage 等导言区内容；不要使用 Markdown 代码块（```）；只输出纯 LaTeX 正文。\n"

            "2) **完整性强制**：\n"
            "   - 解答必须步骤完整、推导严谨。\n"
            "   - **禁止省略**：绝对不要使用“同理可得”、“略”、“步骤省略”等字眼。\n"
            "   - **禁止截断**：必须把每一个运算步骤都写出来，直到得出最终答案。\n"
            "   - **解题完整**：遇到一个题目有多个小问时，你必须将每个小问的解答都完整的写出\n"
            "   - **结尾标记强制**：在 <solution> 解答内容的最后，请在最后一句公式或文字后加上：\\hfill$\\blacksquare$。\n"
            "     并确保这是解答的最终结尾（后面不要再有其他文字）。\n"

            "3) **数学环境**：\n"
            "   - 需要用到的定理/引理/公式请放在 <theorems> 段中。\n"
            "   - 使用对应环境：\\\\begin{theoremBox}{名称}...\\\\end{theoremBox}、\\\\begin{lemmaBox}{名称}...\\\\end{lemmaBox}。\n"
            f"{self._pdf_compact_prompt_rules(enabled=compact_prompt_rules)}"

            "4) **【绘图支持】(重要)**：\n"
            "   - 如果题目涉及几何图形、函数图像或证明示意图（如三角形剖分、积分路径、区域 $D$），**请务必绘制示意图，并且尽可能的美观**。\n"
            "   - 使用 **TikZ** 环境：\\\\begin{tikzpicture} ... \\\\end{tikzpicture}。\n"
            "   - **防重叠强制规则**：\n"
            "     a) 全局缩放：请默认使用 `[scale=1.5]` 或更大，给文字留出空间。\n"
            "     b) 节点间距：设置 `node distance=2.5cm` 或更宽，不要挤在一起。\n"
            "     c) 标签避让：不要只用 `above/below`，请使用偏移量，例如 `label={[yshift=0.2cm]above:文字}` 或 `node[above=0.3cm]{文字}`。\n"
            "     d) 长文本处理：对于较长的中文说明，请使用 `align=center` 并强制换行（\\\\），例如 `node[align=center]{第一行\\\\第二行}`。\n"
            "     e) 防遮挡强制：若需填充颜色（如积分区域 D），请务必先写填充代码(\\fill)，再写坐标轴和轮廓线(\\draw)；或者将填充代码包裹在 `\\begin{pgfonlayer}{bg} ... \\end{pgfonlayer}` 环境中，确保坐标轴永远浮在颜色上方。\n"       
            "   - 保持代码简洁兼容，例如使用 \\\\draw, \\\\node, \\\\coordinate 等基础命令。\n"
            "   - 将绘图代码放在 <solution> 的合适位置（通常在文字解析之前或之中），可以在解答的不同位置放置多个绘图代码来方便说明。\n"

            "5) **【特殊模式：知识总结与详解】**：\n"
            "   若用户请求的是“知识点总结”、“概念复习”、“详细讲解”或“证明定理”，而非单纯解题，请按以下标准执行：\n"
            "   - **全面性**：必须覆盖定义、核心性质、几何意义（如有）、易错点。\n"
            "   - **证明**：对于核心定理或公式，**必须给出严谨的数学证明（Proof）**，不要只列结论。\n"
            "   - **示例**：在讲解完概念后，**必须编写 1-2 个具体的典型例题（Example）**并给出完整解答，以辅助用户理解。\n"
            "   - **排版映射**：请将“知识点标题/核心摘要”放入 <problem> 标签；将“详细讲解、证明过程、例题分析、TikZ插图”放入 <solution> 标签。\n"

            "6) **编译安全**：\n"
            "   - 文本模式下不要出现裸的特殊符号：_, ^, \\\\, %, &, #, <, > 等，必须转义（如 \\\\_）。\n"
            "   - 下标/上标请严格放在数学环境 $...$ 中。\n"
            "   - 【重要】若需加粗，请直接使用 \\\\textbf{...}，禁止使用 Markdown 的 **...** 语法。\n"  

            "7) **输出结构**（标签必须保留）：\n"
            "   - 如果用户一次给了多题（如“题1/题2/…”或编号列表），请按题号把每一题拆开，\n"
            "     并按如下 3 个标签为一组重复输出多次（每题一组、顺序固定、不要输出任何其它标签）。\n"
            "   - 单题则只输出一组。\n"
            "   - 注意：<solution> 中不要复述题面，题面只放在 <problem>。\n"
            "   - 多题时，各题的 <theorems> 可为空；系统会把所有题出现过的结论统一汇总到 PDF 开头。\n"
            "   - 相关的数学公式应当尽量排在一块，不要无意义地换行，符合标准习题解答规范。\n"
            "<problem>（第1题题面 LaTeX）...</problem>\n"
            "<theorems>（第1题用到的定理/公式）...</theorems>\n"
            "<solution>（第1题完整解答）...\\hfill$\\blacksquare$</solution>\n"
            "\n"
            "<problem>（第2题题面 LaTeX）...</problem>\n"
            "<theorems>（第2题用到的定理/公式）...</theorems>\n"
            "<solution>（第2题完整解答）...\\hfill$\\blacksquare$</solution>\n"
        )

    def _build_pdf_latex_repair_prompt(
            self,
            problem_text: str,
            has_image: bool,
            compile_log: str,
            prev_tex_src: str = "",
            prev_raw: str = "",
            compact_prompt_rules: Optional[bool] = None,
    ) -> str:
        """当 /pdf 的 LaTeX 编译失败时，构造“修复重写”提示词。
        目标：让同一个“生成 LaTeX 的模型”根据编译日志重写一个可编译版本（仍输出 <problem>/<theorems>/<solution>）。"""
        log_tail_chars = int(self._cfg("pdf_guard_log_tail_chars", 4000) or 4000)
        include_tex = bool(self._cfg("pdf_guard_include_tex_in_feedback", True))

        log_txt = (compile_log or "").strip()
        if log_tail_chars > 0 and len(log_txt) > log_tail_chars:
            log_txt = log_txt[-log_tail_chars:]

        tex_txt = (prev_tex_src or "").strip() if include_tex else ""
        # 避免把超长 TeX 源码塞爆上下文：仅取尾部（通常错误就在后面）
        if tex_txt and len(tex_txt) > 12000:
            tex_txt = tex_txt[-12000:]

        raw_txt = (prev_raw or "").strip()
        if raw_txt and len(raw_txt) > 6000:
            raw_txt = raw_txt[-6000:]

        # 修复提示词尽量短、指令强，避免模型再输出解释文字
        image_hint = (
            "你还会收到一张/多张图片，请先识别图片中的题目内容，再给出可编译的 LaTeX。\n"
            if has_image else ""
        )

        parts = [
            "你刚才生成的 LaTeX 在 XeLaTeX 编译时失败了。\n",
            "请根据编译日志修复并重新生成一份**可直接编译**的 LaTeX 输出。\n",
            "硬性要求：\n",
            "1) 必须输出并且只输出三个标签块：<problem>...</problem>、<theorems>...</theorems>、<solution>...</solution>。\n",
            "2) 标签必须完整闭合；不要输出 Markdown 代码块/反引号；不要输出任何解释文字。\n",
            "3) 禁止使用需要 shell-escape 的写法；避免不常见/不稳定宏包；保持语法简单稳定。\n",
            "4) 对于文本中的特殊字符（如 %, #, &, _, {, }）需正确转义。\n",
            self._pdf_compact_prompt_rules("- ", enabled=compact_prompt_rules),
            "5) 在 <solution> 最后必须加入：\\hfill$\\blacksquare$，确保这是最终结尾。\n",
            image_hint,
            "\n【题目/追问】\n" + (problem_text or "(无文字，仅图片)") + "\n",
            "\n【编译错误日志（截断）】\n```text\n" + (log_txt or "(无日志)") + "\n```\n",
        ]

        if tex_txt:
            parts.append("\n【上一版完整 TeX 源码（供你参考并修复）】\n```latex\n" + tex_txt + "\n```\n")
        elif raw_txt:
            parts.append("\n【上一版模型原始输出（供你参考并修复）】\n" + raw_txt + "\n")

        parts.append("\n现在请输出修复后的 <problem>/<theorems>/<solution>：\n")
        return "".join(parts)

    def _build_pdf_incomplete_repair_prompt(
            self,
            problem_text: str,
            has_image: bool,
            issues: List[str],
            prev_tex_src: str = "",
            prev_raw: str = "",
            compact_prompt_rules: Optional[bool] = None,
    ) -> str:
        """当 /pdf 的解答“未写完/漏答”时，构造“补全重写”提示词。"""
        issue_lines: List[str] = []
        for it in (issues or []):
            s = str(it).strip()
            if s:
                issue_lines.append(f"- {s}")
        issue_block = "\n".join(issue_lines) if issue_lines else "- （未能解析出具体缺失点，请你自查是否漏答/未写完）"

        include_tex = bool(self._cfg("pdf_completeness_guard_include_tex_in_feedback", False))
        tex_txt = (prev_tex_src or "").strip() if include_tex else ""
        raw_txt = (prev_raw or "").strip() if (not include_tex) else ""

        base = r"""你是一位严谨的数学解题助手。
你上一版给出的 PDF LaTeX 解答被检测为“内容不完整/有小问漏答/某一问未写完”。
请根据下方【缺失点】补全并重写整份解答。

硬性要求：
1) 必须输出完整的三段标签：<problem>...</problem>、<theorems>...</theorems>、<solution>...</solution>。
2) 如果题目包含多问（如 (1)(2)(3)、①②③、第一问/第二问/第三问、a)b)c) 等），你必须逐问作答，并在解答中显式标注每一问。
3) 不要只追加“缺失部分”，而是输出一份可直接编译的完整版本（包含所有小问的完整解答）。
4) 避免半句/半段就结束；最后要有完整的收尾。
5) 继续保持 XeLaTeX 兼容：不要使用不受支持的宏包/命令，不要留下未闭合的环境或括号。

8) 在 <solution> 解答内容的最后必须加入：\hfill$\blacksquare$，并确保这是最终结尾。

"""

        img_line = ""
        if has_image:
            img_line = "你还会收到图片（题目截图）。若题目文字不全，请结合图片补全题意并完整解答。\n\n"

        parts = [
            base,
            self._pdf_compact_prompt_rules("6) ", "7) ", enabled=compact_prompt_rules),
            img_line,
            "【题目（用户输入/意图）】\n" + (problem_text or "(空)") + "\n\n",
            "【缺失点（请逐条修复）】\n" + issue_block + "\n\n",
        ]

        if tex_txt:
            parts.append("【上一版完整 TeX 源码（供参考）】\n```latex\n" + tex_txt + "\n```\n\n")
        elif raw_txt:
            parts.append("【上一版模型原始输出（供参考）】\n" + raw_txt + "\n\n")

        parts.append("现在请输出修复后的 <problem>/<theorems>/<solution>：\n")
        return "".join(parts)

    async def _pdf_completeness_guard_judge(
            self,
            problem_text: str,
            problem_tex: str,
            solution_tex: str,
            preferred_provider_id: str = "",
            fallback_provider_id: str = "",
            preference_scope: str = "",
    ) -> (bool, List[str]):
        """可选：用一个“完整性守门模型”判断解答是否写完/是否漏答多问。
        返回：(是否通过, 缺失点列表)。"""
        if not bool(self._cfg("pdf_enable_completeness_guard", False)):
            return True, []

        provider = (preferred_provider_id or fallback_provider_id or "").strip()
        if not provider:
            return True, []

                # 强机制：结尾标记（用于可靠判断“是否写完”）
        end_marker_required = bool(self._cfg("pdf_completeness_require_end_marker", True))
        marker_only = bool(self._cfg("pdf_completeness_marker_only", True))
        # 结尾标记默认使用 \blacksquare（黑色实心方块）
        end_marker_text = str(self._cfg("pdf_completeness_end_marker_text", r"\blacksquare") or r"\blacksquare").strip()

        def _marker_line(marker: str) -> str:
            marker = (marker or "").strip()
            # 若是 LaTeX 命令（以 \ 开头），默认放在数学模式中
            if marker.startswith("\\"):
                # --- 修改开始 ---
                # 原代码：return f"\\begin{{center}}${marker}$\\end{{center}}"
                # 新代码：使用 hfill 居右
                return f"\\hfill${marker}$"
                # --- 修改结束 ---
            # 否则按“文字标记”处理（兼容旧配置）
            return f"\\begin{{center}}\\heiti\\bfseries {marker}\\end{{center}}"

        def _marker_at_end(solution: str, marker: str) -> bool:
            st = (solution or "").strip()
            if (not st) or (not marker):
                return False
            idx = st.rfind(marker)
            if idx < 0:
                return False

            # 标记必须出现在解答尾部（避免“中间出现但后面还有内容”）
            if idx < max(0, len(st) - 1600):
                return False

            after = st[idx + len(marker):]
            after_s = (after or "").strip()
            if not after_s:
                return True

            # 允许的尾部：若标记在 $...$ 或 center/proof 环境中，可能还有 $ 或 \end{...}
            norm = re.sub(r"\s+", "", after_s)
            if re.fullmatch(r"\$*\}*(\\end\{(?:center|proof)\})*\$*", norm or ""):
                return True

            return False

        if end_marker_required and end_marker_text:
            st = (solution_tex or "").strip()
            marker_line = _marker_line(end_marker_text)

            if end_marker_text not in st:
                fix = f"缺少结尾标记：{end_marker_text}（请在 <solution> 最后一行加入 {marker_line}）"
                return False, [fix]

            if not _marker_at_end(st, end_marker_text):
                fix = (
                    "结尾处未按要求放置证明完毕标记：请把 "
                    f"{marker_line} 放在 <solution> 的最后一行，并确保它是解答最终结尾"
                )
                return False, [fix]

            # 若配置为“只依赖结尾标记”，则直接通过（更强更稳）
            if marker_only:
                return True, []

        # 控制输入长度，避免 token 爆炸
        pt = (problem_text or "").strip()
        if pt and len(pt) > 8000:
            pt = pt[:8000] + "\n...(truncated)..."

        ptex = (problem_tex or "").strip()
        if ptex and len(ptex) > 8000:
            ptex = ptex[:8000] + "\n...(truncated)..."

        stex = (solution_tex or "").strip()
        if stex and len(stex) > 12000:
            stex = stex[:12000] + "\n...(truncated)..."

        prompt = (
            """你是一个严格的数学解答“完整性审查”模型。
你的任务：判断解答是否回答了题目中的所有小问，并且每一问是否写完（不存在明显截断/半段结束）。

注意：
特别规则：解答最后必须包含证明完毕标记 `\\blacksquare`（通常位于段落末尾，写作 `\\hfill$\\blacksquare$`）。如果没有这一标记，一律判定 complete=false，并在 issues 中说明缺少结尾标记。

- 题目可能包含 (1)(2)(3)、①②③、第一问/第二问/第三问、a)b)c) 等多问结构。
- 如果发现漏答或某一问明显没写完，请在 issues 里指出“缺哪一问/哪一问未写完/缺少关键步骤或结论”。
- 如果无法确定是否多问，请更保守：只有在非常明显的情况下才判定不通过。

只输出 JSON，不要输出任何额外文字：
{"complete": true/false, "issues": ["...","..."], "note": "一行总结"}

"""
            + "[problem_text]\n" + (pt or "(empty)") + "\n\n"
            + "[problem_tex]\n" + (ptex or "(empty)") + "\n\n"
            + "[solution_tex]\n" + (stex or "(empty)") + "\n"
        )

        if preference_scope:
            prompt += (preference_scope + "\n完整性仅按上述有效要求审查。用户要求的省略、提示式解答、"
                       "限定小问、简短回答或不附例题，不属于漏答。仍只输出上述 JSON。\n")

        try:
            timeout_sec = self._pdf_timeout_for_provider(provider)
            record_model_call(provider)
            call = generate_with_token_usage(self,
                chat_provider_id=provider,
                prompt=prompt,
                image_urls=[],
            )
            llm_resp = await asyncio.wait_for(call, timeout=timeout_sec) if timeout_sec > 0 else await call
            out = (
                getattr(llm_resp, "completion_text", None)
                or getattr(llm_resp, "completion", None)
                or getattr(llm_resp, "text", None)
                or ""
            )
            out = (out or "").strip()
        except Exception as e:
            logger.warning(f"pdf_completeness_guard_judge failed: {e}")
            return True, []

        issues: List[str] = []
        try:
            mm = re.search(r"\{[\s\S]*\}", out)
            if mm:
                obj = json.loads(mm.group(0))
                if isinstance(obj, dict):
                    complete = bool(obj.get("complete", True))
                    raw_issues = obj.get("issues", [])
                    if isinstance(raw_issues, list):
                        for it in raw_issues[:10]:
                            s = str(it).strip()
                            if s:
                                issues.append(s)
                    elif isinstance(raw_issues, str) and raw_issues.strip():
                        issues.append(raw_issues.strip())
                    return complete, issues
        except Exception:
            pass

        low = (out or "").lower()
        if "complete" in low and "false" in low:
            for line in out.splitlines():
                if any(k in line for k in ["漏", "缺", "未写完", "没写完", "missing", "incomplete"]):
                    s = line.strip("-• \t")
                    if s and len(s) < 200:
                        issues.append(s)
            return False, issues[:10]

        return True, []

    async def _pdf_guard_judge(
            self,
            compile_ok: bool,
            log_text: str,
            tex_src: str,
            preferred_provider_id: str = "",
            fallback_provider_id: str = "",
    ) -> bool:
        """可选：用一个“守门模型”判定是否存在编译错误。
        - compile_ok=True 通常直接 PASS
        - compile_ok=False 则根据日志判定 FAIL/是否需要继续重试
        返回：是否通过（True=通过，False=不通过/需要重试）"""
        if not bool(self._cfg("pdf_enable_compile_guard", False)):
            return compile_ok

        # 若未配置守门模型，退化为“是否编译成功”
        provider = (preferred_provider_id or fallback_provider_id or "").strip()
        if not provider:
            return compile_ok

        log_tail_chars = int(self._cfg("pdf_guard_log_tail_chars", 4000) or 4000)
        log_txt = (log_text or "").strip()
        if log_tail_chars > 0 and len(log_txt) > log_tail_chars:
            log_txt = log_txt[-log_tail_chars:]

        # 不把完整 TeX 都塞给判错模型（token 成本高）；截断一下即可
        tex_snip = (tex_src or "").strip()
        if tex_snip and len(tex_snip) > 6000:
            tex_snip = tex_snip[-6000:]

        prompt = (
            "你是一个严格的 XeLaTeX 编译判错模型。\n"
            "你的任务：判断是否存在“会导致编译失败的错误”。\n"
            "已知：compile_ok 表示是否已经成功生成 PDF。\n"
            "- 如果 compile_ok=true：你必须返回 pass=true（除非你能确定这是伪成功，但通常不应这样判断）。\n"
            "- 如果 compile_ok=false：请根据日志判断 pass=false，并给出一句最关键的失败原因。\n\n"
            "只输出 JSON，不要输出任何额外文字：\n"
            "{\"pass\": true/false, \"reason\": \"...\"}\n\n"
            f"compile_ok={str(bool(compile_ok)).lower()}\n"
            "\n[log]\n" + (log_txt or "(empty)") + "\n"
            "\n[tex_tail]\n" + (tex_snip or "(empty)") + "\n"
        )

        try:
            timeout_sec = self._pdf_timeout_for_provider(provider)
            record_model_call(provider)
            call = generate_with_token_usage(self,
                chat_provider_id=provider,
                prompt=prompt,
                image_urls=[],
            )
            llm_resp = await asyncio.wait_for(call, timeout=timeout_sec) if timeout_sec > 0 else await call
            out = (
                getattr(llm_resp, "completion_text", None)
                or getattr(llm_resp, "completion", None)
                or getattr(llm_resp, "text", None)
                or ""
            )
            out = (out or "").strip()
        except Exception as e:
            # 判错模型失败时，不阻断主流程：退化为 compile_ok
            logger.warning(f"pdf_guard_judge failed: {e}")
            return compile_ok

        # 解析 JSON
        try:
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "pass" in obj:
                    return bool(obj.get("pass"))
        except Exception:
            pass

        low = out.lower()
        if "pass" in low and "true" in low:
            return True
        if "pass" in low and "false" in low:
            return False
        # 最后的兜底：编译是否成功
        return compile_ok

    def _build_pdf_followup_prompt(
            self,
            question_text: str,
            ref_pdf_latex: str,
            has_image: bool = False,
            compact_prompt_rules: Optional[bool] = None,
    ) -> str:
        """/pdf 追问提示词（携带整个追问线程的上下文）"""
        q = (question_text or "").strip()
        ref = (ref_pdf_latex or "").strip()

        # 默认不截断（0 = 不限制）；如需限制可配置 pdf_followup_ref_max_chars
        max_ref_len = int(self._cfg("pdf_followup_ref_max_chars", 0) or 0)
        if max_ref_len > 0 and len(ref) > max_ref_len:
            ref = ref[:max_ref_len] + "\n...(已截断)"

        img_note = (
            "用户本次还附带了图片（可能是题目截图或对上一份 PDF 某处的截图），请结合图片内容回答。\n"
            if has_image else ""
        )
        q_block = q if q else "（用户未写文字，请针对该题继续深入讲解或补全解答细节）"

        intro_text = (
            "用户正在追问之前 PDF 解答的相关问题。\n"
            "下面提供本题的完整对话线程（原始题目 + 历次解答的 LaTeX 源码 + 历次追问），请据此回答用户的最新追问，"
            "并仍然输出一份可编译的 LaTeX 正文片段。\n"
            f"{img_note}"
            "\n"
            "【本题对话线程】\n"
            f"{ref}\n"
            "\n"
            "【用户最新追问】\n"
            f"{q_block}\n"
        )

        return (
            "你是数学题标准解答的排版与写作助手。\n"
            "你的任务：基于用户最新追问 + 本题对话线程上下文，生成一份新的可编译 LaTeX 正文片段（中文可用）。\n"
            f"{intro_text}"
            "要求：\n"
            "1) 不要输出 \\\\documentclass、\\\\usepackage 等导言区内容，只输出正文片段。\n"
            "2) 只允许使用 LaTeX（不得使用 Markdown，不得输出代码块，也不要输出```）。\n"
            "3) 解答必须严格规范：步骤完整、推导严谨、符号定义清晰、结论明确。\n"
            "4) 若需要引用上一份解答中的结论，请在 <theorems> 段中补全或重述必要的定理/引理/公式，并使用对应环境（参数必填）。\n"
            "5) 【特殊模式：延伸讲解与证明】\n"
            "   若用户追问的是“为什么要这样”、“这个公式怎么来的”或“总结一下相关知识点”，请务必：\n"
            "   - **详细展开**：不要只回答是或否，要给出**定义背景**和**推导证明**。\n"
            "   - **补充例题**：针对用户询问的疑难点，额外编造一个简单的应用例题来辅助说明。\n"
            "   - **结构映射**：将“追问的主题”放入 <problem>，将“详细解释与证明”放入 <solution>。\n"
            "6) 关键：必须保证 LaTeX 可编译。\n"
            "   - 文本模式下不要出现裸的特殊符号：_, ^, 反斜杠(\\\\), %, &, #, <, > 等。\n"
            "   - 若要显示这些符号，请使用转义：\\\\_, \\\\textasciicircum{}, \\\\textbackslash{}, \\\\%, \\\\&, \\\\#, \\\\textless{}, \\\\textgreater{}。\n"
            "   - 下标/上标请放在数学环境中，例如 $x_1$, $a^2$。\n"
            f"{self._pdf_compact_prompt_rules('7) 排版紧凑：', '   - ', enabled=compact_prompt_rules)}"
            "8) 输出格式必须严格如下（标签要保留；不要输出其它标签）：\n"
            "<problem>这里写追问对应的题意/子问题/知识点标题...</problem>\n"
            "<theorems>这里写 theoremBox/lemmaBox/formulaBox...</theorems>\n"
            "<solution>这里写完整解答 或 详细知识总结...</solution>\n"
        )

    def _tex_hash(self, tex_src: str) -> str:
        return hashlib.md5(tex_src.encode("utf-8", errors="ignore")).hexdigest()

    def _prune_texlive_cache(self):
        max_files = int(self._cfg("texlive_cache_max_files", 500) or 500)
        try:
            files = []
            for fn in os.listdir(self.TEXLIVE_CACHE_DIR):
                if fn.endswith(".pdf"):
                    fp = os.path.join(self.TEXLIVE_CACHE_DIR, fn)
                    try:
                        st = os.stat(fp)
                        files.append((st.st_mtime, fp))
                    except Exception:
                        pass
            if len(files) <= max_files:
                return
            files.sort(key=lambda x: x[0])  # oldest first
            for _, fp in files[: max(0, len(files) - max_files)]:
                try:
                    os.remove(fp)
                except Exception:
                    pass
        except Exception:
            pass

    async def _compile_tex_to_pdf(self, tex_src: str) -> Optional[bytes]:
        self._last_texlive_log = ""
        h = self._tex_hash(tex_src)

        # 缓存
        if bool(self._cfg("texlive_cache_enabled", True)):
            cache_pdf = os.path.join(self.TEXLIVE_CACHE_DIR, f"{h}.pdf")
            if os.path.exists(cache_pdf):
                try:
                    with open(cache_pdf, "rb") as f:
                        return f.read()
                except Exception:
                    pass

        # 直接本地编译（MikTex 环境 xelatex）
        pdf = await self._compile_local_xelatex(tex_src)
        if pdf:
            await self._save_tex_cache(h, pdf)
            return pdf

        return None

    async def _save_tex_cache(self, h: str, pdf_bytes: bytes):
        if not bool(self._cfg("texlive_cache_enabled", True)):
            return
        try:
            cache_pdf = os.path.join(self.TEXLIVE_CACHE_DIR, f"{h}.pdf")
            tmp_pdf = cache_pdf + f".tmp-{uuid.uuid4().hex}"
            with open(tmp_pdf, "wb") as f:
                f.write(pdf_bytes)
            os.replace(tmp_pdf, cache_pdf)
            self._prune_texlive_cache()
        except Exception:
            pass

    async def _compile_local_xelatex(self, tex_src: str) -> Optional[bytes]:
        timeout_sec = int(self._cfg("local_xelatex_timeout_sec", 60) or 60)

        # 重要：xelatex 编译是阻塞操作。若直接 subprocess.run 会卡住整个事件循环，
        # 并发 /pdf 时会影响其它 LLM 请求，表现为输出被截断/乱码。
        async with self._tex_compile_sema:
            try:
                with tempfile.TemporaryDirectory(prefix="astrbot_xelatex_") as td:
                    tex_path = os.path.join(td, "document.tex")
                    with open(tex_path, "w", encoding="utf-8") as f:
                        f.write(tex_src)

                    # 为 XeLaTeX/Fontconfig 提供可写缓存目录（避免 “No writable cache directories”）
                    env = os.environ.copy()
                    env["HOME"] = td
                    env["XDG_CACHE_HOME"] = os.path.join(td, ".cache")
                    try:
                        os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)
                    except Exception:
                        pass

                    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "document.tex"]

                    # 默认只跑 1 次（更快）；若日志提示需要 rerun（交叉引用/目录等），再补跑第 2 次。
                    rerun_needed = False
                    last_returncode = 0
                    last_stdout = ""
                    for run_idx in range(2):
                        # Cancellation must stop/reap XeLaTeX before deleting its
                        # working directory and releasing the concurrency slot.
                        p = await run_process(cmd, cwd=td, env=env, timeout=timeout_sec)
                        last_returncode = int(getattr(p, "returncode", 0) or 0)
                        try:
                            last_stdout = (getattr(p, "stdout", b"") or b"").decode("utf-8", errors="ignore")
                        except Exception:
                            last_stdout = str(getattr(p, "stdout", "") or "")

                        if last_returncode != 0:
                            break

                        if run_idx == 0:
                            log_path = os.path.join(td, "document.log")
                            if os.path.exists(log_path):
                                try:
                                    with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                                        log_txt = lf.read()
                                    rerun_needed = (
                                        ("Rerun to get cross-references right." in log_txt)
                                        or ("LaTeX Warning: Label(s) may have changed" in log_txt)
                                        or ("There were undefined references" in log_txt)
                                    )
                                except Exception:
                                    rerun_needed = False
                            if not rerun_needed:
                                break

                    pdf_path = os.path.join(td, "document.pdf")
                    if last_returncode != 0 or (not os.path.exists(pdf_path)):
                        # 尝试读取日志；若 log 尚未生成，则使用 xelatex stdout，避免错误信息为空。
                        log_path = os.path.join(td, "document.log")
                        log_txt = ""
                        if os.path.exists(log_path):
                            try:
                                with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                                    log_txt = lf.read()
                            except Exception:
                                log_txt = ""
                        if not log_txt.strip():
                            log_txt = last_stdout
                        if not log_txt.strip():
                            log_txt = f"Local xelatex failed with returncode={last_returncode}"
                        self._last_texlive_log = log_txt
                        return None
                    with open(pdf_path, "rb") as fpdf:
                        return fpdf.read()
            except (subprocess.TimeoutExpired, asyncio.TimeoutError):
                self._last_texlive_log = "Local xelatex timeout"
                return None
            except Exception as e:
                self._last_texlive_log = f"Local xelatex error: {e}"
                return None
