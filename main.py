# -*- coding: utf-8 -*-
import os as _os
import sys as _sys

_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PLUGIN_DIR not in _sys.path:
    _sys.path.insert(0, _PLUGIN_DIR)

try:
    from .shared import *
    from .memory import MemoryMixin
    from .spdf import SpdfMixin
    from .pdf import PdfMixin
    from .render import RenderMixin
    from .daily_report import DailyReportMixin
except ImportError:
    from shared import *
    from memory import MemoryMixin
    from spdf import SpdfMixin
    from pdf import PdfMixin
    from render import RenderMixin
    from daily_report import DailyReportMixin


# =============================================================================
# LLM / Agent Tools（将本插件能力封装为可被 Agent Function Calling 调用的 Tool）
# =============================================================================

# 插件实例（单例）。Tool 运行时通过它调用插件内部方法。
_MD2IMG_PLUGIN_INSTANCE = None  # type: Optional["MarkdownConverterPlugin"]

def _md2img_get_plugin():
    global _MD2IMG_PLUGIN_INSTANCE
    if _MD2IMG_PLUGIN_INSTANCE is None:
        raise RuntimeError("md2img 插件实例尚未初始化，Tool 暂不可用")
    return _MD2IMG_PLUGIN_INSTANCE

def _md2img_tool_extract_ctx_event(wrapper: Any):
    """兼容不同 AstrBot 版本的 ContextWrapper 结构，尽量取到 ctx 与 event。"""
    agent_ctx = getattr(wrapper, "context", None)
    if agent_ctx is None:
        agent_ctx = getattr(wrapper, "_context", None)
    if agent_ctx is None:
        agent_ctx = wrapper
    event = getattr(agent_ctx, "event", None) or getattr(wrapper, "event", None)
    ctx = getattr(agent_ctx, "context", None) or getattr(wrapper, "ctx", None)
    return ctx, event

async def _md2img_tool_send_components(plugin: Any, ctx: Any, event: Any, comps: List[Any]) -> Tuple[bool, Any]:
    """在 Tool 内主动发送消息（图片/PDF 等）。
    返回 (ok, send_ret_or_error)。优先 event.send（可拿到回执/ message_id），失败则回退 ctx.send_message。
    """
    last_err: Any = None
    mc_or_list = plugin._build_msg_chain_from_components(comps)
    try:
        if event is not None and hasattr(event, "send") and callable(getattr(event, "send")):
            send_ret = await event.send(mc_or_list)
            return True, send_ret
    except Exception as e:
        last_err = e
    try:
        if ctx is not None and hasattr(ctx, "send_message") and event is not None:
            await ctx.send_message(event.unified_msg_origin, mc_or_list)
            return True, None
    except Exception as e:
        last_err = e
    return False, last_err

@dataclass
class Md2ImgRenderMarkdownTool(FunctionTool[AstrAgentContext]):
    """把 Markdown 渲染成图片并发送到当前会话。"""
    name: str = "md2img_render_markdown"
    description: str = (
        "将一段 Markdown(含 LaTeX/内联 SVG) 渲染成图片并发送到当前会话。"
        "数学证明/解答必须在 Markdown 开头提供题目或证明目标卡片："
        "`> [problem-card]` 换行 `> **题目/证明目标：** ...`，随后再用 `## 一、证明过程` 等分节。"
    )
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "markdown": {
                "type": "string",
                "description": (
                    "要渲染的 Markdown 文本（可包含 LaTeX 公式和内联 SVG 示意图）。数学内容必须包含浅蓝题目卡片，格式："
                    "`> [problem-card]\\n> **题目/证明目标：** ...\\n\\n## 一、证明过程\\n...`。"
                ),
            },
            "mode": {"type": "string", "description": "渲染模式：mobile 或 pc", "enum": ["mobile", "pc"], "default": "mobile"},
            "scale": {"type": "integer", "description": "缩放倍率（1~3，越大越清晰但更慢）", "minimum": 1, "maximum": 3, "default": 2},
        },
        "required": ["markdown"],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        md_text = str(kwargs.get("markdown") or kwargs.get("md") or "").strip()
        if not md_text:
            return json.dumps({"ok": False, "error": "markdown 不能为空"}, ensure_ascii=False)

        mode = str(kwargs.get("mode") or "mobile").strip().lower()
        if mode not in ("mobile", "pc"):
            mode = "mobile"

        try:
            scale = int(kwargs.get("scale", 2))
        except Exception:
            scale = 2
        scale = max(1, min(3, scale))

        os.makedirs(plugin.IMAGE_CACHE_DIR, exist_ok=True)
        out_path = os.path.join(plugin.IMAGE_CACHE_DIR, f"tool_{uuid.uuid4().hex}.png")

        img_paths = await plugin._markdown_to_image_playwright(
            md_text=md_text,
            output_image_path=out_path,
            scale=scale,
            mode=mode,
        )

        # 发送图片
        try:
            import astrbot.api.message_components as Comp
            comps_all = [Comp.Image.fromFileSystem(p) for p in img_paths]
        except Exception:
            comps_all = [Image.fromFileSystem(p) for p in img_paths]  # type: ignore

        sent = 0
        max_per_msg = int(getattr(plugin, "_cfg", lambda *_: 4)("tool_max_images_per_message", 4) or 4)
        batch: List[Any] = []
        for c in comps_all:
            batch.append(c)
            if len(batch) >= max_per_msg:
                ok, _ = await _md2img_tool_send_components(plugin, ctx, event, batch)
                if ok:
                    sent += len(batch)
                batch = []
        if batch:
            ok, _ = await _md2img_tool_send_components(plugin, ctx, event, batch)
            if ok:
                sent += len(batch)

        return json.dumps(
            {"ok": bool(img_paths), "images": img_paths, "sent": sent, "mode": mode, "scale": scale},
            ensure_ascii=False,
        )

@dataclass
class Md2ImgSolveMathPdfTool(FunctionTool[AstrAgentContext]):
    """调用插件内置 PDF 生成（/pdf）逻辑。"""
    name: str = "md2img_solve_math_pdf"
    description: str = "根据题目文字(可选)+图片URL(可选)生成 LaTeX 并用本地 xelatex 编译 PDF，然后把 PDF 发送到当前会话。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "problem_text": {"type": "string", "description": "题目文字（可为空；若为空需提供 image_urls）"},
            "image_urls": {"type": "array", "items": {"type": "string"}, "description": "题目图片 URL 列表（可为空；若为空需提供 problem_text）"},
            "ref_pdf_latex": {"type": "string", "description": "可选：上一份 PDF 的 LaTeX 源码（用于追问/续写）", "default": ""},
        },
        "required": [],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        problem_text = str(kwargs.get("problem_text") or kwargs.get("text") or "").strip()
        image_urls = kwargs.get("image_urls") or kwargs.get("images") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not isinstance(image_urls, list):
            image_urls = []

        ref_pdf_latex = str(kwargs.get("ref_pdf_latex") or "").strip()

        if (not problem_text) and (not image_urls):
            return json.dumps({"ok": False, "error": "需要提供 problem_text 或 image_urls"}, ensure_ascii=False)

        # 标记：避免其它 on_llm_request 注入干扰
        try:
            if event is not None and hasattr(event, "set_extra"):
                event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        pdf_path, fname, generated_tex_src = await plugin._solve_math_to_pdf(
            problem_text=problem_text,
            event=event,
            image_urls=image_urls,
            ref_pdf_latex=ref_pdf_latex,
        )

        # 发送 PDF
        sent_msg_id: Optional[str] = None
        send_ok = False
        send_error: Any = None
        try:
            import astrbot.api.message_components as Comp
            comps = [Comp.File(file=pdf_path, name=fname)]
            ok, send_ret = await _md2img_tool_send_components(plugin, ctx, event, comps)
            if ok:
                send_ok = True
                sent_msg_id = plugin._extract_message_id_from_send_ret(send_ret)
            else:
                send_error = send_ret
        except Exception as e:
            send_error = e

        # 更新 session 状态（供后续 /pdf 追问）
        try:
            if event is not None:
                skey = _get_session_key(event)
                async with plugin._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    now_ts = time.time()
                    st["last_active_ts"] = now_ts
                    if generated_tex_src:
                        st["last_pdf_context"] = generated_tex_src
                    if problem_text:
                        st["last_problem"] = problem_text
                    else:
                        if image_urls:
                            st["last_problem"] = "[图片题目]"
                    st["last_had_img"] = bool(image_urls)
                    if image_urls:
                        st["last_image_urls"] = list(image_urls)
                        st["last_image_ts"] = now_ts
                    if sent_msg_id and generated_tex_src:
                        mp = st.setdefault("pdf_ctx_map", {})
                        if isinstance(mp, dict):
                            mp[str(sent_msg_id)] = generated_tex_src
                            while len(mp) > 30:
                                try:
                                    mp.pop(next(iter(mp)))
                                except Exception:
                                    break
                        st["last_pdf_msg_id"] = str(sent_msg_id)
        except Exception:
            pass

        if not send_ok:
            plugin._log_pdf_file_send_failure("PDF Tool", pdf_path, fname, send_error)
            return json.dumps(
                {
                    "ok": False,
                    "send_ok": False,
                    "pdf_path": pdf_path,
                    "filename": fname,
                    "error": plugin._pdf_file_send_failure_message(pdf_path, fname, send_error, label="PDF"),
                },
                ensure_ascii=False,
            )

        return json.dumps({"ok": True, "send_ok": True, "pdf_path": pdf_path, "filename": fname, "message_id": sent_msg_id}, ensure_ascii=False)

@dataclass
class Md2ImgSolveMathSpdfTool(FunctionTool[AstrAgentContext]):
    """调用插件内置 DeepThink 多角色 PDF 生成（/spdf）逻辑。"""
    name: str = "md2img_solve_math_spdf"
    description: str = "DeepThink 多角色迭代生成 LaTeX 并本地编译 PDF，然后把 PDF 发送到当前会话。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "problem_text": {"type": "string", "description": "题目文字（可为空；若为空需提供 image_urls）"},
            "image_urls": {"type": "array", "items": {"type": "string"}, "description": "题目图片 URL 列表（可为空；若为空需提供 problem_text）"},
            "ref_pdf_latex": {"type": "string", "description": "可选：上一份 PDF 的 LaTeX 源码（用于追问/续写）", "default": ""},
        },
        "required": [],
    })

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        plugin = _md2img_get_plugin()
        ctx, event = _md2img_tool_extract_ctx_event(context)

        problem_text = str(kwargs.get("problem_text") or kwargs.get("text") or "").strip()
        image_urls = kwargs.get("image_urls") or kwargs.get("images") or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        if not isinstance(image_urls, list):
            image_urls = []

        ref_pdf_latex = str(kwargs.get("ref_pdf_latex") or "").strip()

        if (not problem_text) and (not image_urls):
            return json.dumps({"ok": False, "error": "需要提供 problem_text 或 image_urls"}, ensure_ascii=False)

        try:
            if event is not None and hasattr(event, "set_extra"):
                event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        pdf_path, fname, generated_tex_src = await plugin._solve_math_to_spdf(
            problem_text=problem_text,
            event=event,
            image_urls=image_urls,
            ref_pdf_latex=ref_pdf_latex,
        )

        # 发送 PDF
        sent_msg_id: Optional[str] = None
        send_ok = False
        send_error: Any = None
        try:
            import astrbot.api.message_components as Comp
            comps = [Comp.File(file=pdf_path, name=fname)]
            ok, send_ret = await _md2img_tool_send_components(plugin, ctx, event, comps)
            if ok:
                send_ok = True
                sent_msg_id = plugin._extract_message_id_from_send_ret(send_ret)
            else:
                send_error = send_ret
        except Exception as e:
            send_error = e

        # 更新 session 状态（供后续 /pdf 追问）
        try:
            if event is not None:
                skey = _get_session_key(event)
                async with plugin._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    now_ts = time.time()
                    st["last_active_ts"] = now_ts
                    if generated_tex_src:
                        st["last_pdf_context"] = generated_tex_src
                    if problem_text:
                        st["last_problem"] = problem_text
                    else:
                        if image_urls:
                            st["last_problem"] = "[图片题目]"
                    st["last_had_img"] = bool(image_urls)
                    if image_urls:
                        st["last_image_urls"] = list(image_urls)
                        st["last_image_ts"] = now_ts
                    if sent_msg_id and generated_tex_src:
                        mp = st.setdefault("pdf_ctx_map", {})
                        if isinstance(mp, dict):
                            mp[str(sent_msg_id)] = generated_tex_src
                            while len(mp) > 30:
                                try:
                                    mp.pop(next(iter(mp)))
                                except Exception:
                                    break
                        st["last_pdf_msg_id"] = str(sent_msg_id)
        except Exception:
            pass

        if not send_ok:
            plugin._log_pdf_file_send_failure("SPDF Tool", pdf_path, fname, send_error)
            return json.dumps(
                {
                    "ok": False,
                    "send_ok": False,
                    "pdf_path": pdf_path,
                    "filename": fname,
                    "error": plugin._pdf_file_send_failure_message(pdf_path, fname, send_error, label="SPDF"),
                },
                ensure_ascii=False,
            )

        return json.dumps({"ok": True, "send_ok": True, "pdf_path": pdf_path, "filename": fname, "message_id": sent_msg_id}, ensure_ascii=False)


# 插件定义
# -------------------------------------------------------------------------
@register(
    "astrbot_plugin_mathsolve",
    "blueraina",
    "Markdown转图片 + 数学图文解答 + /pdf LaTeX解答 + /spdf DeepThink多角色迭代 + 知识库检索 + 对话记忆",
    "1.13.4",
)
class MarkdownConverterPlugin(DailyReportMixin, MemoryMixin, SpdfMixin, PdfMixin, RenderMixin, Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config: Dict[str, Any] = dict(config) if isinstance(config, dict) else {}

        self.DATA_DIR = _safe_get_data_dir()
        self.IMAGE_CACHE_DIR = os.path.join(self.DATA_DIR, "md2img_cache")
        self.PDF_CACHE_DIR = os.path.join(self.DATA_DIR, "md2img_pdf_cache")
        self.TEXLIVE_CACHE_DIR = os.path.join(self.PDF_CACHE_DIR, "_texlive_cache")
        self.DAILY_REPORT_DIR = os.path.join(self.DATA_DIR, "daily_reports")
        self.DAILY_REPORT_RECORD_DIR = os.path.join(self.DAILY_REPORT_DIR, "records")
        self.DAILY_REPORT_TEX_DIR = os.path.join(self.DAILY_REPORT_DIR, "tex")
        self.DAILY_REPORT_OUTPUT_DIR = os.path.join(self.DAILY_REPORT_DIR, "reports")

        # Playwright 复用
        self._pw = None
        self._browser = None
        self._pw_lock = asyncio.Lock()
        self._playwright_install_lock = asyncio.Lock()
        self._playwright_install_task: Optional[asyncio.Task] = None
        self._render_sema = asyncio.Semaphore(int(self._cfg("render_concurrency", 2) or 2))
        # xelatex 编译并发限制（避免同时跑太多编译占满 CPU/IO）
        self._tex_compile_sema = asyncio.Semaphore(int(self._cfg("tex_compile_concurrency", 2) or 2))
        # /spdf solver 并发限制
        self._spdf_solver_sema = asyncio.Semaphore(int(self._cfg("spdf_solver_concurrency", 2) or 2))

        # Session 清理任务
        self._session_cleaner_task: Optional[asyncio.Task] = None
        self._cache_cleaner_task: Optional[asyncio.Task] = None
        self._daily_report_task: Optional[asyncio.Task] = None
        self._daily_report_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()  # 最小侵入：统一保护 MATH_SESSION_STATE

        # TeXLive 编译日志
        self._last_texlive_log = ""
        # /pdf provider 连通性预检缓存：provider_id -> {ok, ts, error}
        self._pdf_provider_preflight_cache: Dict[str, Dict[str, Any]] = {}


        # Tool / Function Calling：把本插件能力注册为 LLM Tools，供 Agent 调用
        # 官方文档：>= v4.5.1 使用 self.context.add_llm_tools(...) 注册 Tool
        # 兼容 < v4.5.1：使用 self.context.provider_manager.llm_tools.func_list.append(...)
        global _MD2IMG_PLUGIN_INSTANCE
        _MD2IMG_PLUGIN_INSTANCE = self
        try:
            self.context.add_llm_tools(
                Md2ImgRenderMarkdownTool(),
                Md2ImgSolveMathPdfTool(),
                Md2ImgSolveMathSpdfTool(),
            )
            logger.info("[md2img] 已注册 LLM Tools: md2img_render_markdown / md2img_solve_math_pdf / md2img_solve_math_spdf")
        except Exception as e:
            try:
                tool_mgr = self.context.provider_manager.llm_tools
                tool_mgr.func_list.append(Md2ImgRenderMarkdownTool())
                tool_mgr.func_list.append(Md2ImgSolveMathPdfTool())
                tool_mgr.func_list.append(Md2ImgSolveMathSpdfTool())
                logger.info("[md2img] 已通过旧版接口注册 LLM Tools")
            except Exception as e2:
                logger.warning(f"[md2img] 注册 LLM Tools 失败（不影响其它功能）: {e} / {e2}")

    def _cfg(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = DEFAULT_CFG.get(key)
        try:
            if key in self.config:
                return self.config.get(key)
            # Newer WebUI schema groups settings under object sections. Keep
            # reading legacy flat configs while also accepting grouped configs.
            for section in self.config.values():
                if isinstance(section, dict) and key in section:
                    return section.get(key)
        except Exception:
            pass
        return default

    async def initialize(self):
        try:
            os.makedirs(self.IMAGE_CACHE_DIR, exist_ok=True)
            os.makedirs(self.PDF_CACHE_DIR, exist_ok=True)
            os.makedirs(self.TEXLIVE_CACHE_DIR, exist_ok=True)
            self._daily_report_init_storage()

            if bool(self._cfg("auto_install_playwright_chromium", True)):
                self._schedule_playwright_chromium_install("startup")

            # 启动 Playwright（可选）
            if bool(self._cfg("reuse_playwright_browser", True)):
                await self._ensure_browser()

            # 启动 session 清理任务
            self._start_session_cleaner()
            self._start_cache_cleaner()
            self._start_daily_report_scheduler()

            logger.info("AstrBot mathsolve 插件已就绪 (1.13.4 - 本地编译版)")
        except Exception as e:
            logger.error(f"初始化失败: {e}")

    async def terminate(self):
        try:
            self._stop_daily_report_scheduler()
        except Exception:
            pass

    def _schedule_playwright_chromium_install(self, reason: str = "") -> None:
        if not bool(self._cfg("auto_install_playwright_chromium", True)):
            return
        try:
            if self._playwright_install_task is not None and not self._playwright_install_task.done():
                return
            self._playwright_install_task = asyncio.create_task(
                self._install_playwright_chromium_background(reason)
            )
        except Exception as e:
            logger.warning(f"Playwright Chromium 后台安装任务启动失败: {e}")

    async def _install_playwright_chromium_background(self, reason: str = "") -> None:
        async with self._playwright_install_lock:
            timeout_sec = int(self._cfg("playwright_install_timeout_sec", 600) or 600)
            reason_suffix = f" reason={reason}" if reason else ""
            logger.info(f"开始后台安装/确认 Playwright Chromium{reason_suffix}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "playwright",
                    "install",
                    "chromium",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=max(1, timeout_sec))
                out_text = (stdout or b"").decode("utf-8", errors="ignore").strip()
                err_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
                if proc.returncode == 0:
                    tail = (out_text or err_text)[-1200:]
                    logger.info(f"Playwright Chromium 安装/确认完成{(': ' + tail) if tail else ''}")
                else:
                    tail = ((err_text or "") + "\n" + (out_text or "")).strip()[-2000:]
                    logger.error(f"Playwright Chromium 安装失败，returncode={proc.returncode}: {tail}")
            except asyncio.TimeoutError:
                try:
                    proc.kill()  # type: ignore[name-defined]
                except Exception:
                    pass
                logger.error(f"Playwright Chromium 后台安装超时: {timeout_sec}s")
            except Exception as e:
                logger.error(f"Playwright Chromium 后台安装异常: {e}")

    # ------------------------- 生命周期：尽量释放资源 -------------------------
    async def _ensure_browser(self):
        if self._browser is not None:
            return
        async with self._pw_lock:
            if self._browser is not None:
                return
            try:
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch()
                logger.info("Playwright Chromium 已启动（复用模式）")
            except Exception as e:
                logger.error(f"Playwright 启动失败，将回退为每次渲染临时启动: {e}")
                self._schedule_playwright_chromium_install("launch_failed")
                self._pw = None
                self._browser = None

    async def _close_browser(self):
        async with self._pw_lock:
            try:
                if self._browser is not None:
                    await self._browser.close()
                if self._pw is not None:
                    await self._pw.stop()
            except Exception:
                pass
            self._browser = None
            self._pw = None

    # ------------------------- Session 清理 -------------------------


    # === 全局消息监听：记录用户最后发的图片 & 活跃时间 ===
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        try:
            imgs = await self._pdf_get_event_image_inputs(event)
            skey = _get_session_key(event)
            now_ts = time.time()
            raw = (getattr(event, "message_str", "") or "").strip()

            is_private = self._is_private_chat(event)
            reply_id = self._extract_reply_msg_id(event)
            cur_mid = self._extract_current_msg_id(event)

            async with self._state_lock:
                state = MATH_SESSION_STATE.setdefault(skey, {})
                state["last_active_ts"] = now_ts

                # --- 图片消息：记录图片；私聊纯图片则进入“待引用”状态 ---
                if imgs:
                    state["last_image_urls"] = imgs
                    state["last_image_ts"] = now_ts
                    state["last_had_img"] = True
                    if not raw:
                        state["last_problem"] = "[图片题目]"
                    if cur_mid:
                        state["last_image_msg_id"] = cur_mid
                        image_map = state.setdefault("image_ctx_map", {})
                        if isinstance(image_map, dict):
                            image_map[str(cur_mid)] = list(imgs)
                            max_keep = 50
                            while len(image_map) > max_keep:
                                try:
                                    image_map.pop(next(iter(image_map)))
                                except Exception:
                                    break

                    if not raw:
                        # 只发图片：进入“待引用”状态，等待用户“引用该图片 + 文字”再答疑
                        state["pending_image_only"] = True
                        if cur_mid:
                            state["pending_image_msg_id"] = cur_mid
                        state["pending_image_ts"] = now_ts
                    else:
                        # 非“纯图片待引用”场景：清掉 pending，并正常记录 last_problem
                        state.pop("pending_image_only", None)
                        state.pop("pending_image_msg_id", None)
                        state.pop("pending_image_ts", None)
                        if raw:
                            state["last_problem"] = raw
                        else:
                            state["last_problem"] = "[图片题目]"

                # --- 非图片消息：如果上一条是“私聊纯图片待引用”，则仅在回复/引用它时保留 pending ---
                else:
                    if state.get("pending_image_only"):
                        pid = str(state.get("pending_image_msg_id") or "").strip()
                        # 若本条不是“引用 pending 图片”的消息，则认为用户已开始新话题，清掉 pending
                        if not (reply_id and ((not pid) or (str(reply_id).strip() == pid))):
                            state.pop("pending_image_only", None)
                            state.pop("pending_image_msg_id", None)
                            state.pop("pending_image_ts", None)

            # 私聊纯图片：直接 stop_event，阻止默认“收到就回”行为；群聊仍交给平台/其它逻辑处理
            if is_private and imgs and (not raw):
                try:
                    event.stop_event()
                except Exception:
                    pass
        except Exception:
            pass


    # ------------------------- Reply/引用消息识别（用于 PDF 二次追问） -------------------------
    def _extract_reply_msg_id(self, event: AstrMessageEvent) -> Optional[str]:
        """尽量从不同适配器的 event/message_obj 中提取“引用/回复”的原消息 ID。

        说明：不同 QQ 适配器（NapCat/OneBot/mirai 等）字段差异很大，所以这里做多路兜底：
        1) message_obj 常见字段（reply/quote/source 等）
        2) message 组件中可能存在的 Reply/Quote 组件
        3) raw_message（CQ 码 / JSON 片段）正则提取
        """
        try:
            msg_obj = getattr(event, "message_obj", None)

            # --- 1) message_obj 直接字段 ---
            if msg_obj is not None:
                for attr in (
                        "reply_id", "reply_msg_id", "reply_message_id", "replyMessageId",
                        "quote_id", "quote_msg_id", "quote_message_id", "quoted_msg_id",
                        "source_id", "source_msg_id", "source_message_id",
                        "reply", "quote", "reference", "source",
                ):
                    try:
                        v = getattr(msg_obj, attr, None)
                    except Exception:
                        v = None
                    if v is None:
                        continue

                    # 直接是 id
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()

                    # dict 里找 id
                    if isinstance(v, dict):
                        for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                            vv = v.get(k)
                            if isinstance(vv, (str, int)) and str(vv).strip():
                                return str(vv).strip()

                    # object 里找 id
                    for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                        try:
                            vv = getattr(v, k, None)
                        except Exception:
                            vv = None
                        if isinstance(vv, (str, int)) and str(vv).strip():
                            return str(vv).strip()

                # --- 2) message 组件中可能存在 Reply/Quote ---
                try:
                    comps = getattr(msg_obj, "message", None) or []
                except Exception:
                    comps = []
                for comp in comps:
                    try:
                        cname = (comp.__class__.__name__ or "").lower()
                    except Exception:
                        cname = ""
                    if cname in ("reply", "quote", "reference", "replymessage", "quotemessage", "source"):
                        for k in ("id", "message_id", "msg_id", "seq", "reply_id", "source_id", "messageId"):
                            try:
                                vv = getattr(comp, k, None)
                            except Exception:
                                vv = None
                            if isinstance(vv, (str, int)) and str(vv).strip():
                                return str(vv).strip()
                        # 有的组件会把原消息塞在 data 里
                        try:
                            data = getattr(comp, "data", None)
                        except Exception:
                            data = None
                        if isinstance(data, dict):
                            for k in ("id", "message_id", "msg_id", "seq", "messageId"):
                                vv = data.get(k)
                                if isinstance(vv, (str, int)) and str(vv).strip():
                                    return str(vv).strip()
        except Exception:
            pass

        # --- 3) raw_message（CQ 码 / JSON）兜底 ---
        raws: List[Any] = []
        try:
            raws.append(getattr(event, "raw_message", None))
        except Exception:
            pass

        # 很多适配器把原始事件挂在 message_obj 上，而不是 event 上
        if msg_obj is not None:
            for attr in ("raw_message", "raw_event", "raw", "_raw_message", "_raw_event"):
                try:
                    v = getattr(msg_obj, attr, None)
                    if v is not None:
                        raws.append(v)
                except Exception:
                    continue

        for raw in raws:
            if not raw:
                continue
            s = str(raw)

            # OneBot v11 / CQ 码： [CQ:reply,id=123]
            m = re.search(r"\[CQ:reply,id=([0-9A-Za-z_-]+)\]", s)
            if m:
                return m.group(1)

            # OneBot v12 / JSON 片段： {"type":"reply","data":{"id":"123"}}
            m = re.search(r'"type"\s*:\s*"reply"[\s\S]*?"id"\s*:\s*"?([0-9A-Za-z_-]+)"?', s)
            if m:
                return m.group(1)

            # 其他可能的字段
            m = re.search(r'(?:reply|quote)[_\- ]?(?:id|msg_id|message_id)\s*[:=]\s*"?([0-9A-Za-z_-]+)"?',
                          s, flags=re.I)
            if m:
                return m.group(1)

        return None

    def _is_private_chat(self, event: AstrMessageEvent) -> bool:
        """尽量判断是否为私聊（不同适配器字段差异较大，做多路兜底）。"""
        # 1) 若事件本身提供 is_private() 之类的方法
        for fn_name in ("is_private", "isPrivate", "is_private_chat"):
            fn = getattr(event, fn_name, None)
            if callable(fn):
                try:
                    v = fn()
                    if isinstance(v, bool):
                        return v
                except Exception:
                    pass

        # 2) 常见：群/频道字段存在则视作非私聊
        msg_obj = getattr(event, "message_obj", None)
        for attr in (
            "group_id", "groupId", "group", "guild_id", "guildId", "guild",
            "channel_id", "channelId", "channel", "room_id", "roomId", "room",
            "server_id", "serverId", "chat_id", "chatId"
        ):
            v = None
            try:
                if msg_obj is not None:
                    v = getattr(msg_obj, attr, None)
            except Exception:
                v = None
            if not v:
                try:
                    v = getattr(event, attr, None)
                except Exception:
                    v = None
            if isinstance(v, (str, int)) and str(v).strip() and str(v).strip() != "0":
                return False

        # 3) unified_msg_origin 兜底：包含 group/guild/channel/room 等字样视为非私聊
        try:
            umo = getattr(event, "unified_msg_origin", None)
            if isinstance(umo, str) and re.search(r"(group|guild|channel|room|server)", umo, flags=re.I):
                return False
        except Exception:
            pass

        return True

    def _extract_current_msg_id(self, event: AstrMessageEvent) -> Optional[str]:
        """尽量从 event/message_obj 中提取“当前这条消息”的 message_id。"""
        try:
            msg_obj = getattr(event, "message_obj", None)
            for root in (msg_obj, event, getattr(event, "raw_message", None)):
                if root is None:
                    continue
                if isinstance(root, (str, int)) and str(root).strip():
                    # raw_message 可能是文本/JSON，不直接作为 msg_id
                    pass
                # 常见字段
                for attr in ("message_id", "msg_id", "id", "messageId", "seq"):
                    try:
                        v = getattr(root, attr, None)
                    except Exception:
                        v = None
                    if isinstance(v, (str, int)) and str(v).strip():
                        return str(v).strip()
                # dict 里找
                if isinstance(root, dict):
                    for k in ("message_id", "msg_id", "id", "messageId", "seq"):
                        v = root.get(k)
                        if isinstance(v, (str, int)) and str(v).strip():
                            return str(v).strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _attach_images_to_req(req: ProviderRequest, image_urls: List[str], replace_existing: bool = False) -> None:
        """把图片列表尽量塞进 ProviderRequest（不同 AstrBot 版本字段名可能不同）。"""
        if not image_urls:
            return

        tried_attrs = (
            "images", "image_urls", "image_url_list", "image_inputs", "vision_inputs",
            "input_images", "image_paths", "files", "attachments",
        )

        for attr in tried_attrs:
            if not hasattr(req, attr):
                continue
            try:
                cur = getattr(req, attr, None)
                if replace_existing or cur is None:
                    setattr(req, attr, list(image_urls))
                    return
                if isinstance(cur, list):
                    for u in image_urls:
                        if u not in cur:
                            cur.append(u)
                    setattr(req, attr, cur)
                    return
                if isinstance(cur, tuple):
                    merged = list(cur)
                    for u in image_urls:
                        if u not in merged:
                            merged.append(u)
                    setattr(req, attr, merged)
                    return
                # 其他类型：尝试直接覆盖
                setattr(req, attr, list(image_urls))
                return
            except Exception:
                continue

        # 最后兜底：放进 prompt（不保证所有视觉模型都能用 URL 取图，但至少不丢信息）
        try:
            req.prompt = (req.prompt or "") + "\n[用户引用的图片链接]\n" + "\n".join(image_urls) + "\n"
        except Exception:
            pass


    @staticmethod
    def _disable_llm_tools_in_req(req: ProviderRequest) -> None:
        """在 PDF 生成内部调用时，尽量禁用 tools/function-calling，避免部分上游返回 tool_calls 造成解析失败。"""
        try:
            # 常见字段：tools / functions
            for attr in (
                "tools", "functions",
                "tool_definitions", "function_definitions",
                "available_tools", "toolset", "tool_sets",
            ):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, [])
                    except Exception:
                        pass

            # 常见开关
            for attr in ("enable_tools", "enable_function_call", "use_tools"):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, False)
                    except Exception:
                        pass

            # OpenAI 兼容字段
            for attr in ("tool_choice", "function_call"):
                if hasattr(req, attr):
                    try:
                        setattr(req, attr, "none")
                    except Exception:
                        pass

            # 有些实现把额外参数放在 dict 里
            for attr in ("extra", "extra_params", "kwargs", "options", "provider_kwargs", "request_kwargs"):
                try:
                    d = getattr(req, attr, None)
                except Exception:
                    d = None
                if isinstance(d, dict):
                    d["tools"] = []
                    d["functions"] = []
                    d["tool_choice"] = "none"
                    d["function_call"] = "none"
                    try:
                        setattr(req, attr, d)
                    except Exception:
                        pass
        except Exception:
            pass

    def _reply_msg_has_pdf_hint(self, event: AstrMessageEvent) -> bool:
        """尝试判断用户引用的那条消息是否包含 PDF 文件（若适配器提供了被引用消息内容）。

        注意：很多 QQ 协议在 reply 里只给 message_id，并不给原消息内容。
        因此这个函数只能“尽量判断”，判断不到就返回 False。
        """
        try:
            msg_obj = getattr(event, "message_obj", None)
        except Exception:
            msg_obj = None

        candidates: List[Any] = []
        for root in (msg_obj, getattr(event, "raw_message", None)):
            if root is not None:
                candidates.append(root)

        # 额外尝试：某些适配器会把 quote/reply 的原消息内容放在 message_obj.reply/message_obj.quote 里
        if msg_obj is not None:
            for attr in ("reply", "quote", "reference", "source", "origin", "original_message"):
                try:
                    v = getattr(msg_obj, attr, None)
                except Exception:
                    v = None
                if v is not None:
                    candidates.append(v)

        def _looks_like_pdf_string(x: str) -> bool:
            xl = (x or "").lower()
            return (".pdf" in xl) or ("application/pdf" in xl)

        # 尝试在对象图里找 ".pdf" / File 组件 / file name
        try:
            for root in candidates:
                for obj, _depth in _walk_object_graph(root, max_depth=6, max_nodes=2000):
                    if isinstance(obj, str) and _looks_like_pdf_string(obj):
                        return True
                    if isinstance(obj, dict):
                        for k in ("name", "filename", "file", "path", "url", "mime", "mimetype"):
                            v = obj.get(k)
                            if isinstance(v, str) and _looks_like_pdf_string(v):
                                return True
                    try:
                        cname = (obj.__class__.__name__ or "").lower()
                    except Exception:
                        cname = ""
                    if "file" in cname and "pdf" in cname:
                        return True
        except Exception:
            pass

        return False

    def _build_msg_chain_from_components(self, comps: List[Any]) -> Any:
        """把 Comp.* 组件列表转换成可用于 event.send 的对象（尽量兼容不同 AstrBot 版本）。"""
        comps = list(comps or [])

        # 1) 有的版本支持 MessageChain(list)
        try:
            return MessageChain(comps)  # type: ignore
        except Exception:
            pass

        # 2) 常规：MessageChain() 然后 append/extend/chain
        try:
            mc = MessageChain()
        except Exception:
            mc = None

        if mc is not None:
            for mname in ("chain", "extend", "append", "add"):
                fn = getattr(mc, mname, None)
                if callable(fn):
                    try:
                        if mname in ("extend", "chain"):
                            fn(comps)
                        else:
                            for c in comps:
                                fn(c)
                        return mc
                    except Exception:
                        continue

            # 3) 极端兜底：直接写入 message 列表
            try:
                lst = getattr(mc, "message", None)
                if isinstance(lst, list):
                    lst.extend(comps)
                    return mc
            except Exception:
                pass

        # 4) 最后兜底：直接返回组件列表（部分版本 event.send 可以直接吃 list）
        return comps

    @staticmethod
    def _extract_message_id_from_send_ret(send_ret: Any) -> Optional[str]:
        """从 event.send(...) 的返回值中尽量提取 message_id。"""
        if send_ret is None:
            return None
        if isinstance(send_ret, (str, int)) and str(send_ret).strip():
            return str(send_ret).strip()

        if isinstance(send_ret, list) and send_ret:
            # 有的版本会返回 list[receipt]
            return MarkdownConverterPlugin._extract_message_id_from_send_ret(send_ret[0])

        if isinstance(send_ret, dict):
            for k in ("message_id", "msg_id", "id", "messageId", "seq"):
                v = send_ret.get(k)
                if isinstance(v, (str, int)) and str(v).strip():
                    return str(v).strip()
            # 可能嵌套在 data/result 里
            for kk in ("data", "result", "message"):
                v = send_ret.get(kk)
                if isinstance(v, dict):
                    mid = MarkdownConverterPlugin._extract_message_id_from_send_ret(v)
                    if mid:
                        return mid

        for attr in ("message_id", "msg_id", "id", "messageId", "seq"):
            try:
                v = getattr(send_ret, attr, None)
            except Exception:
                v = None
            if isinstance(v, (str, int)) and str(v).strip():
                return str(v).strip()

        return None

    @staticmethod
    def _astrbot_callback_api_base() -> str:
        """读取 AstrBot 全局文件回调地址；为空时本地 File 会直接交给协议端读取。"""
        try:
            from astrbot.core import astrbot_config
            return str(astrbot_config.get("callback_api_base", "") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _pdf_local_file_info(pdf_path: str) -> Tuple[str, bool, int]:
        try:
            p = Path(str(pdf_path)).expanduser()
            resolved = str(p.resolve())
            exists = p.is_file()
            size = int(p.stat().st_size) if exists else 0
            return resolved, exists, size
        except Exception:
            return str(pdf_path), False, 0

    @staticmethod
    def _short_error_text(err: Any, limit: int = 700) -> str:
        text = str(err or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _log_pdf_file_send_failure(self, label: str, pdf_path: str, fname: str, err: Any) -> None:
        resolved, exists, size = self._pdf_local_file_info(pdf_path)
        callback_api_base = self._astrbot_callback_api_base()
        logger.error(
            f"{label} 文件发送失败: name={fname} path={resolved} "
            f"exists={exists} size={size} callback_api_base={callback_api_base or '<empty>'} "
            f"err={self._short_error_text(err, 1000)}"
        )

    def _pdf_file_send_failure_message(self, pdf_path: str, fname: str, err: Any, label: str = "PDF") -> str:
        resolved, exists, size = self._pdf_local_file_info(pdf_path)
        callback_api_base = self._astrbot_callback_api_base()
        if not exists:
            return (
                f"❌ {label} 已生成流程结束，但本地文件不存在，无法发送。\n"
                f"文件名：{fname}\n"
                f"路径：{resolved}\n"
                "请查看插件日志里的 PDF 生成/缓存目录错误。"
            )

        if callback_api_base:
            hint = (
                f"已检测到 callback_api_base：{callback_api_base}\n"
                "请确认 NapCat/OneBot 协议端能访问这个地址；如果 AstrBot 在容器里，也要确认端口已对外暴露。"
            )
        else:
            hint = (
                "当前 AstrBot 的 callback_api_base 为空，AstrBot 会把本地文件路径直接交给 NapCat/OneBot。\n"
                "如果协议端和 AstrBot 不在同一个文件系统/容器里，就会出现 ENOENT。"
            )

        err_text = self._short_error_text(err, 500)
        return (
            f"⚠️ {label} 已生成，但发送文件失败。\n"
            f"文件名：{fname}\n"
            f"大小：{size} bytes\n"
            f"服务器路径：{resolved}\n\n"
            f"{hint}\n\n"
            "处理方式：在 AstrBot WebUI 配置可被协议端访问的 callback_api_base，"
            "或让协议端容器/进程挂载并能读取同一个 plugins_data 路径。"
            + (f"\n\n错误：{err_text}" if err_text else "")
        )

    async def _send_pdf_file_component(
        self,
        event: AstrMessageEvent,
        Comp: Any,
        chain_prefix: List[Any],
        pdf_path: str,
        fname: str,
        label: str = "PDF",
    ) -> Tuple[Optional[str], Optional[str]]:
        resolved, exists, _size = self._pdf_local_file_info(pdf_path)
        if not exists:
            err = FileNotFoundError(resolved)
            self._log_pdf_file_send_failure(label, pdf_path, fname, err)
            return None, self._pdf_file_send_failure_message(pdf_path, fname, err, label=label)

        chain = list(chain_prefix)
        chain.append(Comp.File(file=resolved, name=fname))

        try:
            mc_or_chain = self._build_msg_chain_from_components(chain)
            send_ret = await event.send(mc_or_chain)
            return self._extract_message_id_from_send_ret(send_ret), None
        except Exception as e:
            self._log_pdf_file_send_failure(label, pdf_path, fname, e)
            return None, self._pdf_file_send_failure_message(pdf_path, fname, e, label=label)

    # === 快捷指令 1: PC 模式 ===
    @filter.command("pc")
    async def cmd_pc(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if not user_id:
            yield event.plain_result("无法获取用户ID")
            return
        USER_PREFERENCES[user_id] = "pc"
        yield event.plain_result("✅ 已切换为 PC 模式 (A4分页)")

    # === 快捷指令 2: Mobile/PE 模式 ===
    @filter.command("pe")
    async def cmd_pe(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if not user_id:
            yield event.plain_result("无法获取用户ID")
            return
        USER_PREFERENCES[user_id] = "mobile"
        yield event.plain_result("✅ 已切换为 手机 模式 (长截屏)")


    # === 快捷指令：清空本会话对话记忆 ===
    @filter.command("memclear")
    async def cmd_memclear(self, event: AstrMessageEvent):
        skey = _get_session_key(event)

        # 1) 清除插件自身维护的记忆（追问上下文、上一题、PDF 上下文等）
        async with self._state_lock:
            st = MATH_SESSION_STATE.setdefault(skey, {})
            st.pop("chat_history", None)
            st.pop("last_problem", None)
            st.pop("last_image_urls", None)
            st.pop("last_image_ts", None)
            st.pop("last_had_img", None)
            st.pop("last_image_msg_id", None)
            st.pop("pending_image_only", None)
            st.pop("pending_image_msg_id", None)
            st.pop("pending_image_ts", None)
            st.pop("image_ctx_map", None)
            st.pop("last_pdf_context", None)
            st.pop("last_pdf_msg_id", None)
            st.pop("pdf_ctx_map", None)

        # 2) 清除 AstrBot 框架层面的对话上下文（多种方式兼容不同版本）
        umo = getattr(event, "unified_msg_origin", None) or ""
        framework_cleared = False

        # --- 方式 A: self.context.conversation_flow（v4.x 常见） ---
        try:
            cf = getattr(self.context, "conversation_flow", None)
            if cf is not None:
                # 有些版本是 dict[umo] -> list[messages]
                if isinstance(cf, dict) and umo and umo in cf:
                    cf[umo] = []
                    framework_cleared = True
                # 有些版本有 clear / reset 方法
                for fn_name in ("clear", "reset", "clear_history", "delete"):
                    fn = getattr(cf, fn_name, None)
                    if callable(fn):
                        try:
                            r = fn(umo)
                            if asyncio.iscoroutine(r):
                                await r
                            framework_cleared = True
                            break
                        except Exception:
                            continue
        except Exception:
            pass

        # --- 方式 B: provider_manager.session_memory / conversations ---
        if not framework_cleared:
            try:
                pm = getattr(self.context, "provider_manager", None)
                if pm is not None:
                    for attr in ("session_memory", "conversations", "conversation_map",
                                 "contexts", "session_contexts", "chat_contexts"):
                        store = getattr(pm, attr, None)
                        if isinstance(store, dict) and umo and umo in store:
                            store[umo] = [] if isinstance(store.get(umo), list) else {}
                            framework_cleared = True
                            break
                    # 有些版本 provider_manager 有 clear_conversation 等方法
                    if not framework_cleared:
                        for fn_name in ("clear_conversation", "reset_conversation",
                                        "clear_session", "delete_conversation",
                                        "clear_context"):
                            fn = getattr(pm, fn_name, None)
                            if callable(fn):
                                try:
                                    r = fn(umo)
                                    if asyncio.iscoroutine(r):
                                        await r
                                    framework_cleared = True
                                    break
                                except Exception:
                                    continue
            except Exception:
                pass

        # --- 方式 C: self.context 上直接有 clear / reset 方法 ---
        if not framework_cleared:
            for fn_name in ("clear_conversation", "reset_conversation",
                            "clear_session", "delete_conversation_context",
                            "clear_context"):
                fn = getattr(self.context, fn_name, None)
                if callable(fn):
                    try:
                        r = fn(umo)
                        if asyncio.iscoroutine(r):
                            await r
                        framework_cleared = True
                        break
                    except Exception:
                        continue

        # --- 方式 D: 通过数据库清除（部分 AstrBot 版本将对话历史持久化到 DB） ---
        if not framework_cleared:
            try:
                db = None
                for attr in ("db", "db_helper", "_db", "database"):
                    db = getattr(self.context, attr, None)
                    if db is not None:
                        break
                if db is None:
                    for fn_name in ("get_db", "get_registered_db", "get_database"):
                        fn = getattr(self.context, fn_name, None)
                        if callable(fn):
                            try:
                                db = fn()
                                if asyncio.iscoroutine(db):
                                    db = await db
                            except Exception:
                                db = None
                            if db is not None:
                                break
                if db is not None:
                    for fn_name in ("clear_session_history", "delete_conversation",
                                    "clear_conversation", "delete_history",
                                    "clear_history", "remove_conversation"):
                        fn = getattr(db, fn_name, None)
                        if callable(fn):
                            try:
                                r = fn(umo)
                                if asyncio.iscoroutine(r):
                                    await r
                                framework_cleared = True
                                break
                            except Exception:
                                continue
            except Exception:
                pass

        if framework_cleared:
            yield event.plain_result("✅ 已清空本会话的全部对话记忆（含框架上下文）")
        else:
            yield event.plain_result(
                "✅ 已清空本插件的对话记忆\n"
                "⚠️ 未能自动清除 AstrBot 框架的上下文记忆，"
                "如仍有残留记忆，可尝试在 AstrBot 管理面板中手动删除该会话。"
            )

    # === 快捷指令：手动生成当前会话答疑报告 ===
    @filter.command("答疑报告生成")
    async def cmd_daily_report_generate(self, event: AstrMessageEvent):
        yield event.plain_result("正在整理当前会话的答疑记录并生成报告...")
        try:
            msg = await self._daily_report_handle_manual_command(event)
        except Exception as e:
            logger.error(f"手动答疑报告生成失败: {e}")
            msg = "答疑报告生成失败：\n" + str(e)[:1200]
        if msg:
            yield event.plain_result(msg)

    # === 快捷指令 3: 生成 PDF 解答 ===
    @filter.command("pdf")
    async def cmd_pdf(self, event: AstrMessageEvent):
        if not self._cfg("enable_pdf_output", True):
            yield event.plain_result("PDF 输出未启用")
            return

        raw = (getattr(event, "message_str", "") or "").strip()
        arg = ""
        m = re.match(r"^\s*/?pdf(?:\s+([\s\S]*))?$", raw, flags=re.I)
        if m:
            arg = (m.group(1) or "").strip()

        current_images = await self._pdf_get_event_image_inputs(event)
        skey = _get_session_key(event)

        async with self._state_lock:
            state = MATH_SESSION_STATE.get(skey, {}).copy()

        last_problem_text = (state.get("last_problem", "") or "").strip()
        last_images = state.get("last_image_urls", [])
        last_image_ts = float(state.get("last_image_ts", 0) or 0)
        reply_id = self._extract_reply_msg_id(event)
        reply_image_inputs: List[str] = []
        if reply_id:
            try:
                image_map = state.get("image_ctx_map", {})
                if isinstance(image_map, dict):
                    cached = image_map.get(str(reply_id)) or []
                    if isinstance(cached, list):
                        reply_image_inputs = [str(x) for x in cached if str(x or "").strip()]
                if (not reply_image_inputs) and str(reply_id) == str(state.get("last_image_msg_id", "") or "").strip():
                    cached = state.get("last_image_urls") or []
                    if isinstance(cached, list):
                        reply_image_inputs = [str(x) for x in cached if str(x or "").strip()]
            except Exception:
                reply_image_inputs = []

        final_text_input = ""
        final_image_inputs: List[str] = []

        now_ts = time.time()
        valid_sec = int(self._cfg("last_image_valid_sec", 3600) or 3600)
        is_last_img_likely_valid = (now_ts - last_image_ts) <= valid_sec

        if current_images:
            final_image_inputs = current_images
            final_text_input = arg
        elif reply_image_inputs:
            # 精确引用了某条图片消息：优先使用该图片，不依赖 smart 文本判断。
            final_image_inputs = reply_image_inputs
            final_text_input = arg
        elif arg:
            final_text_input = arg

            # 文字 + 上一张图（追问场景）
            # 新逻辑：默认 smart —— 仅在文本看起来是在追问/引用图片时才携带上一张图。
            reuse_mode = str(self._cfg("pdf_reuse_last_image_with_text_mode", "smart") or "smart").lower().strip()
            followup_max_len = int(self._cfg("pdf_followup_text_max_len", 40) or 40)

            allow_reuse = False
            if last_images and is_last_img_likely_valid:
                # 只有“上一题确实是图片题”时才考虑复用（避免跨题串图）
                last_had_img = bool(state.get("last_had_img", False))
                last_img_context = last_had_img or (last_problem_text == "[图片题目]")
                if last_img_context:
                    pending_mid = str(state.get("pending_image_msg_id") or state.get("last_image_msg_id") or "").strip()
                    if reply_id and ((not pending_mid) or (str(reply_id).strip() == pending_mid)):
                        allow_reuse = True
                    elif reuse_mode == "always":
                        allow_reuse = True
                    elif reuse_mode == "never":
                        allow_reuse = False
                    else:
                        # smart 模式：出现明显“追问/引用图片”的信号 or 文本较短且不像新题
                        t = (arg or "").strip()
                        # 用户明确要求“不用图片”
                        if re.search(r"(不用|不使用|别|不要).{0,6}(图片|图|图像)", t):
                            allow_reuse = False
                        elif re.search(
                                r"(如图|见图|图中|图片|上题|上一题|刚才|继续|小问|第\s*(?:\d+|[一二两三四五六七八九十])\s*问|\(\s*\d+\s*\))",
                                t):
                            allow_reuse = True
                        elif len(t) <= max(10, followup_max_len) and (not _is_math_question(t)):
                            allow_reuse = True
                        else:
                            allow_reuse = False

            if allow_reuse:
                final_image_inputs = last_images
        elif last_images:
            # 只发 /pdf：回溯上一张图
            if not is_last_img_likely_valid:
                yield event.plain_result("⚠️ 上一张图片可能已过期（链接可能失效），建议重新发送图片 + /pdf")
                # 仍允许用户“试一试”：很多平台 URL 实际更久
            final_image_inputs = last_images
            final_text_input = last_problem_text if last_problem_text != "[图片题目]" else ""
        elif last_problem_text and last_problem_text != "[图片题目]":
            final_text_input = last_problem_text
        else:
            yield event.plain_result("请发送 /pdf 题目文字，或发送包含题目的图片 + /pdf")
            return

        # === 随机等待提示语 Start ===
        yield event.plain_result("正在生成中...\n你知道吗？" + _get_daily_math_fact())
        # === 随机等待提示语 End ===

        # 若用户“引用/回复”了机器人之前发出的 PDF 文件消息：尽量把对应 PDF 的 LaTeX 源码取出来，
        # 作为本次 /pdf 追问的参考上下文（避免模型遗忘上一份 PDF 的细节）。
        ref_pdf_ctx = ""
        if reply_id:
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    ref_pdf_ctx = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                ref_pdf_ctx = ""

            # 兜底：没有精确 message_id 映射时，弱绑定到“最近一次 PDF”
            if not ref_pdf_ctx:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        if str(reply_id) == last_mid:
                            ref_pdf_ctx = last_pdf_ctx
                        else:
                            # 若适配器能拿到被引用消息内容且确实包含 PDF，则允许兜底
                            if self._reply_msg_has_pdf_hint(event):
                                ref_pdf_ctx = last_pdf_ctx
                    else:
                        ref_pdf_ctx = last_pdf_ctx

        # 标记：本次 LLM 调用用于 /pdf 生成，避免其它 on_llm_request 逻辑误注入
        try:
            event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        try:
            pdf_path, fname, generated_tex_src = await self._solve_math_to_pdf(
                problem_text=final_text_input,
                event=event,
                image_urls=final_image_inputs,
                ref_pdf_latex=ref_pdf_ctx,
            )

            # 保存 PDF 上下文 + 更新“上一题/上一张图”状态
            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                now_ts = time.time()
                st["last_active_ts"] = now_ts
                if generated_tex_src:
                    st["last_pdf_context"] = generated_tex_src

                # 记录更干净的“上一题”（去掉 /pdf 前缀）
                clean_last = (final_text_input or "").strip()
                if clean_last:
                    clean_last = re.sub(r"^\s*/?pdf\b", "", clean_last, flags=re.I).strip()
                if clean_last:
                    st["last_problem"] = clean_last
                else:
                    # 没有文字则标记为图片题
                    if final_image_inputs:
                        st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(final_image_inputs)

                if final_image_inputs:
                    st["last_image_urls"] = list(final_image_inputs)
                    st["last_image_ts"] = now_ts
        except Exception as e:
            err = str(e)
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    log_snip = log_snip[-2000:]
                logger.error(f"PDF 生成失败: {err}\nCompile log snippet:\n{log_snip}")
            except Exception:
                logger.error(f"PDF 生成失败: {err}")

            if "Upstream Error" in err:
                _sn = (event.get_sender_name() or "我").strip() or "我"
                yield event.plain_result(f"呜哇，{_sn} 够不到那张图片啦！它好像溜走了（链接过期或拒绝访问）。😣\n能不能麻烦你重新截图再发给我一次?")
            elif "PDF 模型调用全部失败" in err:
                yield event.plain_result(
                    "❌ PDF 生成失败：已按顺序尝试所有配置的模型，但都超时或调用失败。\n"
                    "请检查 pdf_provider_id 与候补模型 1~5 里的 provider_id 是否可用，"
                    "或在 WebUI 里适当调大对应模型的超时时间。\n\n"
                    + err[:700]
                )
            else:
                # 给用户一个简短的日志末尾，方便快速定位是“缺包/字体”还是“内容里有非法字符”
                msg = "哎呀，PDF 生成失败了，看起来是排版出了点小差错，重问一次试试吧~"
                try:
                    log_snip = (self._last_texlive_log or "").strip()
                    if log_snip:
                        tail_lines = "\n".join(log_snip.splitlines()[-20:])
                        tail_lines = tail_lines[-1500:]
                        msg += "\n\nxelatex 日志末尾(截取)：\n" + tail_lines
                except Exception:
                    pass
                yield event.plain_result(msg)
            return

        # 发送 PDF 文件
        try:
            import astrbot.api.message_components as Comp
        except Exception:
            yield event.plain_result("当前 AstrBot 版本缺少文件消息组件，无法发送 PDF")
            return

        uid = event.get_sender_id()
        chain = []

        # === 修复开始：仅在非私聊（群聊）场景下才添加 @ ===
        is_private = self._is_private_chat(event)
        if uid and (not is_private):
            chain.append(Comp.At(qq=uid))
        # === 修复结束 ===

        # 优先用 event.send 发送（若适配器支持，可拿到 message_id，便于后续“引用这条 PDF 消息”追问）。
        # 文件消息发送失败时不要再 yield chain_result(chain)，否则 respond.stage 会用同一个坏路径重试一次。
        sent_msg_id, send_error_msg = await self._send_pdf_file_component(event, Comp, chain, pdf_path, fname, label="PDF")
        if send_error_msg:
            yield event.plain_result(send_error_msg)
            return

        # 若成功拿到 message_id，则把本次 PDF 的 LaTeX 源码与该消息绑定，供用户引用追问
        if sent_msg_id and generated_tex_src:
            try:
                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    mp = st.setdefault("pdf_ctx_map", {})
                    if isinstance(mp, dict):
                        mp[str(sent_msg_id)] = generated_tex_src
                        # 控制缓存大小：最多保留 30 条（FIFO）
                        max_keep = 30
                        while len(mp) > max_keep:
                            try:
                                mp.pop(next(iter(mp)))
                            except Exception:
                                break
                    st["last_pdf_msg_id"] = str(sent_msg_id)
            except Exception:
                pass

        await self._daily_report_record_answer(
            event=event,
            kind="pdf",
            question=(final_text_input or ("[图片题目]" if final_image_inputs else "")),
            answer="PDF 解答",
            tex_src=generated_tex_src or "",
            pdf_path=pdf_path,
            image_urls=final_image_inputs,
        )

    # === 快捷指令 4: DeepThink 多角色迭代生成 PDF 解答（/spdf） ===
    @filter.command("spdf")
    async def cmd_spdf(self, event: AstrMessageEvent):
        if not self._cfg("enable_spdf_output", True):
            yield event.plain_result("SPDF 输出未启用")
            return

        raw = (getattr(event, "message_str", "") or "").strip()
        arg = ""
        m = re.match(r"^\s*/?spdf(?:\s+([\s\S]*))?$", raw, flags=re.I)
        if m:
            arg = (m.group(1) or "").strip()

        current_images = _get_event_images(event)
        skey = _get_session_key(event)

        async with self._state_lock:
            state = MATH_SESSION_STATE.get(skey, {}).copy()

        last_problem_text = (state.get("last_problem", "") or "").strip()
        last_images = state.get("last_image_urls", [])
        last_image_ts = float(state.get("last_image_ts", 0) or 0)
        reply_id = self._extract_reply_msg_id(event)
        reply_image_inputs: List[str] = []
        if reply_id:
            try:
                image_map = state.get("image_ctx_map", {})
                if isinstance(image_map, dict):
                    cached = image_map.get(str(reply_id)) or []
                    if isinstance(cached, list):
                        reply_image_inputs = [str(x) for x in cached if str(x or "").strip()]
                if (not reply_image_inputs) and str(reply_id) == str(state.get("last_image_msg_id", "") or "").strip():
                    cached = state.get("last_image_urls") or []
                    if isinstance(cached, list):
                        reply_image_inputs = [str(x) for x in cached if str(x or "").strip()]
            except Exception:
                reply_image_inputs = []

        final_text_input = ""
        final_image_inputs: List[str] = []

        now_ts = time.time()
        valid_sec = int(self._cfg("last_image_valid_sec", 3600) or 3600)
        is_last_img_likely_valid = (now_ts - last_image_ts) <= valid_sec

        if current_images:
            final_image_inputs = current_images
            final_text_input = arg
        elif reply_image_inputs:
            final_image_inputs = reply_image_inputs
            final_text_input = arg
        elif arg:
            final_text_input = arg

            # 文字 + 上一张图（追问场景）
            reuse_mode = str(self._cfg("pdf_reuse_last_image_with_text_mode", "smart") or "smart").lower().strip()
            followup_max_len = int(self._cfg("pdf_followup_text_max_len", 40) or 40)

            allow_reuse = False
            if last_images and is_last_img_likely_valid:
                last_had_img = bool(state.get("last_had_img", False))
                last_img_context = last_had_img or (last_problem_text == "[图片题目]")
                if last_img_context:
                    pending_mid = str(state.get("pending_image_msg_id") or state.get("last_image_msg_id") or "").strip()
                    if reply_id and ((not pending_mid) or (str(reply_id).strip() == pending_mid)):
                        allow_reuse = True
                    elif reuse_mode == "always":
                        allow_reuse = True
                    elif reuse_mode == "never":
                        allow_reuse = False
                    else:
                        t = (arg or "").strip()
                        if re.search(r"(不用|不使用|别|不要).{0,6}(图片|图|图像)", t):
                            allow_reuse = False
                        elif re.search(
                                r"(如图|见图|图中|图片|上题|上一题|刚才|继续|小问|第\s*(?:\d+|[一二两三四五六七八九十])\s*问|\(\s*\d+\s*\))",
                                t):
                            allow_reuse = True
                        elif len(t) <= max(10, followup_max_len) and (not _is_math_question(t)):
                            allow_reuse = True
                        else:
                            allow_reuse = False

            if allow_reuse:
                final_image_inputs = last_images
        elif last_images:
            if not is_last_img_likely_valid:
                yield event.plain_result("⚠️ 上一张图片可能已过期（链接可能失效），建议重新发送图片 + /spdf")
            final_image_inputs = last_images
            final_text_input = last_problem_text if last_problem_text != "[图片题目]" else ""
        elif last_problem_text and last_problem_text != "[图片题目]":
            final_text_input = last_problem_text
        else:
            yield event.plain_result("请发送 /spdf 题目文字，或发送包含题目的图片 + /spdf")
            return

        yield event.plain_result("正在生成中...\n你知道吗？" + _get_daily_math_fact())

        ref_pdf_ctx = ""
        if reply_id:
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    ref_pdf_ctx = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                ref_pdf_ctx = ""
            if not ref_pdf_ctx:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        if str(reply_id) == last_mid:
                            ref_pdf_ctx = last_pdf_ctx
                        else:
                            if self._reply_msg_has_pdf_hint(event):
                                ref_pdf_ctx = last_pdf_ctx
                    else:
                        ref_pdf_ctx = last_pdf_ctx

        try:
            event.set_extra("md2img_pdf_generation", True)
        except Exception:
            pass

        try:
            pdf_path, fname, generated_tex_src = await self._solve_math_to_spdf(
                problem_text=final_text_input,
                event=event,
                image_urls=final_image_inputs,
                ref_pdf_latex=ref_pdf_ctx,
            )

            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                now_ts = time.time()
                st["last_active_ts"] = now_ts
                if generated_tex_src:
                    st["last_pdf_context"] = generated_tex_src

                clean_last = (final_text_input or "").strip()
                if clean_last:
                    clean_last = re.sub(r"^\s*/?spdf\b", "", clean_last, flags=re.I).strip()
                if clean_last:
                    st["last_problem"] = clean_last
                else:
                    if final_image_inputs:
                        st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(final_image_inputs)

                if final_image_inputs:
                    st["last_image_urls"] = list(final_image_inputs)
                    st["last_image_ts"] = now_ts
        except Exception as e:
            err = str(e)
            # 先记录日志（若有编译日志则一并打印，便于排查）
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    log_snip = log_snip[-2000:]
                logger.error(f"SPDF 生成失败: {err}\nCompile log snippet:\n{log_snip}")
            except Exception:
                logger.error(f"SPDF 生成失败: {err}")

            low_err = (err or "").lower()

            # 1) 图片读取失败
            if "Upstream Error" in err:
                _sn = (event.get_sender_name() or "我").strip() or "我"
                yield event.plain_result(f"呜呜，{_sn} 够不到那张图片啦！它好像溜走了（链接过期或拒绝访问）。😣\n能不能麻烦你重新截图再发给我一次？")
                return

            # 2) 鉴权 / Key 缺失
            if _is_auth_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：模型鉴权不可用（未配置 API Key / Token，或 provider 暂不可用）。\n"
                    "排查建议：\n"
                    "1) 检查 AstrBot 对应模型 provider 是否已配置 Key/Token；\n"
                    "2) 如果你在 SPDF 的 Solver/Judge/质询等模型下拉框中选择了其他 provider，确保它们也配置了鉴权；\n"
                    "3) 可先把 spdf_judge_provider_id 留空，让 /spdf 跟随当前会话模型。"
                )
                return

            # 3) tool_calls / function_call 解析失败（常见于上游启用了 tools）
            if _is_toolcall_parse_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：上游模型返回了工具/函数调用格式（tool_calls/function_call），但当前 AstrBot 解析失败。\n"
                    "改进建议：\n"
                    "1) 保持插件配置 `spdf_disable_tools_during_generation=true`（默认开启），用于在 /pdf / /spdf 内部尽量禁用 tools；\n"
                    "2) 更换/下调为不容易触发 tool calling 的模型通道；\n"
                    "3) 升级 AstrBot 或对应 provider 适配（某些版本对 tool_calls 支持不完整）。"
                )
                return

            # 4) 上下文过长 / token 超限
            if _is_context_length_error(err):
                yield event.plain_result(
                    "❌ SPDF 生成失败：提示词/上下文过长（超出模型 token 限制）。\n"
                    "可以尝试：\n"
                    "1) 降低 spdf_candidate_char_budget 或 spdf_post_formatter_input_char_budget；\n"
                    "2) 关闭交叉质询/自一致投票或降低 spdf_num_solvers / spdf_iter_rounds；\n"
                    "3) 选择上下文更大的模型。"
                )
                return

            # 5) LaTeX 编译失败
            msg = "❌ SPDF 生成失败（本地 xelatex 编译出错）。"
            try:
                log_snip = (self._last_texlive_log or "").strip()
                if log_snip:
                    tail_lines = "\n".join(log_snip.splitlines()[-20:])
                    tail_lines = tail_lines[-1500:]
                    msg += "\n\nxelatex 日志末尾(截取)：\n" + tail_lines
            except Exception:
                pass

            # 如果明显不是编译错误，则给出更通用的提示
            if ("compile failed" not in low_err) and ("xelatex" not in low_err) and ("latex" not in low_err):
                msg = "❌ SPDF 生成失败：模型/网络调用异常。\n" + self._spdf_trunc(err, 400)

            yield event.plain_result(msg)
            return

        try:
            import astrbot.api.message_components as Comp
        except Exception:
            yield event.plain_result("当前 AstrBot 版本缺少文件消息组件，无法发送 PDF")
            return

        uid = event.get_sender_id()
        chain = []

        is_private = self._is_private_chat(event)
        if uid and (not is_private):
            chain.append(Comp.At(qq=uid))

        sent_msg_id: Optional[str] = None
        sent_msg_id, send_error_msg = await self._send_pdf_file_component(event, Comp, chain, pdf_path, fname, label="SPDF")
        if send_error_msg:
            yield event.plain_result(send_error_msg)
            return

        if sent_msg_id and generated_tex_src:
            try:
                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    mp = st.setdefault("pdf_ctx_map", {})
                    if isinstance(mp, dict):
                        mp[str(sent_msg_id)] = generated_tex_src
                        max_keep = 30
                        while len(mp) > max_keep:
                            try:
                                mp.pop(next(iter(mp)))
                            except Exception:
                                break
                    st["last_pdf_msg_id"] = str(sent_msg_id)
            except Exception:
                pass

        await self._daily_report_record_answer(
            event=event,
            kind="spdf",
            question=(final_text_input or ("[图片题目]" if final_image_inputs else "")),
            answer="SPDF DeepThink PDF 解答",
            tex_src=generated_tex_src or "",
            pdf_path=pdf_path,
            image_urls=final_image_inputs,
        )

    # ---------------------------------------------------------------------
    # 可选：路由模型辅助判断（只看文本）
    # ---------------------------------------------------------------------
    async def _router_classify(self, user_text: str) -> Optional[Dict[str, bool]]:
        if not self._cfg("use_router_model", False):
            return None
        router_provider_id = str(self._cfg("router_provider_id", "") or "").strip()
        if not router_provider_id:
            return None

        prompt = (
            "你是一个意图路由器 只输出JSON 不要输出多余文本\n"
            "请根据用户消息判断下面字段 字段必须都给出\n"
            "{\n"
            '  "is_math": boolean,\n'
            '  "wants_full_solution": boolean\n'
            "}\n"
            f"用户消息：{user_text}"
        )
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=router_provider_id,
                prompt=prompt,
            )
            text = (llm_resp.completion_text or "").strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            data = json.loads(m.group(0))
            return {
                "is_math": bool(data.get("is_math")),
                "wants_full_solution": bool(data.get("wants_full_solution")),
            }
        except Exception as e:
            logger.warning(f"router classify failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # 数学流程：默认注入完整图文讲义提示
    # ---------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_req(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._cfg("enable_math_coach", True):
            return

        # /pdf 命令内部也会触发 llm_generate。此时我们不应再额外注入“解题教练”系统提示，
        # 否则可能破坏 /pdf 专用 LaTeX 输出格式，进而导致 xelatex 编译失败。
        try:
            if event.get_extra("md2img_pdf_generation"):
                # /pdf / /spdf 内部调用：避免工具注入破坏 LaTeX 输出格式或触发 tool_calls 解析问题
                try:
                    if bool(self._cfg("spdf_disable_tools_during_generation", True)):
                        self._disable_llm_tools_in_req(req)
                except Exception:
                    pass
                return
        except Exception:
            pass

        user_text = (getattr(event, "message_str", "") or "").strip()
        plugin_kb_enabled = bool(self._cfg("enable_plugin_kb_integration", False))
        raw_kb_query = _is_kb_query(user_text)
        if raw_kb_query and not plugin_kb_enabled:
            return
        kb_query = raw_kb_query
        kb_n = _extract_kb_pick_count(user_text, int(self._cfg("kb_default_pick_count", 2) or 2))

        # 确保知识库工具处于激活状态（适配 kb_agentic_mode 等场景）
        if kb_query:
            try:
                self.context.activate_llm_tool("astr_kb_search")
            except Exception:
                pass
        current_images = _get_event_images(event)
        reply_id = self._extract_reply_msg_id(event)
        is_private = self._is_private_chat(event)

        skey = _get_session_key(event)
        now_ts = time.time()

        # 引用上一条“纯图片”消息：把那张图片带入本次 LLM 请求
        effective_images: List[str] = list(current_images or [])
        used_pending_image = False

        async with self._state_lock:
            state = MATH_SESSION_STATE.setdefault(skey, {})
            state["last_active_ts"] = now_ts

            # 当前消息带图：刷新 last_image_*
            if current_images:
                state["last_image_urls"] = list(current_images)
                state["last_image_ts"] = now_ts

            # 本条消息没有图，但“引用/回复”了某条已缓存图片消息 -> 精确使用该图片作为上下文
            if (not effective_images) and user_text and reply_id:
                try:
                    image_map = state.get("image_ctx_map", {})
                    if isinstance(image_map, dict):
                        cached = image_map.get(str(reply_id)) or []
                        if isinstance(cached, list) and cached:
                            effective_images = list(cached)
                            used_pending_image = True
                except Exception:
                    pass

            # 兼容旧状态：引用/回复上一条纯图片 pending 消息 -> 使用 last_image_urls
            if (not effective_images) and user_text and reply_id and state.get("pending_image_only"):
                pid = str(state.get("pending_image_msg_id") or "").strip()
                if (not pid) or (str(reply_id).strip() == pid):
                    cached = state.get("last_image_urls") or []
                    if isinstance(cached, list) and cached:
                        effective_images = list(cached)
                        used_pending_image = True

            if used_pending_image:
                # 用过就清掉 pending，避免后续串图；image_ctx_map 保留以支持再次精确引用。
                state.pop("pending_image_only", None)
                state.pop("pending_image_msg_id", None)
                state.pop("pending_image_ts", None)

            # 取快照供后续逻辑使用
            state_snapshot = dict(state)

        state = state_snapshot
        if used_pending_image:
            self._attach_images_to_req(req, effective_images, replace_existing=True)

        # 后续逻辑统一使用 current_images（可能来自 pending）
        current_images = effective_images


        # ---------------- 对话记忆：保存问答并在本次请求前注入相似历史 ----------------
        try:
            # 图片题最怕“历史题目粘连”。当前消息带图或引用图片时，不注入旧问答片段。
            if bool(self._cfg("enable_chat_memory", True)) and (not current_images):
                store_cmd = bool(self._cfg("chat_memory_store_commands", False))
                if user_text and (store_cmd or (not re.match(r"^\s*/\w+", user_text))):
                    # 标记本次请求，供 on_llm_resp 把“用户-助手”配对写入 history
                    event.set_extra("_chatmem_pending", {
                        "skey": skey,
                        "ts": now_ts,
                        "user": user_text,
                        "user_images": list(current_images) if current_images else [],
                    })

                # 构造检索片段并注入 system_prompt（不影响用户原 prompt）
                recent_n = int(self._cfg("chat_memory_recent_turns", 6) or 6)
                top_k = int(self._cfg("chat_memory_retrieve_k", 6) or 6)
                min_score = float(self._cfg("chat_memory_min_score", 0.12) or 0.12)
                char_budget = int(self._cfg("chat_memory_char_budget", 3500) or 3500)
                half_life = float(self._cfg("chat_memory_recency_half_life_sec", 21600) or 21600)

                async with self._state_lock:
                    st = MATH_SESSION_STATE.setdefault(skey, {})
                    hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                    snips = _select_memory_snippets(hist, user_text or "", now_ts,
                                                   recent_n=recent_n, top_k=top_k,
                                                   min_score=min_score, half_life_sec=half_life)
                mem_ctx = _format_memory_context(snips, char_budget)
                if mem_ctx:
                    req.system_prompt = (req.system_prompt or "") + mem_ctx
        except Exception as e:
            logger.warning(f"chat memory inject failed: {e}")

        treat_img_as_math = bool(self._cfg("treat_image_as_math", True))
        last_problem = state.get("last_problem", "")
        last_had_img = bool(state.get("last_had_img", False))

        # 注入 PDF 上下文（仅在用户“引用/回复”了机器人发出的 PDF 消息时才注入，避免影响下一题）
        # reply_id 已在前面提取
        pdf_ctx_to_inject = ""
        if reply_id:
            # 1) 优先：按“被引用消息 id -> LaTeX 源码”精确匹配
            try:
                mp = state.get("pdf_ctx_map", {})
                if isinstance(mp, dict):
                    pdf_ctx_to_inject = (mp.get(str(reply_id)) or "").strip()
            except Exception:
                pdf_ctx_to_inject = ""

            # 2) 兜底：如果无法获取 bot 消息 id（或历史版本没存），则在“存在引用”时弱绑定到最近一次 PDF
            if not pdf_ctx_to_inject:
                last_pdf_ctx = (state.get("last_pdf_context") or "").strip()
                if last_pdf_ctx:
                    last_mid = str(state.get("last_pdf_msg_id", "") or "").strip()
                    if last_mid:
                        # 只有引用 id 与最后一次 PDF 消息一致时才注入（避免串题）
                        if str(reply_id) == last_mid:
                            pdf_ctx_to_inject = last_pdf_ctx
                        else:
                            # 某些适配器能拿到被引用消息内容：若确实包含 pdf，则允许兜底
                            if self._reply_msg_has_pdf_hint(event):
                                pdf_ctx_to_inject = last_pdf_ctx
                    else:
                        # 没有任何 message_id 绑定信息：只能退化为“只要用户引用，就使用最近一次 PDF”
                        pdf_ctx_to_inject = last_pdf_ctx

        if pdf_ctx_to_inject and user_text:
            req.system_prompt = (req.system_prompt or "") + (
                "\n[历史信息] 用户正在引用你之前发出的 PDF 解答消息。以下是该 PDF 的 LaTeX 源码（用于回答用户对 PDF 的追问）。\n"
                "仅在回答与该 PDF 相关的问题时使用这些信息；若用户问的是新问题，不要被历史信息干扰。\n"
                f"[START LATEX CONTEXT]\n{pdf_ctx_to_inject}\n[END LATEX CONTEXT]\n"
            )

        router = await self._router_classify(user_text) if user_text else None

        heuristic_is_math = _is_math_question(user_text)
        explicit_hint = any(k in user_text for k in _HINT_PREF)
        wants_full = _wants_full_solution(user_text) or (router.get("wants_full_solution") if router else False)

        # 知识库检索意图：强制 full（可通过配置关闭）
        if kb_query and bool(self._cfg("force_full_on_kb_query", True)):
            wants_full = True

        is_math = heuristic_is_math or (router.get("is_math") if router else False)
        if kb_query:
            # 让知识库检索问法也走“数学答疑”管线（输出 <md> 图片）
            is_math = True
        if bool(current_images) and treat_img_as_math:
            is_math = True

        if is_math and explicit_hint and not wants_full:
            return
        if is_math:
            wants_full = True

        is_followup_full = (not is_math) and wants_full and (last_problem or last_had_img)
        if not (is_math or is_followup_full):
            return

        # 记录“上一题”
        if is_math:
            async with self._state_lock:
                st = MATH_SESSION_STATE.setdefault(skey, {})
                st["last_active_ts"] = time.time()
                if user_text:
                    st["last_problem"] = user_text
                else:
                    st["last_problem"] = "[图片题目]"
                st["last_had_img"] = bool(current_images)

        mode = "full"
        event.set_extra("math_flow",
                        {"mode": mode, "has_image": bool(current_images), "kb_query": kb_query, "kb_n": kb_n})

        if mode == "full" and is_followup_full:
            req.prompt = (
                "用户要求你给出上一题的完整解答过程\n"
                "请结合上下文中上一题（可能是图片题）来解题\n"
                f"用户补充：{user_text}\n"
            )

        persona = str(self._cfg("math_persona", DEFAULT_CFG["math_persona"]) or "").strip()
        req.system_prompt = (req.system_prompt or "") + self._build_math_system_prompt(
            mode=mode,
            persona=persona,
            has_image=bool(current_images),
            kb_query=kb_query,
            kb_n=kb_n,
        )

    def _build_math_system_prompt(self, mode: str, persona: str, has_image: bool, kb_query: bool = False,
                                  kb_n: int = 2) -> str:
        persona = persona.strip() or DEFAULT_CFG["math_persona"]
        img_hint = "用户发来的是图片题 请先读题 再给出完整解答" if has_image else ""
        visual_rules = ""
        if bool(self._cfg("full_solution_prefer_svg_diagram", True)):
            visual_rules = (
                "图文讲义要求：当题目涉及几何图形、函数图像、积分区域、向量、空间立体、坐标变换或结构关系时，"
                "请在相关小节直接写一个内联 SVG 示意图，例如 `<svg width=\"720\" height=\"320\" viewBox=\"0 0 720 320\">...</svg>`。\n"
                "SVG 只画辅助理解所需的坐标轴、边界、阴影区域、箭头、标签和关键点；不要声称示意图精确到未给出的数值。\n"
                "不要输出 Python 绘图代码，不要输出外链图片，不要把 SVG 放进代码块；SVG 会和 Markdown/LaTeX 一起被渲染成图片。\n"
            )

        if mode == "full":
            if kb_query:
                # 知识库检索/题库挑题：不要输出提示流，直接给"完整回答格式"
                return (
                    "\n[数学答疑/知识库]\n"
                    f"你的人格设定：{persona}\n"
                    + (img_hint + "\n" if img_hint else "")
                    + "用户希望你从知识库/题库中检索相关题目，并且给出具体出处\n"
                    "你必须用 <md>...</md> 包裹整个回答内容\n"
                    "在 <md> 内先输出：\n"
                    "## 知识库检索结果\n"
                    f"- 你要给出最相关的 {max(1, min(int(kb_n or 2), 10))} 条结果（若系统检索不到，明确说明未命中）\n"
                    "- 每条结果必须包含：题目内容（可用 LaTeX）、出处（文档/条目标题/章节/页码/ID 等）\n"
                    "- 如果需要解答某道题，解答部分开头必须写浅蓝题目卡片：`> [problem-card]` 下一行 `> **题目/证明目标：** ...`\n"
                    "- 重要结论、定理、提示、推论请优先用 Markdown 引用块 `>` 表示，便于渲染成讲义卡片\n"
                    f"{visual_rules}"
                    "- 不要只输出泛泛的引导或学习建议，不要输出 <stream>\n"
                    "如果用户同时给了要解的具体题目或要求解答，请在检索结果后追加：\n"
                    "## 解答\n"
                    "并给出该题完整推导与最终答案（可用 LaTeX 例如 $$...$$）\n"
                )

            return (
                "\n[数学答疑]\n"
                f"你的人格设定：{persona}\n"
                + (img_hint + "\n" if img_hint else "")
                + "现在用户明确要求完整解答\n"
                "你必须给出完整的解题过程 并且必须用 <md>...</md> 包裹整个解答内容\n"
                "<md> 内允许使用 Markdown 与 LaTeX 例如 $$...$$\n"
                "排版请使用数学讲义结构：第一行必须用 # 写总标题，后续必须用 ## 写“一、核心思想”“二、解题过程”“三、总结”等分节标题\n"
                "在任何证明/解答正文前，必须先输出浅蓝题目卡片，格式严格如下：\n"
                "> [problem-card]\n"
                "> **题目/证明目标：** 这里简要写明题目、已知条件或要证明的命题\n"
                "然后再输出 `## 一、证明过程` 或 `## 一、解答过程`。\n"
                "重要定理、关键公式、易错提醒、结论推广请使用 Markdown 引用块 `>`，例如 `> **定理：** ...`；除必要的内联 SVG 示意图外，不要直接写 HTML\n"
                f"{visual_rules}"
                "不要只输出 `### 证明` 或纯段落；不要省略题目/证明目标卡片\n"
                "步骤要清晰 推导要完整 最后给出明确答案\n"
            )

        return (
                "\n[数学答疑]\n"
                f"你的人格设定：{persona}\n"
                "用户没有要求完整过程 你要像真人答疑一样先引导\n"
                + (img_hint + "\n" if img_hint else "")
                + "输出格式必须严格如下 只输出这一段 不要输出其它内容\n"
                  "<stream>\n"
                  "</stream>\n"
                  "规则\n"
                  "根据题目难度和内容 输出2到6条短消息 每条占一行\n"
                  "简单题少说几句(2-3条) 复杂题可以多引导几步(4-6条) 根据需要自行判断\n"
                  "每条消息只写1-2句话\n"
                  "每条消息句末不要出现句号\n"
                  "不要用 Markdown 不要用 LaTeX 不要用代码块\n"
                  "不要直接给最终答案或完整推导\n"
                  "如果图片或题面不清晰 先让用户补充关键条件或把题抄成文字\n"
        )

    # ---------- 可选：用二次模型把输出改写成 <stream> ----------
    async def _rewrite_to_stream(self, raw: str, has_image: bool) -> Optional[List[str]]:
        if not self._cfg("use_hint_rewriter_model", False):
            return None
        pid = str(self._cfg("hint_rewriter_provider_id", "") or "").strip()
        if not pid:
            return None
        prompt = (
                "把下面的回答改写成真人答疑式提示流\n"
                "根据内容复杂度输出2到6行 每行1-2句话 句末不要加句号\n"
                "不要用Markdown 不要用LaTeX 不要给最终答案\n"
                "只输出 <stream>...</stream> 这一段\n"
                + ("如果图片题看不清 先让用户补充关键条件\n" if has_image else "")
                + "原始文本如下：\n"
                + raw
        )
        try:
            llm_resp = await self.context.llm_generate(chat_provider_id=pid, prompt=prompt)
            lines = _extract_stream_lines_only(llm_resp.completion_text or "")
            if lines:
                return _normalize_stream_msgs(lines)
        except Exception as e:
            logger.warning(f"hint rewrite failed: {e}")
        return None

    # ---------- LLM响应：full 强制 <md>；hint 预计算提示流 ----------
    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        raw = resp.completion_text or ""
        event.set_extra("raw_llm", raw)

        mf = event.get_extra("math_flow")
        if not isinstance(mf, dict):
            # 非数学流程：也写入对话记忆（普通聊天）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(resp.completion_text or ""),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(normal) failed: {e}")
            return

        mode = mf.get("mode")
        has_img = bool(mf.get("has_image"))

        if mode == "full":
            md_blocks = re.findall(r"<md>([\s\S]*?)</md>", raw)
            if md_blocks:
                joined = "\n\n".join([b.strip() for b in md_blocks if b.strip()]).strip()
                resp.completion_text = f"<md>\n{joined or raw.strip()}\n</md>"
            else:
                resp.completion_text = f"<md>\n{raw.strip()}\n</md>"
            event.set_extra("raw_llm", resp.completion_text)
            # 写入对话记忆（full）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(resp.completion_text or ""),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(full) failed: {e}")
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict):
                    await self._daily_report_record_answer(
                        event=event,
                        kind="full",
                        question=str(pending.get("user") or ""),
                        answer=resp.completion_text or raw,
                        tex_src="",
                        pdf_path="",
                        image_urls=list(pending.get("user_images") or []),
                    )
            except Exception as e:
                logger.warning(f"daily report store(full) failed: {e}")
            return

        if mode == "hint":
            lines = _extract_stream_lines_only(raw)

            if (not lines) or _looks_like_markdown_or_full(raw):
                rewritten = await self._rewrite_to_stream(raw, has_img)
                if rewritten:
                    msgs = rewritten
                else:
                    msgs = _normalize_stream_msgs([])
            else:
                msgs = _normalize_stream_msgs(lines)

            event.set_extra("math_stream_msgs", msgs)
            # 写入对话记忆（hint：把提示流拼成一段存）
            try:
                pending = event.get_extra("_chatmem_pending")
                if isinstance(pending, dict) and bool(self._cfg("enable_chat_memory", True)):
                    skey = str(pending.get("skey") or "")
                    assistant_text = "\n".join([str(x) for x in msgs if str(x).strip()])
                    async with self._state_lock:
                        st = MATH_SESSION_STATE.setdefault(skey, {})
                        hist = _ensure_chat_history(st, int(self._cfg("chat_memory_max_turns", 120) or 120))
                        hist.append({
                            "ts": float(pending.get("ts", time.time()) or time.time()),
                            "user": str(pending.get("user") or ""),
                            "assistant": _strip_tags_for_memory(assistant_text),
                            "user_images": list(pending.get("user_images") or []),
                        })
            except Exception as e:
                logger.warning(f"chat memory store(hint) failed: {e}")

    # ---------------- 装饰结果：hint 逐条消息；否则 Markdown->图片 ----------------
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        mf = event.get_extra("math_flow")
        if isinstance(mf, dict) and mf.get("mode") == "hint":
            msgs = event.get_extra("math_stream_msgs")
            if not isinstance(msgs, list) or not msgs:
                raw_llm = event.get_extra("raw_llm") or ""
                lines = _extract_stream_lines_only(raw_llm)
                msgs = _normalize_stream_msgs(lines)

            delay_ms = int(self._cfg("hint_send_delay_ms", 0) or 0)
            for m in msgs:
                m = _strip_trailing_periods(_sanitize_hint_text(str(m)))
                if not m:
                    continue
                try:
                    await event.send(MessageChain().message(m))
                except Exception as e:
                    logger.error(f"send stream msg failed: {e}")
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)

            # 阻止原始输出继续发送
            try:
                result = event.get_result()
                result.chain = []
            except Exception:
                pass

            if bool(self._cfg("stop_event_on_hint", True)):
                try:
                    event.stop_event()
                except Exception:
                    pass
            return

        # 非 hint：保持原插件行为（Markdown -> 图片）
        result = event.get_result()
        new_chain: List[Any] = []
        for item in result.chain:
            if isinstance(item, Plain):
                components = await self._process_text_with_markdown(item.text, event)
                new_chain.extend(components)
            else:
                new_chain.append(item)
        result.chain = new_chain
