# -*- coding: utf-8 -*-
try:
    from .shared import *
except ImportError:
    from shared import *


class RenderMixin:
    """Markdown, MathJax, PagedJS, and screenshot rendering helpers."""

    async def _process_text_with_markdown(self, text: str, event: AstrMessageEvent = None) -> List[Any]:
        if "<md>" not in text:
            is_complex = re.search(r"(```|\$\$|\|.*?\|.*?\||\\begin\{)", text, re.DOTALL)
            if is_complex:
                text = f"<md>\n{text}\n</md>"

        user_mode = "mobile"
        if event:
            user_mode = get_user_mode(event)

        components: List[Any] = []
        parts = re.split(r"(<md>.*?</md>)", text, flags=re.DOTALL)

        for part in parts:
            if part.startswith("<md>") and part.endswith("</md>"):
                md_content = part[4:-5].strip()
                if not md_content:
                    continue

                output_path = os.path.join(self.IMAGE_CACHE_DIR, f"{uuid.uuid4()}.png")

                try:
                    img_paths = await self._markdown_to_image_playwright(
                        md_text=md_content,
                        output_image_path=output_path,
                        scale=2,
                        mode=user_mode,
                    )

                    if img_paths:
                        for p in img_paths:
                            components.append(Image.fromFileSystem(p))
                    else:
                        components.append(Plain(f"--- 渲染失败 ---\n{md_content}"))
                except Exception as e:
                    logger.error(f"渲染异常: {e}")
                    components.append(Plain(f"--- 渲染错误 ---\n{md_content}"))
            else:
                if part:
                    components.append(Plain(part))

        return components

    async def _markdown_to_image_playwright(
            self,
            md_text: str,
            output_image_path: str,
            scale: int = 2,
            mode: str = "mobile",
    ) -> List[str]:
        async with self._render_sema:
            # 尽量复用 browser；若无法复用则本次临时启动
            browser = self._browser
            pw = self._pw

            if bool(self._cfg("reuse_playwright_browser", True)) and browser is None:
                await self._ensure_browser()
                browser = self._browser
                pw = self._pw

            return await self._markdown_to_image_playwright_impl(
                md_text=md_text,
                output_image_path=output_image_path,
                scale=scale,
                mode=mode,
                shared_browser=browser,
                shared_pw=pw,
            )

    async def _markdown_to_image_playwright_impl(
            self,
            md_text: str,
            output_image_path: str,
            scale: int = 2,
            mode: str = "mobile",
            shared_browser=None,
            shared_pw=None,
    ) -> List[str]:
        # --- 1. 根据模式设定参数 ---
        if mode == "pc":
            target_width = 794
            padding_val = 46
            font_size = "17px"
            is_mobile_css = False
        else:
            target_width = 760
            padding_val = 40
            font_size = "17px"
            is_mobile_css = True

        # --- 2. CSS 样式：数学讲义风（米白纸张 + 棕色标题 + 讲义卡片） ---
        base_css = f"""
            * {{ box-sizing: border-box; }}
            :root {{
                --paper: #ffffff;
                --paper-soft: #f7f8fa;
                --ink: #1f2933;
                --muted: #5c6470;
                --brown: #6f3f12;
                --brown-deep: #4c2a0b;
                --gold: #c8942d;
                --gold-soft: #fff4cf;
                --green: #5f8d43;
                --green-soft: #eef6e8;
                --blue: #4d82bb;
                --blue-soft: #eef6ff;
                --line: #e2c27a;
            }}
            body {{
                margin: 0; padding: 0;
                background: #ffffff;
                font-family: "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", "SimHei", sans-serif;
            }}
            .content-wrapper {{
                background-color: var(--paper);
                color: var(--ink);
                font-size: {font_size};
                line-height: 1.78;

                width: {target_width}px;
                padding: {padding_val}px;

                overflow-wrap: break-word;
                word-wrap: break-word;
            }}

            img, svg, video {{ max-width: 100% !important; height: auto; }}
            svg {{
                display: block;
                margin: 1em auto;
                background: rgba(255,255,255,0.55);
                border-radius: 7px;
            }}
            svg text {{
                font-family: "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", "SimHei", sans-serif;
            }}
            h1, h2, h3, h4 {{
                color: var(--brown-deep);
                font-family: "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", "SimHei", sans-serif;
                font-weight: 800;
                letter-spacing: 0.02em;
            }}
            h1 {{
                text-align: center;
                font-size: 1.62em;
                line-height: 1.25;
                margin: 0 0 1.35em;
                padding: 0.7em 0 0.65em;
                border-bottom: 2px solid #8a5a14;
            }}
            h2 {{
                font-size: 1.22em;
                margin: 1.55em 0 0.72em;
                padding-left: 0.62em;
                border-left: 4px solid #936310;
            }}
            h3 {{
                font-size: 1.08em;
                margin: 1.15em 0 0.45em;
            }}
            p {{
                margin: 0.58em 0;
                text-align: justify;
            }}
            strong {{
                color: #1f2933;
                font-weight: 800;
            }}
            ul, ol {{
                padding-left: 1.45em;
                margin: 0.56em 0 0.8em;
            }}
            li {{
                margin: 0.25em 0;
            }}
            hr {{
                border: 0;
                border-top: 2px solid #8a5a14;
                margin: 1.2em 0;
            }}
            blockquote {{
                margin: 1.05em 0;
                padding: 0.86em 1.05em;
                border: 1px solid var(--gold);
                border-radius: 7px;
                background: linear-gradient(180deg, var(--gold-soft), #fff9e8);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.65);
                page-break-inside: avoid;
            }}
            blockquote:nth-of-type(3n + 2) {{
                background: var(--green-soft);
                border-color: var(--green);
                border-left: 4px solid var(--green);
            }}
            blockquote:nth-of-type(3n) {{
                background: var(--blue-soft);
                border-color: var(--blue);
            }}
            blockquote:has(> p:first-child [data-card="problem"]) {{
                background: linear-gradient(180deg, #eef6ff, #f7fbff);
                border-color: var(--blue);
                text-align: left;
                text-align-last: auto;
                word-spacing: normal;
            }}
            blockquote [data-card="problem"] {{
                display: none;
            }}
            blockquote p {{
                margin: 0.3em 0;
                text-align: left;
                text-align-last: auto;
                word-spacing: normal;
            }}
            pre {{
                background-color: #f5eedb;
                color: #2f2b24;
                border: 1px solid #e2c27a;
                border-radius: 7px;
                padding: 12px 14px;
                white-space: pre-wrap;
                word-break: break-all;
                max-width: 100%;
                font-size: 85%;
                margin: 12px 0;
                page-break-inside: avoid;
            }}
            code {{
                font-family: "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
                background: #f4ead1;
                border: 1px solid #ead49b;
                border-radius: 4px;
                padding: 0.08em 0.28em;
                font-size: 0.88em;
            }}
            pre code {{
                background: transparent;
                border: 0;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
                margin: 1em 0;
                page-break-inside: avoid;
                background: rgba(255,255,255,0.45);
            }}
            th, td {{
                border: 1px solid #d5b56d;
                padding: 7px 8px;
                word-wrap: break-word;
            }}
            th {{
                background: #f4ead1;
                color: var(--brown-deep);
                font-weight: 800;
            }}
            .MathJax_Display, .MathJax {{
                max-width: 100% !important;
                overflow-x: hidden; overflow-y: hidden;
            }}
            .MathJax_Display {{
                margin: 0.95em 0 !important;
            }}

            /* Paged.js 样式 */
            .pagedjs_pages .content-wrapper {{ width: auto !important; padding: 0 !important; }}
            .pagedjs_page {{ background-color: var(--paper) !important; box-shadow: none !important; overflow: hidden; }}
            .pagedjs_margin-top, .pagedjs_margin-bottom {{ display: none; }}
        """

        # --- 4. HTML 模板 ---
        mathjax_local_js = _plugin_resource_path("vendor", "md2img", "mathjax-2.7.7", "MathJax.js")
        if os.path.exists(mathjax_local_js):
            mathjax_script_src = _file_url(mathjax_local_js, "config=TeX-MML-AM_CHTML")
        else:
            mathjax_script_src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"
            logger.warning("[md2img] local MathJax not found, fallback to CDN")

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{css_content}</style>
            <script type="text/x-mathjax-config">
                MathJax.Hub.Config({{
                    tex2jax: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true }},
                    "HTML-CSS": {{ scale: 90, linebreaks: {{ automatic: true, width: "container" }} }},
                    SVG: {{ linebreaks: {{ automatic: true, width: "container" }} }}
                }});
            </script>
            <script type="text/javascript"
              src="{mathjax_script_src}"></script>
        </head>
        <body>
            <div class="content-wrapper" id="main-content">{content}</div>
        </body>
        </html>
        """

        # --- 5. Markdown -> HTML ---
        md_text = _prepare_markdown_for_render(md_text)
        protected_md, math_pieces = _protect_math_for_markdown(md_text)

        markdown_parser = mistune.create_markdown(
            escape=False, plugins=["table", "url", "strikethrough", "task_lists"]
        )
        html_content = markdown_parser(protected_md)
        html_content = _restore_math_tokens(html_content, math_pieces)
        html_content = html_content.replace("[problem-card]", '<span data-card="problem"></span>')

        full_html = html_template.format(css_content=base_css, content=html_content, mathjax_script_src=mathjax_script_src)

        generated_images: List[str] = []

        # browser 复用：若传入 shared_browser，则只创建 context/page；否则本次启动
        if shared_browser is not None:
            browser = shared_browser
            own_pw_cm = None
        else:
            own_pw_cm = async_playwright()

        if own_pw_cm is not None:
            async with own_pw_cm as p:
                browser = await p.chromium.launch()
                generated_images = await self._render_with_browser(browser, full_html, output_image_path, scale, mode,
                                                                   is_mobile_css)
                try:
                    await browser.close()
                except Exception:
                    pass
            return generated_images

        # shared browser path
        return await self._render_with_browser(browser, full_html, output_image_path, scale, mode, is_mobile_css)

    async def _render_with_browser(self, browser, full_html: str, output_image_path: str, scale: int, mode: str,
                                   is_mobile_css: bool) -> List[str]:
        generated_images: List[str] = []

        vp_width = (794 if mode == "pc" else 760) + 50
        context = await browser.new_context(
            device_scale_factor=scale, viewport={"width": vp_width, "height": 2000}
        )
        page = await context.new_page()
        render_html_path = ""

        try:
            render_timeout_sec = int(self._cfg("playwright_render_timeout_sec", 60) or 60)
        except Exception:
            render_timeout_sec = 60
        render_timeout_ms = max(1, render_timeout_sec) * 1000
        context.set_default_timeout(render_timeout_ms)
        context.set_default_navigation_timeout(render_timeout_ms)
        page.set_default_timeout(render_timeout_ms)
        page.set_default_navigation_timeout(render_timeout_ms)

        wait_until = str(self._cfg("playwright_wait_until", "networkidle") or "networkidle").lower().strip()
        valid_wait_until = {"commit", "domcontentloaded", "load", "networkidle"}
        if wait_until not in valid_wait_until:
            logger.warning(f"[md2img] invalid playwright_wait_until={wait_until!r}, fallback to networkidle")
            wait_until = "networkidle"
        render_html_path = os.path.splitext(output_image_path)[0] + ".html"
        os.makedirs(os.path.dirname(render_html_path), exist_ok=True)
        with open(render_html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        render_html_url = _file_url(render_html_path)
        try:
            await page.goto(render_html_url, wait_until=wait_until)
        except Exception:
            # 回退策略：networkidle 卡住时，使用 domcontentloaded
            await page.goto(render_html_url, wait_until="domcontentloaded")

        async def _typeset_mathjax(stage: str = "") -> None:
            try:
                await page.wait_for_function(
                    "() => !!(window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Queue)",
                    timeout=15000,
                )
                await page.evaluate("""
                    () => new Promise((resolve) => {
                        const hub = window.MathJax && window.MathJax.Hub;
                        if (!hub || typeof hub.Queue !== "function") {
                            resolve(false);
                            return;
                        }
                        hub.Queue.call(hub, ["Typeset", hub], () => resolve(true));
                    })
                """)
            except Exception as e:
                suffix = f"({stage})" if stage else ""
                logger.warning(f"MathJax typeset skipped{suffix}: {e}")

        async def _fallback_content_screenshot(reason: str = "") -> None:
            if generated_images:
                return
            for selector in (".content-wrapper", "#main-content", ".pagedjs_pages", "body"):
                element = await page.query_selector(selector)
                if element:
                    logger.warning(
                        f"Render fallback: {selector} screenshot{(': ' + reason) if reason else ''}"
                    )
                    await element.screenshot(path=output_image_path)
                    generated_images.append(output_image_path)
                    return
            logger.error("Render Error: no screenshot target found")

        async def _full_page_screenshot(reason: str = "") -> None:
            if generated_images:
                return
            try:
                logger.warning(f"Render fallback: full page screenshot{(': ' + reason) if reason else ''}")
                await page.screenshot(path=output_image_path, full_page=True)
                generated_images.append(output_image_path)
            except Exception as e:
                logger.error(f"Render fallback full page screenshot failed: {e}")

        async def _restore_original_content_screenshot(reason: str = "") -> None:
            if generated_images:
                return
            try:
                logger.warning(f"Render fallback: restore original content screenshot{(': ' + reason) if reason else ''}")
                await page.goto(render_html_url, wait_until="domcontentloaded")
                await _typeset_mathjax("restored fallback")
                await page.wait_for_timeout(1000)
                element = await page.query_selector(".content-wrapper")
                if element:
                    await element.screenshot(path=output_image_path)
                    generated_images.append(output_image_path)
                    return
                logger.error("Render fallback: restored content-wrapper not found")
            except Exception as e:
                logger.error(f"Render fallback restore original content failed: {e}")
            await _full_page_screenshot(reason or "restore original content failed")

        # 等待 MathJax 加载并完成排版
        await _typeset_mathjax()
        await page.wait_for_timeout(5000)
        # === Mobile: 长截屏 ===
        if is_mobile_css:
            element = await page.query_selector(".content-wrapper")
            if element:
                await element.screenshot(path=output_image_path)
                generated_images.append(output_image_path)
            else:
                await _fallback_content_screenshot("content-wrapper not found")
        else:
            # === PC: 智能分页检测 ===
            try:
                content_height = await page.evaluate("document.getElementById('main-content').scrollHeight")
            except Exception:
                content_height = 0

            A4_HEIGHT = 1123

            if content_height <= A4_HEIGHT:
                element = await page.query_selector(".content-wrapper")
                if element:
                    await element.screenshot(path=output_image_path)
                    generated_images.append(output_image_path)
                else:
                    await _fallback_content_screenshot("content-wrapper not found")
            else:
                paged_style = """
                    @page { size: 794px 1123px; margin: 45px; }
                    .pagedjs_page { box-sizing: border-box; }
                """
                await page.add_style_tag(content=paged_style)

                paged_local_candidates = [
                    _plugin_resource_path("vendor", "md2img", "pagedjs", "paged.polyfill.js"),
                    _plugin_resource_path("paged.polyfill.js"),
                ]
                local_js = next((p for p in paged_local_candidates if os.path.exists(p)), "")
                if local_js:
                    await page.add_script_tag(path=local_js)
                else:
                    logger.warning("[md2img] local PagedJS not found, fallback to CDN")
                    await page.add_script_tag(url="https://unpkg.com/pagedjs/dist/paged.polyfill.js")

                try:
                    pagedjs_wait_timeout_sec = int(self._cfg("pagedjs_wait_timeout_sec", 30) or 30)
                except Exception:
                    pagedjs_wait_timeout_sec = 30
                pagedjs_wait_timeout_ms = max(1, pagedjs_wait_timeout_sec) * 1000
                pagedjs_ready = False
                try:
                    await page.wait_for_selector(".pagedjs_page", timeout=pagedjs_wait_timeout_ms)
                    await page.wait_for_timeout(5000)
                    pagedjs_ready = True
                except Exception as e:
                    logger.error(f"PagedJS Timeout: {e}")

                if pagedjs_ready:
                    await _typeset_mathjax("after pagedjs")

                    pages = await page.query_selector_all(".pagedjs_page")
                    base_path, ext = os.path.splitext(output_image_path)
                    pagedjs_screenshot_failed = False
                    written_page_images: List[str] = []
                    for i, page_elem in enumerate(pages):
                        curr_path = output_image_path if i == 0 else f"{base_path}_{i + 1}{ext}"
                        try:
                            await page_elem.screenshot(path=curr_path)
                            generated_images.append(curr_path)
                            written_page_images.append(curr_path)
                        except Exception as e:
                            logger.error(f"PagedJS page screenshot failed: {e}")
                            pagedjs_screenshot_failed = True
                            break
                    if pagedjs_screenshot_failed:
                        for img_path in written_page_images:
                            try:
                                os.remove(img_path)
                            except OSError:
                                pass
                        generated_images.clear()
                        await _restore_original_content_screenshot("PagedJS page screenshot failed")
                    elif not generated_images:
                        await _restore_original_content_screenshot("PagedJS did not create pages")
                else:
                    await _restore_original_content_screenshot("PagedJS did not create pages")

        try:
            await page.close()
            await context.close()
        except Exception:
            pass
        if render_html_path:
            try:
                os.remove(render_html_path)
            except OSError:
                pass

        return generated_images
