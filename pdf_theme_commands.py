"""Command implementation without provider calls or theme-dependent rendering."""
import asyncio
from pathlib import Path
import re

try:
    from .pdf_theme_preferences import ThemePreferences, user_key, parse_choice, choice_label, help_text
except ImportError:
    from pdf_theme_preferences import ThemePreferences, user_key, parse_choice, choice_label, help_text


class PdfThemeMixin:
    def _pdf_theme_store(self):
        return ThemePreferences(self.DATA_DIR)

    async def _pdf_theme_choice(self, event):
        return await asyncio.to_thread(self._pdf_theme_store().get, user_key(event))

    async def _pdf_theme_command(self, event, arg=""):
        # Do not let the generic text-to-image or coaching hooks transform UI replies.
        event.set_extra("md2img_pdf_theme_command", True)
        event.should_call_llm(False)
        key = user_key(event)
        if not key:
            yield event.plain_result("无法识别你的用户账号，未修改任何 PDF 主题设置。")
            return
        raw = str(arg or "").strip()
        if not raw:
            text = str(getattr(event, "message_str", "") or "").strip()
            match = re.fullmatch(r"/?pdf主题(?:\s+([\s\S]*))?", text, flags=re.I)
            raw = (match.group(1) or "").strip() if match else ""
        try:
            if not raw:
                choice = await self._pdf_theme_choice(event)
                reply = help_text(choice)
            else:
                choice = parse_choice(raw)
                if choice is None:
                    reply = "主题序号应为 1～9，或发送「随机」「默认」。设置未改变。\n发送 /pdf主题概览 查看全部主题。"
                else:
                    await asyncio.to_thread(self._pdf_theme_store().set, key, choice)
                    reply = (
                        f"✅ 你的 PDF 主题已设为：{choice_label(choice)}。\n"
                        "仅对本人后续的 /pdf、/spdf 和对应 Agent 工具生效；"
                        "跨群/私聊保存，正在生成的任务不变。\n"
                        "发送 /pdf主题概览 查看全部主题。"
                    )
        except Exception:
            # Never acknowledge a successful save on a failed/locked database.
            reply = "PDF 主题设置读取或保存失败，未确认修改成功，请稍后重试。"
        yield event.plain_result(reply)

    async def _pdf_theme_overview_command(self, event):
        event.set_extra("md2img_pdf_theme_command", True)
        event.should_call_llm(False)
        path = Path(__file__).resolve().parent / "assets" / "pdf_themes_overview.png"
        if not path.is_file():
            yield event.plain_result("主题概览图暂不可用，可先发送 /pdf主题 查看编号列表。")
            return
        # A bundled real-render image: no LLM, compiler or screenshot browser on command.
        yield event.image_result(str(path))
