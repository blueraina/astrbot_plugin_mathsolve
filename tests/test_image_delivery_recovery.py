"""Regression coverage for explicit image requests that formerly lost their tool."""
import asyncio
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch
import main as plugin_main
from image_reply_policy import classify_image_reply, constrain_image_tool
from test_grading import Event
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import ProviderRequest, LLMResponse
from astrbot.core.message.message_event_result import MessageEventResult
import astrbot.api.message_components as Comp


class ImageIntentTests(unittest.TestCase):
    def test_modifiers_and_question_references(self):
        for text in ['用简洁的图片回答此题', '用清晰易读的长图解答这题', '请用图片简要回答',
                     '图片回答此题', '解答此题', '回答该题', '本题怎么做', '用一张图片回答']:
            with self.subTest(text=text):self.assertTrue(classify_image_reply(text,has_reply=True).allowed)
        for text in ['不要图片，直接文字回答此题', '用PDF解答此题', '用Markdown文件回答本题',
                     '你好', '```markdown\n用图片回答\n```']:
            with self.subTest(text=text):self.assertFalse(classify_image_reply(text).allowed)
    def test_request_local_filtered_tool_returns_on_followup(self):
        original=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool(), plugin_main.Md2ImgSolveMathPdfTool()])
        req=ProviderRequest(func_tool=original)
        constrain_image_tool(req,classify_image_reply('你好'))
        self.assertNotIn('md2img_render_markdown',[t.name for t in req.func_tool.tools])
        constrain_image_tool(req,classify_image_reply('用简洁的图片回答此题'))
        self.assertIn('md2img_render_markdown',[t.name for t in req.func_tool.tools])
        self.assertNotIn('[本轮图片发送限制]',req.system_prompt)
        self.assertEqual(len(original.tools),2)
        constrain_image_tool(req,classify_image_reply('用纯文字回答此题'))
        self.assertNotIn('[本轮优先图片回答]',req.system_prompt)
        self.assertNotIn('md2img_render_markdown',[t.name for t in req.func_tool.tools])
    def test_does_not_add_a_tool_absent_from_original_request(self):
        req=ProviderRequest(func_tool=ToolSet(tools=[]))
        constrain_image_tool(req,classify_image_reply('你好'))
        constrain_image_tool(req,classify_image_reply('用图片回复'))
        self.assertEqual(req.func_tool.tools,[])


class ImageDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.old=plugin_main._MD2IMG_PLUGIN_INSTANCE
        self.context=types.SimpleNamespace(add_llm_tools=lambda *x:None)
        with patch.object(plugin_main,'_safe_get_data_dir',lambda:self.temp.name):
            self.plugin=plugin_main.MarkdownConverterPlugin(self.context,{'enable_math_coach':False})
        self.plugin._pdf_prepare_agent_request_images=AsyncMock(return_value=0)
        self.path=Path(self.temp.name)/'image.png';self.path.write_bytes(b'fake')
        self.plugin._markdown_to_image_playwright=AsyncMock(return_value=[str(self.path)])
        self.event=Event(text='用简洁的图片回答此题')
        self.event.send=AsyncMock(return_value={'message_id':'fake-send'})
        self.wrapper=types.SimpleNamespace(context=types.SimpleNamespace(event=self.event,context=self.context))
    async def asyncTearDown(self):
        plugin_main._MD2IMG_PLUGIN_INSTANCE=self.old;self.temp.cleanup()
    async def test_actual_request_hook_keeps_image_tool(self):
        req=ProviderRequest(func_tool=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool()]))
        await self.plugin.on_llm_req(self.event,req)
        self.assertEqual([t.name for t in req.func_tool.tools],['md2img_render_markdown'])
    async def test_entire_answer_renders_despite_broken_wrappers(self):
        for text in ['<md>## 解答\n$x=1$', '<MD>证明</MD>\n补充结论', '```markdown\n# 解答\n$x=1$\n```']:
            parts=await self.plugin._process_text_with_markdown(text,self.event)
            self.assertTrue(all(isinstance(x,Comp.Image) for x in parts))
            self.assertNotIn('<md>',self.plugin._markdown_to_image_playwright.call_args.kwargs['md_text'].lower())
        self.assertNotIn('```markdown',self.plugin._markdown_to_image_playwright.call_args.kwargs['md_text'])
    async def test_render_error_and_send_error_have_no_raw_fallback(self):
        for renderer in [AsyncMock(return_value=[]),AsyncMock(side_effect=RuntimeError('dead browser'))]:
            self.plugin._markdown_to_image_playwright=renderer
            result=json.loads(await plugin_main.Md2ImgRenderMarkdownTool().call(self.wrapper,markdown='# 解答\n$x=1$'))
            self.assertFalse(result['ok']);self.assertEqual(result['sent'],0)
            resp=LLMResponse(role='assistant',completion_text='# 解答\n$x=1$')
            await self.plugin.on_llm_resp(self.event,resp)
            self.assertIn('失败',resp.completion_text);self.assertNotIn('$x',resp.completion_text)
        self.plugin._markdown_to_image_playwright=AsyncMock(return_value=[str(self.path)])
        self.event.send=AsyncMock(side_effect=RuntimeError('offline'))
        result=json.loads(await plugin_main.Md2ImgRenderMarkdownTool().call(self.wrapper,markdown='# 解答'))
        self.assertFalse(result['ok']);self.assertIn('发送失败',result['error'])
        rendered=MessageEventResult(chain=[Comp.Node(uin='1',name='test',content=[Comp.Plain('# 解答')])])
        self.event.get_result=lambda:rendered
        await self.plugin.on_decorating_result(self.event)
        self.assertEqual(rendered.get_plain_text(),result['error'])
    async def test_success_suppresses_ack_and_internal_pdf_is_not_overwritten(self):
        result=json.loads(await plugin_main.Md2ImgRenderMarkdownTool().call(self.wrapper,markdown='# 解答'))
        self.assertTrue(result['ok'])
        resp=LLMResponse(role='assistant',completion_text='# 重复解答')
        await self.plugin.on_llm_resp(self.event,resp)
        self.assertEqual(resp.completion_text,'')
        rendered=MessageEventResult(chain=[Comp.Reply(id='quoted'),Comp.Plain('图片解答已发送。')])
        self.event.get_result=lambda:rendered
        await self.plugin.on_decorating_result(self.event)
        self.assertEqual(rendered.chain,[])
        self.event.send.assert_awaited_once()
        self.event.set_extra('md2img_pdf_generation',True)
        resp=LLMResponse(role='assistant',completion_text='<solution>保留</solution>')
        await self.plugin.on_llm_resp(self.event,resp)
        self.assertEqual(resp.completion_text,'<solution>保留</solution>')
    async def test_failed_render_retries_without_shared_browser(self):
        self.plugin._ensure_browser=AsyncMock()
        browser=object();self.plugin._browser=browser;self.plugin._pw=object()
        self.plugin._markdown_to_image_playwright_impl=AsyncMock(side_effect=[RuntimeError('dead'),[str(self.path)]])
        paths=await plugin_main.RenderMixin._markdown_to_image_playwright(self.plugin,'answer',str(self.path))
        self.assertEqual(paths,[str(self.path)])
        calls=self.plugin._markdown_to_image_playwright_impl.await_args_list
        self.assertIs(calls[0].kwargs['shared_browser'],browser)
        self.assertIsNone(calls[1].kwargs['shared_browser'])
        self.assertEqual(self.plugin._render_sema._value,2)
    async def test_cancellation_is_not_retried(self):
        self.plugin._ensure_browser=AsyncMock()
        self.plugin._markdown_to_image_playwright_impl=AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await plugin_main.RenderMixin._markdown_to_image_playwright(self.plugin,'answer',str(self.path))
        self.assertEqual(self.plugin._markdown_to_image_playwright_impl.await_count,1)
        self.assertEqual(self.plugin._render_sema._value,2)
