"""No format keyword or image tool call is required for final image delivery."""
import unittest
from unittest.mock import AsyncMock
import astrbot.api.message_components as Comp
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.provider.entities import ProviderRequest, LLMResponse
from astrbot.core.agent.tool import ToolSet
import main as plugin_main
from image_reply_policy import dense_markdown, should_render_answer, image_reply_decision
from test_grading import Event
import test_image_delivery_recovery as fixtures

BODY = ('## 方法\n**第一步**：分析已知条件，再逐步展开说明。\n' * 3 +
        '\n| 步骤 | 结果 |\n| --- | --- |\n| 计算 | $x=1$ |\n' +
        '\n$$\\int_0^1 2x\\,dx=1$$\n')


class DefaultImageFallbackTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ImageDeliveryTests.asyncSetUp
    asyncTearDown = fixtures.ImageDeliveryTests.asyncTearDown
    async def test_ordinary_unspecified_question_retains_tool_without_math_router(self):
        self.plugin._router_classify=AsyncMock(return_value={'is_math':False})
        event=Event(text='请详细解释这个观点')
        req=ProviderRequest(func_tool=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool()]))
        await self.plugin.on_llm_req(event,req)
        self.assertEqual([t.name for t in req.func_tool.tools],['md2img_render_markdown'])
        self.assertTrue(image_reply_decision(event).default_image)
        self.assertFalse(image_reply_decision(event).math_request)
        self.assertIn('未指定输出格式',req.system_prompt)
        result=MessageEventResult(chain=[Comp.Plain('这个观点包含两个部分，需要分别看待。')]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertIsInstance(result.chain[0],Comp.Image)

    async def test_dense_output_renders_even_when_request_has_no_image_or_math_intent(self):
        event=Event(text='你好')
        self.assertFalse(image_reply_decision(event).allowed)
        self.assertTrue(dense_markdown(BODY))
        for text in [BODY, '```markdown\n'+BODY+'\n```', '```md\n'+BODY+'\n```']:
            self.assertTrue(should_render_answer(event,text))
            parts=await self.plugin._process_text_with_markdown(text,event)
            self.assertTrue(all(isinstance(x,Comp.Image) for x in parts))
            self.assertNotIn('```',self.plugin._markdown_to_image_playwright.call_args.kwargs['md_text'])

    async def test_final_chain_coalesces_fragments_and_preserves_equations(self):
        event=Event(text='继续详细说明')
        parts=[Comp.Plain(BODY[:90]),Comp.Plain(BODY[90:])]
        node=Comp.Node(uin='1',name='test',content=parts)
        result=MessageEventResult(chain=[Comp.Nodes(nodes=[node])]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.plugin._markdown_to_image_playwright.assert_awaited_once()
        self.assertEqual(self.plugin._markdown_to_image_playwright.call_args.kwargs['md_text'],BODY.strip())
        self.assertTrue(all(isinstance(x,Comp.Image) for x in node.content))
        await self.plugin.on_decorating_result(event)
        self.plugin._markdown_to_image_playwright.assert_awaited_once()

    async def test_user_formats_internal_generation_and_commands_override_dense_fallback(self):
        for text in ['用纯文字回答', '不要图片', '给我md', '用 Markdown 文档回答',
                     '请输出PDF', 'PDF格式回复', '直接给我源码', '/pdf 求积分', '/md 解释', '/帮助']:
            with self.subTest(text=text):
                event=Event(text=text)
                self.assertFalse(should_render_answer(event,BODY))
                result=MessageEventResult(chain=[Comp.Plain(BODY)]);event.get_result=lambda:result
                await self.plugin.on_decorating_result(event)
                self.assertIsInstance(result.chain[0],Comp.Plain)
        for flag in ['md2img_pdf_generation','mathsolve_md_generation','mathsolve_grading_generation']:
            event=Event(text='详细说明');event.set_extra(flag,True)
            self.assertFalse(should_render_answer(event,BODY))
            parts=await self.plugin._process_text_with_markdown(BODY,event)
            self.assertIsInstance(parts[0],Comp.Plain)
        self.plugin._markdown_to_image_playwright.assert_not_awaited()

    async def test_plain_acknowledgement_stays_short_and_delivery_is_not_duplicated(self):
        event=Event(text='你好')
        result=MessageEventResult(chain=[Comp.Plain('你好！')]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertEqual(result.get_plain_text(),'你好！')
        event=Event(text='详细说明');event.set_extra('mathsolve_image_tool_sent',True)
        self.assertFalse(should_render_answer(event,BODY))
        self.plugin._markdown_to_image_playwright.assert_not_awaited()

    async def test_later_hook_can_add_dense_markdown_after_llm_response(self):
        event=Event(text='你好')
        response=LLMResponse(role='assistant',completion_text='你好！')
        await self.plugin.on_llm_resp(event,response)
        result=MessageEventResult(chain=[Comp.Plain(BODY)]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertIsInstance(result.chain[0],Comp.Image)

    async def test_full_math_flow_keeps_text_outside_model_wrappers(self):
        event=Event(text='解释同构')
        response=LLMResponse(role='assistant',completion_text='前置条件\n<md>定义</md>\n必要补充')
        await self.plugin.on_llm_resp(event,response)
        result=MessageEventResult(chain=[Comp.Plain(response.completion_text)]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        rendered=self.plugin._markdown_to_image_playwright.call_args.kwargs['md_text']
        for value in ['前置条件','定义','必要补充']:self.assertIn(value,rendered)

    async def test_explicit_document_format_persists_in_captured_followup(self):
        event=Event(text='用PDF回答');follow=Event(text='继续详细说明')
        follow.set_extra('_follow_up_captured',{'target_run_id':event.message_obj.message_id})
        event.set_extra('mathsolve_followup_candidates',[(event.message_obj.message_id,follow)])
        self.assertTrue(image_reply_decision(event).document_requested)
        self.assertFalse(should_render_answer(event,BODY))
        follow.message_str='改用图片回答'
        self.assertTrue(should_render_answer(event,BODY))
