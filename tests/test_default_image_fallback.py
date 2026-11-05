"""No format keyword or image tool call is required for final image delivery."""
import unittest
import json
import types
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
    async def test_ordinary_unspecified_question_stays_text(self):
        self.plugin._router_classify=AsyncMock(return_value={'is_math':False})
        event=Event(text='请详细解释这个观点')
        req=ProviderRequest(func_tool=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool()]))
        await self.plugin.on_llm_req(event,req)
        self.assertEqual(req.func_tool.tools,[])
        self.assertFalse(image_reply_decision(event).allowed)
        self.assertFalse(image_reply_decision(event).math_request)
        self.assertIn('[本轮图片发送限制]',req.system_prompt)
        result=MessageEventResult(chain=[Comp.Plain('这个观点包含两个部分，需要分别看待。')]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertIsInstance(result.chain[0],Comp.Plain)
        self.plugin._markdown_to_image_playwright.assert_not_awaited()

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

    async def test_casual_game_chat_never_renders_plain_or_lightly_formatted_replies(self):
        self.plugin._router_classify=AsyncMock(return_value={'is_math':False})
        for question in ['原来你是这个战队的粉丝', '怎么评价刚才那场比赛？', '你今天吃了什么？']:
            for answer in ['我只是觉得刚才那一波操作很精彩。', '这场比赛主要看选手发挥。' * 80,
                           '## 我的看法\n**第一点**：操作确实很精彩。\n**第二点**：配合也很重要。' * 6]:
                event=Event(text=question)
                req=ProviderRequest(func_tool=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool()]))
                await self.plugin.on_llm_req(event,req)
                self.assertEqual(req.func_tool.tools,[])
                response=LLMResponse(role='assistant',completion_text='<md>'+answer+'</md>')
                await self.plugin.on_llm_resp(event,response)
                result=MessageEventResult(chain=[Comp.Plain(response.completion_text)]);event.get_result=lambda:result
                await self.plugin.on_decorating_result(event)
                self.assertEqual(result.get_plain_text(),answer)
                self.assertTrue(all(isinstance(x,Comp.Plain) for x in result.chain))
        self.plugin._markdown_to_image_playwright.assert_not_awaited()

    async def test_casual_chat_cannot_force_the_tool_by_inventing_math_arguments(self):
        event=Event(text='原来你是这个战队的粉丝');event.send=AsyncMock()
        wrapper=types.SimpleNamespace(context=types.SimpleNamespace(event=event,context=self.context))
        result=json.loads(await plugin_main.Md2ImgRenderMarkdownTool().call(wrapper,markdown=BODY))
        self.assertFalse(result['ok']);self.assertEqual(result['sent'],0)
        event.send.assert_not_awaited()
        self.plugin._markdown_to_image_playwright.assert_not_awaited()

    async def test_explicit_image_request_keeps_even_casual_reply_as_image(self):
        event=Event(text='用图片回答：你今天吃了什么？')
        self.assertTrue(image_reply_decision(event).allowed)
        parts=await self.plugin._process_text_with_markdown('今天吃了面条。',event)
        self.assertIsInstance(parts[0],Comp.Image)

    async def test_long_structured_markdown_still_uses_independent_fallback(self):
        event=Event(text='继续')
        structured=''.join(f'## 第 {i} 部分\n**要点**：'+('详细讨论这里的背景、条件和结果。'*5)+'\n' for i in range(5))
        self.assertTrue(dense_markdown(structured))
        parts=await self.plugin._process_text_with_markdown(structured,event)
        self.assertIsInstance(parts[0],Comp.Image)

    async def test_captured_casual_followup_drops_prior_math_delivery_permission(self):
        event=Event(text='解释同构');follow=Event(text='原来你是这个战队的粉丝')
        follow.set_extra('_follow_up_captured',{'target_run_id':event.message_obj.message_id})
        event.set_extra('mathsolve_followup_candidates',[(event.message_obj.message_id,follow)])
        self.assertFalse(image_reply_decision(event).allowed)
        self.assertFalse(should_render_answer(event,'只是聊聊比赛。'))
