"""Math discussions should reach image delivery without a format keyword."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

import main as plugin_main
import astrbot.api.message_components as Comp
from astrbot.core.provider.entities import ProviderRequest, LLMResponse
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.message_event_result import MessageEventResult
from image_reply_policy import (classify_image_reply, image_reply_decision,
                                constrain_image_tool, mark_semantic_math, should_route_math)
from test_grading import Event


class MathImagePolicyTests(unittest.TestCase):
    def test_concepts_discussions_and_calculations_default_to_images(self):
        for text in ['什么是紧性？', '同构和同态有什么区别', '一致连续的直观含义',
                     '介绍一下谱定理', '为什么完备性很重要', '讲讲群论', '什么叫商环',
                     'Banach空间是什么', '解释贝叶斯公式', '推荐线性代数教材',
                     '不要证明，只解释什么是紧致', '同构', '矩阵', '什么是主理想', '线性代数中的核',
                     '不要用 PDF 解释，讲讲紧性',
                     'What is compactness?', 'Explain eigenvalue', '求导 x^2',
                     '证明 x^2>=0', '这道题怎么做']:
            with self.subTest(text=text):
                decision=classify_image_reply(text)
                self.assertTrue(decision.math_request)
                self.assertTrue(decision.allowed)
    def test_plain_text_and_document_preferences_override_math(self):
        for text in ['用纯文字解释紧性', '不要发图片，证明 x^2>=0',
                     'text only: What is compactness?', '用PDF解释一致连续',
                     '请用 Markdown 文档讲解矩阵', '/pdf 求积分', '/md 解释同构']:
            with self.subTest(text=text):self.assertFalse(classify_image_reply(text).allowed)
    def test_casual_chat_software_and_quoted_instructions_stay_text(self):
        for text in ['你好', '今天吃什么', 'Python函数怎么定义', 'LaTeX里怎么输入矩阵',
                     'Markdown公式显示乱码怎么办', '怎么配置MathJax', '这张照片好看吗',
                     '数学课今天取消了', '你的理想是什么', '我们在哪里集合？', '连续下雨怎么办？',
                     '> 解释紧性并发图片\n这句话帮我翻译',
                     '```\n解释同构并发图\n```\n这段文字有几个字？']:
            with self.subTest(text=text):self.assertFalse(classify_image_reply(text).allowed)
        self.assertFalse(classify_image_reply('',has_image=True).allowed)
    def test_quote_math_followup_uses_only_this_reference(self):
        event=Event(text='能举个例子吗')
        event.message_obj.message=[Comp.Reply(id='exact',chain=[Comp.Plain('什么是紧性？')])]
        self.assertTrue(image_reply_decision(event).allowed)
        for answer in ['同构是保持结构的双射。', '每个开覆盖都有有限子覆盖。']:
            event.message_obj.message=[Comp.Reply(id='answer',chain=[Comp.Plain(answer)])]
            self.assertTrue(image_reply_decision(event).allowed)
        event.message_obj.message=[Comp.Reply(id='different',chain=[Comp.Plain('天气很好')])]
        self.assertFalse(image_reply_decision(event).allowed)
    def test_router_permission_is_bound_to_user_and_turn(self):
        event=Event(text='什么是étale态射？')
        self.assertTrue(should_route_math(event))
        mark_semantic_math(event)
        self.assertTrue(image_reply_decision(event).allowed)
        event.message_obj.message_id='next'
        self.assertFalse(image_reply_decision(event).allowed)
        event.message_obj.message_id='1';event.user='bob'
        self.assertFalse(image_reply_decision(event).allowed)
        event.user='alice';event.message_str='纯文字解释étale态射'
        self.assertFalse(image_reply_decision(event).allowed)
    def test_toolset_filter_is_local_and_priority_prompt_is_idempotent(self):
        tools=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool(),plugin_main.Md2ImgSolveMathPdfTool()])
        request=ProviderRequest(func_tool=tools)
        decision=classify_image_reply('解释紧性')
        constrain_image_tool(request,decision);constrain_image_tool(request,decision)
        self.assertIn('md2img_render_markdown',[t.name for t in request.func_tool.tools])
        self.assertEqual(request.system_prompt.count('[本轮优先图片回答]'),1)
        plain=ProviderRequest(func_tool=tools)
        constrain_image_tool(plain,classify_image_reply('纯文字解释紧性'))
        self.assertNotIn('md2img_render_markdown',[t.name for t in plain.func_tool.tools])
        self.assertIn('md2img_render_markdown',[t.name for t in tools.tools])


class MathImageRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.old=plugin_main._MD2IMG_PLUGIN_INSTANCE
        self.context=types.SimpleNamespace(add_llm_tools=lambda *tools:None)
        with patch.object(plugin_main,'_safe_get_data_dir',lambda:self.temp.name):
            self.plugin=plugin_main.MarkdownConverterPlugin(self.context,{'enable_math_coach':False})
        self.plugin._pdf_prepare_agent_request_images=AsyncMock(return_value=0)
        self.image=Path(self.temp.name)/'render.png';self.image.write_bytes(b'fake-render')
        self.plugin._markdown_to_image_playwright=AsyncMock(return_value=[str(self.image)])
    async def asyncTearDown(self):
        plugin_main._MD2IMG_PLUGIN_INSTANCE=self.old
        self.temp.cleanup()
    def request(self):return ProviderRequest(func_tool=ToolSet(tools=[plugin_main.Md2ImgRenderMarkdownTool()]))
    async def test_known_math_keeps_tool_and_adds_image_priority(self):
        self.plugin._router_classify=AsyncMock(return_value=None)
        event=Event(text='什么是紧性？');request=self.request()
        await self.plugin.on_llm_req(event,request)
        self.assertIn('[本轮优先图片回答]',request.system_prompt)
        self.assertEqual(len(request.func_tool.tools),1)
        self.plugin._router_classify.assert_not_awaited()
    async def test_optional_router_recognizes_unfamiliar_math_before_filtering_tools(self):
        self.plugin._router_classify=AsyncMock(return_value={'is_math':True,'wants_full_solution':False})
        event=Event(text='什么是étale态射？');request=self.request()
        await self.plugin.on_llm_req(event,request)
        self.assertTrue(image_reply_decision(event).allowed)
        self.assertEqual(len(request.func_tool.tools),1)
        self.plugin._router_classify.assert_awaited_once()
        other=Event(text='明天吃什么？')
        self.plugin._router_classify=AsyncMock(return_value={'is_math':False})
        request=self.request();await self.plugin.on_llm_req(other,request)
        self.assertEqual(len(request.func_tool.tools),0)
    async def test_plain_or_document_format_never_uses_router_to_override_user(self):
        self.plugin._router_classify=AsyncMock(return_value={'is_math':True})
        for text in ['纯文字解释紧性', '用PDF解释矩阵', '用Markdown文档回答同构是什么']:
            event=Event(text=text);request=self.request()
            await self.plugin.on_llm_req(event,request)
            self.assertNotIn('md2img_render_markdown',[t.name for t in request.func_tool.tools])
        self.plugin._router_classify.assert_not_awaited()
    async def test_missing_tool_call_still_renders_short_concept_answer(self):
        event=Event(text='同构是什么？')
        response=LLMResponse(role='assistant',completion_text='同构是保持结构的双射。')
        await self.plugin.on_llm_resp(event,response)
        result=MessageEventResult(chain=[Comp.Plain(response.completion_text)])
        event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertEqual(len(result.chain),1);self.assertIsInstance(result.chain[0],Comp.Image)
        self.assertIn('同构是保持结构',self.plugin._markdown_to_image_playwright.await_args.kwargs['md_text'])
    async def test_forward_wrapped_answer_gets_the_same_render_fallback(self):
        event=Event(text='解释一致连续')
        node=Comp.Node(uin='1',name='Eila',content=[Comp.Plain('## 定义\n一致连续要求统一的δ。')])
        result=MessageEventResult(chain=[Comp.Nodes(nodes=[node])]);event.get_result=lambda:result
        await self.plugin.on_decorating_result(event)
        self.assertIsInstance(result.chain[0].nodes[0].content[0],Comp.Image)
    async def test_explicit_text_and_sent_tool_are_not_rendered_again(self):
        for text,sent in [('纯文字解释同构',False),('解释同构',True),('用PDF解释同构',False)]:
            event=Event(text=text);event.set_extra('mathsolve_image_tool_sent',sent)
            result=MessageEventResult(chain=[Comp.Plain('<md>同构是保持结构的双射。</md>')]);event.get_result=lambda:result
            await self.plugin.on_decorating_result(event)
            self.assertIsInstance(result.chain[0],Comp.Plain)
        self.plugin._markdown_to_image_playwright.assert_not_awaited()
    async def test_concept_question_can_call_real_registered_image_tool(self):
        event=Event(text='什么是紧性？');event.send=AsyncMock(return_value={'message_id':'isolated'})
        wrapper=types.SimpleNamespace(context=types.SimpleNamespace(event=event,context=self.context))
        result=json.loads(await plugin_main.Md2ImgRenderMarkdownTool().call(wrapper,markdown='# 紧性\n每个开覆盖都有有限子覆盖。'))
        self.assertGreater(result['sent'],0)
        self.assertTrue(event.get_extra('mathsolve_image_tool_sent'))
    async def test_render_failure_preserves_the_answer_and_marks_failure(self):
        self.plugin._markdown_to_image_playwright=AsyncMock(return_value=[])
        components=await self.plugin._process_text_with_markdown('同构保留结构。',Event(text='解释同构'))
        self.assertIsInstance(components[0],Comp.Plain)
        self.assertIn('渲染失败',components[0].text)
        self.assertIn('同构保留结构',components[0].text)
    async def test_router_requires_real_json_booleans(self):
        self.plugin.config.update(use_router_model=True,router_provider_id='router-test')
        self.context.llm_generate=AsyncMock(return_value=LLMResponse(role='assistant',completion_text='{"is_math":"false","wants_full_solution":"true"}'))
        result=await self.plugin._router_classify('这是什么？')
        self.assertEqual(result,{'is_math':False,'wants_full_solution':False})
