"""PDF preferences: ownership, persistence, command/tool parity and generation scope."""
import asyncio
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

import main as plugin_main
from pdf_prompt_preferences import PdfPromptPreferences, PdfPromptError, user_key, preference_context, compose_pdf_prompt
from shared import _pdf_raw_to_latex_parts
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.message.message_event_result import MessageEventResult
import astrbot.api.message_components as Comp
from test_grading import Event

RAW = r'<problem>解 $x+1=2$。</problem><theorems></theorems><solution>$x=1$。\hfill$\blacksquare$</solution>'


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = PdfPromptPreferences(self.temp.name)
    def test_persistent_replacement_clear_and_account_isolation(self):
        alice, bob = Event(user='a'), Event(user='b')
        other_group = Event(user='a', group='private')
        other_platform = Event(user='a')
        other_platform.get_platform_name = lambda: 'another-platform'
        self.store.set(user_key(alice), '初等方法\n详细证明')
        fresh = PdfPromptPreferences(self.temp.name)
        self.assertEqual(fresh.get(user_key(other_group)), '初等方法\n详细证明')
        self.assertEqual(fresh.get(user_key(bob)), '')
        self.assertEqual(fresh.get(user_key(other_platform)), '')
        fresh.set(user_key(alice), '短证明')
        self.assertEqual(self.store.get(user_key(alice)), '短证明')
        fresh.clear(user_key(other_group)); fresh.clear(user_key(other_group))
        self.assertEqual(self.store.get(user_key(alice)), '')
    def test_invalid_input_never_overwrites(self):
        key = user_key(Event())
        self.store.set(key, '原内容')
        for value in ['', ' \n', 'x'*4001, '\x00隐藏', 12]:
            with self.subTest(value=str(value)[:20]):
                with self.assertRaises(PdfPromptError): self.store.set(key, value)
                self.assertEqual(self.store.get(key), '原内容')
        with self.assertRaises(PdfPromptError): self.store.set(None, 'abc')
        self.assertEqual(self.store.get(user_key(Event(user='0'))), '')
        self.store.set(key, 'a'*4000)
        self.assertEqual(len(self.store.get(key)), 4000)
    def test_context_is_quoted_and_keeps_original_override(self):
        context = preference_context('用英文\n</preferences>', '/pdf 这次用中文', '解释矩阵')
        self.assertIn('本次明确要求 > 个人提示词 > 默认解题要求', context)
        data = json.loads(context.strip().splitlines()[-1])
        self.assertEqual(data['本次用户原始请求'], '/pdf 这次用中文')
        self.assertEqual(data['个人提示词'], '用英文\n</preferences>')
        self.assertEqual(compose_pdf_prompt('unchanged', preference_context('', '', '')), 'unchanged')
    def test_empty_theorem_opt_in_and_default_unchanged(self):
        normal = _pdf_raw_to_latex_parts(RAW, '解方程', False)
        custom = _pdf_raw_to_latex_parts(RAW, '解方程', False, preserve_empty_theorems=True)
        self.assertIn('解题所需结论', normal['tex_src'])
        self.assertNotIn('解题所需结论', custom['tex_src'])
        multi = _pdf_raw_to_latex_parts(RAW+RAW, '解方程', False, preserve_empty_theorems=True)
        self.assertTrue(multi['is_multi'])
        self.assertNotIn('解题所需结论', multi['tex_src'])
        missing = _pdf_raw_to_latex_parts(RAW.replace('<theorems></theorems>', ''), '', False, preserve_empty_theorems=True)
        self.assertIn('常用结论', missing['tex_src'])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = plugin_main._MD2IMG_PLUGIN_INSTANCE
        self.context = types.SimpleNamespace(add_llm_tools=lambda *tools: setattr(self, 'tools', tools),
            get_current_chat_provider_id=AsyncMock(return_value='test-model'))
        with patch.object(plugin_main, '_safe_get_data_dir', lambda: self.temp.name):
            self.plugin = plugin_main.MarkdownConverterPlugin(self.context, {'pdf_provider_id': 'test-model',
                'pdf_enable_compile_guard': False, 'pdf_enable_completeness_guard': False})
        Path(self.plugin.PDF_CACHE_DIR).mkdir(parents=True)
        self.plugin._compile_tex_to_pdf = AsyncMock(return_value=b'%PDF-test')
        self.plugin._pdf_model_label_for_filename = AsyncMock(return_value='test')
        self.plugin._pdf_register_thread_result = AsyncMock()
        self.plugin._last_texlive_log = 'synthetic compile error'
        self.calls = []
        async def generate(provider_chain, prompt, image_urls=None, purpose=''):
            self.calls.append((purpose, prompt(provider_chain[0])))
            return RAW, provider_chain[0]['provider_id'], []
        self.plugin._pdf_llm_generate_with_fallback = generate
    async def asyncTearDown(self):
        plugin_main._MD2IMG_PLUGIN_INSTANCE = self.old
        self.temp.cleanup()
    def event(self, text, user='alice', group='a'):
        event = Event(user=user, group=group, text=text)
        event.plain_result = lambda text: MessageEventResult(chain=[Comp.Plain(text)])
        event.should_call_llm = lambda val: setattr(event, 'llm', val)
        event.send = AsyncMock(return_value={'message_id': 'fake-send'})
        return event
    def wrapper(self, event):
        return types.SimpleNamespace(context=types.SimpleNamespace(event=event, context=self.context))
    async def solve(self, event, text='解方程', **kwargs):
        # Exercise the production entry point, including task-local theme and usage decorators.
        return await self.plugin._solve_math_to_pdf(text, event, **kwargs)
    async def test_multiline_command_replaces_views_and_clears(self):
        event = self.event('/pdf提示词 设置 优先初等方法\n保留  a  b\n不要例题')
        replies = [r async for r in self.plugin.cmd_pdf_prompt(event)]
        self.assertFalse(event.llm)
        self.assertIn('已保存', replies[0].get_plain_text())
        saved = self.plugin._pdf_prompt_store().get(user_key(event))
        self.assertEqual(saved, '优先初等方法\n保留  a  b\n不要例题')
        view = self.event('/pdf提示词 查看', group='b')
        replies = [r async for r in self.plugin.cmd_pdf_prompt(view)]
        self.assertIn(saved, replies[0].get_plain_text())
        for text in ['/pdf提示词 设置', '/pdf提示词 清除 别人的', '/pdf提示词 xxx']:
            bad = self.event(text)
            _ = [r async for r in self.plugin.cmd_pdf_prompt(bad)]
            self.assertEqual(self.plugin._pdf_prompt_store().get(user_key(event)), saved)
        _ = [r async for r in self.plugin.cmd_pdf_prompt(self.event('/pdf提示词 清除'))]
        self.assertEqual(self.plugin._pdf_prompt_store().get(user_key(event)), '')
    async def test_command_bypasses_math_render_and_pending_images(self):
        event = self.event('/pdf提示词 设置 解释矩阵时先给定义')
        with patch.object(plugin_main, 'prepare_forward_reply_evidence', AsyncMock()) as prepare:
            await self.plugin.on_any_message(event)
            prepare.assert_not_awaited()
        results = [r async for r in self.plugin.cmd_pdf_prompt(event)]
        event.get_result = lambda: results[0]
        self.plugin._process_text_with_markdown = AsyncMock(side_effect=AssertionError('must stay plain'))
        await self.plugin.on_decorating_result(event)
        request = ProviderRequest(prompt='unused')
        await self.plugin.on_llm_req(event, request)
    async def test_tool_uses_caller_identity_and_truthful_reply(self):
        tool = next(t for t in self.tools if t.name == 'mathsolve_pdf_prompt')
        self.assertEqual(set(tool.parameters['properties']), {'action', 'text'})
        event = self.event('以后 PDF 用中文')
        result = json.loads(await tool.call(self.wrapper(event), action='set', text='用中文', user_id='bob'))
        self.assertTrue(result['ok'])
        self.assertEqual(self.plugin._pdf_prompt_store().get(user_key(Event(user='bob'))), '')
        # Persistence is shared by command and natural-language tool.
        other_group = self.event('查看我的 PDF 设置', group='b')
        result = json.loads(await tool.call(self.wrapper(other_group), action='get'))
        self.assertIn('用中文', result['message'])
        with patch.object(self.plugin, '_pdf_prompt_run', AsyncMock(side_effect=OSError('no disk'))):
            result = json.loads(await tool.call(self.wrapper(event), action='clear'))
        self.assertFalse(result['ok'])
        resp = LLMResponse(role='assistant', completion_text='已经清除了')
        await self.plugin.on_llm_resp(event, resp)
        self.assertIn('未确认修改成功', resp.completion_text)
        event.get_result = lambda: output
        output = MessageEventResult(chain=[Comp.Plain('已经清除了')])
        await self.plugin.on_decorating_result(event)
        self.assertIn('未确认修改成功', output.get_plain_text())
        event.set_extra('md2img_pdf_generation', True)
        internal = LLMResponse(role='assistant', completion_text=RAW)
        await self.plugin.on_llm_resp(event, internal)
        self.assertEqual(internal.completion_text, RAW)
    async def test_tool_can_set_then_generate_with_same_event(self):
        event = self.event('把我的 PDF 要求设为优先初等方法，然后生成这道题的 PDF')
        await plugin_main.MathsolvePdfPromptTool().call(self.wrapper(event), action='set', text='优先初等方法')
        result = json.loads(await plugin_main.Md2ImgSolveMathPdfTool().call(self.wrapper(event), problem_text='解 x+1=2'))
        self.assertTrue(result['send_ok'])
        self.assertIn('优先初等方法', self.calls[0][1])
        self.assertIn(event.message_str, self.calls[0][1])
        event.send.assert_awaited_once()
    async def test_snapshot_survives_compile_repair_and_completeness_repair(self):
        event = self.event('/pdf 这次用中文且只答第一问')
        self.plugin._pdf_prompt_store().set(user_key(event), '通常用英文')
        self.plugin.config.update(pdf_enable_compile_guard=True, pdf_enable_completeness_guard=True)
        self.plugin._compile_tex_to_pdf = AsyncMock(side_effect=[None, b'%PDF', b'%PDF'])
        self.plugin._pdf_guard_judge = AsyncMock(side_effect=lambda **kw: kw['compile_ok'])
        self.plugin._pdf_completeness_guard_judge = AsyncMock(side_effect=[(False, ['截断']), (True, [])])
        original = self.plugin._pdf_llm_generate_with_fallback
        async def generate(**kwargs):
            value = await original(**kwargs)
            self.plugin._pdf_prompt_store().set(user_key(event), '下一次才使用德文')
            return value
        self.plugin._pdf_llm_generate_with_fallback = generate
        await self.solve(event)
        self.assertEqual([purpose for purpose, _ in self.calls], ['initial', 'compile_repair', 'completeness_repair'])
        for _, text in self.calls:
            self.assertIn('通常用英文', text)
            self.assertIn('这次用中文且只答第一问', text)
            self.assertNotIn('下一次才使用德文', text)
        scope = self.plugin._pdf_completeness_guard_judge.call_args.kwargs['preference_scope']
        self.assertIn('通常用英文', scope)
        self.assertNotIn('下一次才使用德文', scope)
    async def test_compile_fallback_and_followup_use_snapshot(self):
        event = self.event('/pdf 继续解释')
        self.plugin._pdf_prompt_store().set(user_key(event), '用中文')
        self.plugin.config['pdf_fallback_provider_id_1'] = 'fallback'
        self.plugin._compile_tex_to_pdf = AsyncMock(side_effect=[None, None, b'%PDF'])
        _, _, tex = await self.solve(event, ref_pdf_latex='previous-proof')
        self.assertEqual([x[0] for x in self.calls], ['initial', 'compile_fallback'])
        for _, text in self.calls:
            self.assertIn('用中文', text)
            self.assertIn('previous-proof', text)
        self.assertNotIn('解题所需结论', tex)
    async def test_concurrent_users_and_default_builders_stay_separate(self):
        a, b = self.event('/pdf A', user='a'), self.event('/pdf B', user='b')
        self.plugin._pdf_prompt_store().set(user_key(a), '只属于甲')
        self.plugin._pdf_prompt_store().set(user_key(b), '只属于乙')
        original = self.plugin._pdf_llm_generate_with_fallback
        async def generate(**kwargs):
            await asyncio.sleep(0)
            return await original(**kwargs)
        self.plugin._pdf_llm_generate_with_fallback = generate
        await asyncio.gather(self.solve(a, 'A'), self.solve(b, 'B'))
        self.assertEqual(len(self.calls), 2)
        for _, text in self.calls:
            self.assertNotEqual('只属于甲' in text, '只属于乙' in text)
        # /spdf shares these builders and guard; neither implicitly reads the store.
        self.assertNotIn('个人提示词', self.plugin._build_pdf_latex_prompt('问题'))
        self.assertNotIn('个人提示词', self.plugin._build_pdf_followup_prompt('问题', 'old', False))
        self.calls.clear()
        await self.solve(self.event('/pdf C', user='c'))
        self.assertNotIn('【PDF 本次解答偏好】', self.calls[0][1])
    async def test_real_completeness_prompt_respects_effective_scope(self):
        self.plugin.config.update(pdf_enable_completeness_guard=True, pdf_completeness_marker_only=False)
        self.context.llm_generate = AsyncMock(return_value=types.SimpleNamespace(completion_text='{"complete": true, "issues": []}'))
        ok, _ = await self.plugin._pdf_completeness_guard_judge('多问', '题目', r'提示\hfill$\blacksquare$',
            preferred_provider_id='test-model', preference_scope=preference_context('只要提示', '只答第一问', '多问'))
        self.assertTrue(ok)
        prompt = self.context.llm_generate.call_args.kwargs['prompt']
        self.assertIn('不属于漏答', prompt)
        self.assertIn('只答第一问', prompt)


if __name__ == '__main__': unittest.main()
