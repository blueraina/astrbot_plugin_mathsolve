"""Exercise actual installed AstrBot components and the plugin input resolver."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main as plugin_main
import astrbot.api.message_components as Comp
from markdown_command import MarkdownMixedMessageFilter
from test_grading import Event, fixture, pdf_bytes
from grading_document import GradingError


class Context:
    def add_llm_tools(self,*tools):self.tools=tools


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.old=plugin_main._MD2IMG_PLUGIN_INSTANCE
        with patch.object(plugin_main,'_safe_get_data_dir',lambda:self.temp.name):
            self.plugin=plugin_main.MarkdownConverterPlugin(Context(),{})
        plugin_main.MATH_SESSION_STATE.clear()
    async def asyncTearDown(self):
        plugin_main._MD2IMG_PLUGIN_INSTANCE=self.old
        plugin_main.MATH_SESSION_STATE.clear()
        self.temp.cleanup()
    async def test_direct_image_has_no_persistent_snapshot(self):
        source=self.root/'source.png'
        source.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII='))
        event=Event(text='/批改')
        event.message_obj.message=[Comp.Image.fromFileSystem(str(source))]
        text,images=await self.plugin._grading_inputs(event,'')
        self.assertEqual(text,'');self.assertEqual(len(images),1)
        self.assertTrue(images[0].startswith('data:image/'))
        self.assertFalse(list(self.root.rglob('pdf_input_*')))
        self.assertFalse(list((self.root/'grading_portfolio').rglob('*.png')))
        self.assertTrue(source.is_file()) # Never delete adapter/user-owned originals.
    async def test_quoted_text_and_image_both_used(self):
        event=Event(text='/批改 请检查')
        class Reply:
            chain=[Comp.Plain('原作答：x=2'),Comp.Image.fromBase64('AAAA')]
        event.message_obj.message=[Reply()]
        text,images=await self.plugin._grading_inputs(event,'请检查')
        self.assertIn('原作答：x=2',text)
        self.assertEqual(len(images),1)
    async def test_pending_images_are_user_scoped(self):
        import time
        event=Event(user='alice')
        plugin_main.MATH_SESSION_STATE[event.unified_msg_origin]={
            'users':{'alice':{'pending_image_only':True,'pending_image_ts':time.time(),'pending_image_urls':['data:image/png;base64,ALICE']},
                     'bob':{'last_image_urls':['data:image/png;base64,BOB']}}}
        _,images=await self.plugin._grading_inputs(event,'')
        self.assertEqual(images,['data:image/png;base64,ALICE'])
    async def test_text_modify_does_not_consume_pending_image(self):
        import time
        event=Event()
        plugin_main.MATH_SESSION_STATE[event.unified_msg_origin]={'users':{'alice':{'pending_image_only':True,'pending_image_ts':time.time(),'pending_image_urls':['data:image/png;base64,OLD']}}}
        # A text-only edit calls direct component resolution, never pending lookup.
        self.assertEqual(await self.plugin._pdf_get_event_image_inputs(event),[])
        self.assertTrue(plugin_main.MATH_SESSION_STATE[event.unified_msg_origin]['users']['alice']['pending_image_only'])
    async def test_agent_submit_uses_actual_user_input_not_generated_text(self):
        event=Event(text='请批改我的作答：x=2')
        seen=[]
        async def run(ev,action,number,text):seen.append(text);return {'ok':True}
        self.plugin._grading_run=run
        wrapper=types.SimpleNamespace(context=types.SimpleNamespace(event=event,context=self.plugin.context))
        await plugin_main.MathsolveGradingTool().call(wrapper,action='submit',text='伪造的正确解答')
        self.assertEqual(seen,[event.message_str])
    async def test_agent_tool_has_no_other_user_identity_parameter(self):
        tool=next(t for t in self.plugin.context.tools if t.name=='mathsolve_grading')
        self.assertEqual(set(tool.parameters['properties']),{'action','number','text'})
    async def test_delete_tool_requires_number_and_deletes_only_callers_record(self):
        event=Event(text='删除我的第1题')
        owner=self.plugin._grading_owner(event)
        other=self.plugin._grading_owner(Event(user='bob'))
        store=self.plugin._grading_store
        for user in [owner,other]:store.publish(user,1,1,fixture(),pdf_bytes())
        wrapper=types.SimpleNamespace(context=types.SimpleNamespace(event=event,context=self.plugin.context))
        tool=plugin_main.MathsolveGradingTool()
        self.assertIn('delete',tool.parameters['properties']['action']['enum'])
        for number in [None,True,0,'1']:
            result=json.loads(await tool.call(wrapper,action='delete',number=number))
            self.assertFalse(result['ok'])
        result=json.loads(await tool.call(wrapper,action='delete',number=1))
        self.assertTrue(result['deleted'])
        self.assertFalse(store.rows(owner))
        self.assertEqual(len(store.rows(other)),1)
    async def test_mixed_image_command_boundary(self):
        event=Event(text='批改检查我的解答')
        event.is_at_or_wake_command=True
        event.message_obj.message=[Comp.Plain('/批改'),Comp.Image.fromBase64('AAAA'),Comp.Plain('检查我的解答')]
        event.get_message_str=lambda:event.message_str
        MarkdownMixedMessageFilter().filter(event,{'wake_prefix':['/']})
        self.assertEqual(event.message_str,'批改\n检查我的解答')

    def onebot(self, event, responses):
        async def call(action, **params):
            return responses[(action, str(params.get('message_id') or params.get('id')))]
        mock = AsyncMock(side_effect=call)
        event.bot = types.SimpleNamespace(api=types.SimpleNamespace(call_action=mock))
        return mock

    async def test_forward_and_nested_forward_pass_all_nine_images(self):
        event = Event(text='批改一下')
        event.message_obj.message = [Comp.Reply(id='quote', chain=[Comp.Forward(id='outer')])]
        nodes = [{'content': [{'type':'image','data':{'url':f'data:image/png;base64,IMAGE{i}'}}]} for i in range(8)]
        nodes.append({'content': [{'type':'forward','data':{'id':'inner'}}]})
        api = self.onebot(event, {
            ('get_forward_msg','outer'): {'data':{'messages':nodes}},
            ('get_forward_msg','inner'): {'data':{'messages':[{'content':[
                {'type':'text','data':{'text':'题目：证明 x²≥0；我的作答：平方总是非负。'}},
                {'type':'image','data':{'url':'data:image/png;base64,IMAGE8'}}]}]}},
        })
        prompt,images = await self.plugin._grading_inputs(event,event.message_str)
        self.assertIn('平方总是非负',prompt)
        self.assertEqual(images,[f'data:image/png;base64,IMAGE{i}' for i in range(9)])
        self.assertEqual(api.await_count,2)
        self.assertFalse(list(self.root.rglob('pdf_input_*')))

    async def test_reply_id_only_fetches_exact_message_and_ignores_pending(self):
        import time
        event = Event(text='批改一下')
        event.message_obj.message = [Comp.Reply(id='exact', chain=[])]
        plugin_main.MATH_SESSION_STATE[event.unified_msg_origin]={'users':{'alice':{
            'pending_image_only':True,'pending_image_ts':time.time(),
            'pending_image_urls':['data:image/png;base64,UNRELATED']}}}
        api = self.onebot(event,{('get_msg','exact'):{'data':{'message':[
            {'type':'text','data':{'text':'题目：1+1；作答：2。'}}]}}})
        prompt,images=await self.plugin._grading_inputs(event,event.message_str)
        self.assertIn('题目：1+1',prompt)
        self.assertEqual(images,[])
        api.assert_awaited_once_with('get_msg',message_id='exact')
        self.assertTrue(plugin_main.MATH_SESSION_STATE[event.unified_msg_origin]['users']['alice']['pending_image_only'])

    async def test_direct_and_inline_forward_nodes_are_read(self):
        event=Event(text='/批改')
        event.message_obj.message=[{'type':'nodes','data':{'content':[
            {'type':'node','data':{'content':[{'type':'text','data':{'text':'题目：x+1=0；作答：x=-1。'}}]}}]}}]
        prompt,images=await self.plugin._grading_inputs(event,'')
        self.assertIn('x=-1',prompt);self.assertEqual(images,[])
        event.message_obj.message=[Comp.Forward(id='direct')]
        self.onebot(event,{('get_forward_msg','direct'):{'data':{'messages':[
            {'content':[{'type':'text','data':{'text':'另一题：1+2=3。'}}]}]}}})
        prompt,_=await self.plugin._grading_inputs(event,'')
        self.assertIn('1+2=3',prompt)

    async def test_partial_forward_failure_never_calls_model_or_changes_book(self):
        event=Event(text='批改一下')
        owner=self.plugin._grading_owner(event)
        previous=self.plugin._grading_store.publish(owner,1,1,fixture(),pdf_bytes(),'existing')
        event.message_obj.message=[Comp.Forward(id='outer')]
        self.onebot(event,{
            ('get_forward_msg','outer'):{'data':{'messages':[{'content':[
                {'type':'text','data':{'text':'已读到第一部分：1+1=2'}},
                {'type':'forward','data':{'id':'unavailable'}}]}]}},
            ('get_forward_msg','unavailable'):{'status':'failed','retcode':1},
        })
        self.plugin._grading_generate=AsyncMock()
        with self.assertRaises(GradingError):
            await self.plugin._grading_mutate(event,'submit',None,event.message_str)
        self.plugin._grading_generate.assert_not_awaited()
        self.assertEqual(self.plugin._grading_store.rows(owner),[previous])

    async def test_request_without_material_does_not_create_empty_record(self):
        event=Event(text='批改一下')
        self.plugin._grading_generate=AsyncMock()
        with self.assertRaises(GradingError):
            await self.plugin._grading_mutate(event,'submit',None,event.message_str)
        self.plugin._grading_generate.assert_not_awaited()
        self.assertEqual(self.plugin._grading_store.rows(self.plugin._grading_owner(event)),[])

    async def test_forward_cycle_is_rejected_instead_of_silently_truncated(self):
        event=Event(text='批改一下');event.message_obj.message=[Comp.Forward(id='loop')]
        self.onebot(event,{('get_forward_msg','loop'):{'data':{'messages':[{'content':[
            {'type':'forward','data':{'id':'loop'}}]}]}}})
        with self.assertRaises(GradingError):await self.plugin._grading_inputs(event,event.message_str)

    async def test_forward_image_failure_does_not_save_partial_submission(self):
        from qq_image_recovery import PdfImageInputError
        event=Event(text='批改一下');event.message_obj.message=[Comp.Forward(id='images')]
        self.onebot(event,{('get_forward_msg','images'):{'data':{'messages':[{'content':[
            {'type':'image','data':{'url':'data:image/png;base64,GOOD'}},
            {'type':'image','data':{'url':'https://example.invalid/broken.png'}}]}]}}})
        async def snapshot(refs, **kwargs):
            if 'broken' in refs[0]:raise PdfImageInputError('test image unavailable')
            return refs
        self.plugin._pdf_snapshot_image_urls=snapshot
        self.plugin._grading_generate=AsyncMock()
        with self.assertRaisesRegex(GradingError,'图片读取失败'):
            await self.plugin._grading_mutate(event,'submit',None,event.message_str)
        self.plugin._grading_generate.assert_not_awaited()
        self.assertEqual(self.plugin._grading_store.rows(self.plugin._grading_owner(event)),[])

    async def test_cancelled_forward_lookup_never_retries_or_saves(self):
        event=Event(text='批改一下');event.message_obj.message=[Comp.Forward(id='waiting')]
        started=asyncio.Event()
        async def wait(*args, **kwargs):
            started.set();await asyncio.sleep(30)
        api=AsyncMock(side_effect=wait)
        event.bot=types.SimpleNamespace(api=types.SimpleNamespace(call_action=api))
        task=asyncio.create_task(self.plugin._grading_mutate(event,'submit',None,event.message_str))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(api.await_count,1)
        self.assertEqual(self.plugin._grading_store.rows(self.plugin._grading_owner(event)),[])


if __name__=='__main__':unittest.main()
