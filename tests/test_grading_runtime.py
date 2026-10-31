"""Exercise actual installed AstrBot components and the plugin input resolver."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main as plugin_main
import astrbot.api.message_components as Comp
from markdown_command import MarkdownMixedMessageFilter
from test_grading import Event


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
    async def test_mixed_image_command_boundary(self):
        event=Event(text='批改检查我的解答')
        event.is_at_or_wake_command=True
        event.message_obj.message=[Comp.Plain('/批改'),Comp.Image.fromBase64('AAAA'),Comp.Plain('检查我的解答')]
        event.get_message_str=lambda:event.message_str
        MarkdownMixedMessageFilter().filter(event,{'wake_prefix':['/']})
        self.assertEqual(event.message_str,'批改\n检查我的解答')


if __name__=='__main__':unittest.main()
