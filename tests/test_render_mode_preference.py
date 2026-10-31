# -*- coding: utf-8 -*-
import asyncio
import json
import os
import sys
import tempfile
import unittest


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

import main as plugin_main
import shared as plugin_shared


class FakeEvent:
    def __init__(self, user_id="10001", raw_message=None):
        self._user_id = user_id
        self.raw_message = raw_message
        self.message_str = "请用图片回复"
        self._extra = {}

    def get_sender_id(self):
        return self._user_id

    def get_extra(self, key):
        return self._extra.get(key)

    def set_extra(self, key, value):
        self._extra[key] = value


class FakeAgentContext:
    def __init__(self, event):
        self.event = event
        self.context = None


class FakeWrapper:
    def __init__(self, event):
        self.context = FakeAgentContext(event)


class FakePlugin:
    def __init__(self, cache_dir):
        self.IMAGE_CACHE_DIR = cache_dir
        self.rendered_mode = None

    async def _markdown_to_image_playwright(self, **kwargs):
        self.rendered_mode = kwargs["mode"]
        return []

    def _cfg(self, _key, default=None):
        return default


class FakeRenderPlugin(plugin_main.RenderMixin):
    def __init__(self, cache_dir):
        self.IMAGE_CACHE_DIR = cache_dir
        self.rendered_mode = None

    async def _markdown_to_image_playwright(self, **kwargs):
        self.rendered_mode = kwargs["mode"]
        return []


class RenderModePreferenceTests(unittest.TestCase):
    def setUp(self):
        plugin_shared.USER_PREFERENCES.clear()

    def tearDown(self):
        plugin_shared.USER_PREFERENCES.clear()

    def test_explicit_pc_overrides_tool_mobile(self):
        event = FakeEvent()
        self.assertTrue(plugin_shared.set_user_mode_preference(event.get_sender_id(), "pc"))
        self.assertEqual(plugin_shared.resolve_render_mode(event, "mobile"), "pc")

    def test_explicit_mobile_overrides_tool_pc(self):
        event = FakeEvent()
        self.assertTrue(plugin_shared.set_user_mode_preference(event.get_sender_id(), "mobile"))
        self.assertEqual(plugin_shared.resolve_render_mode(event, "pc"), "mobile")

    def test_tool_mode_is_used_without_explicit_preference(self):
        event = FakeEvent()
        self.assertEqual(plugin_shared.resolve_render_mode(event, "pc"), "pc")
        self.assertEqual(plugin_shared.resolve_render_mode(event, "mobile"), "mobile")


class RenderToolPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        plugin_shared.USER_PREFERENCES.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plugin = FakePlugin(self.temp_dir.name)
        self.old_plugin = plugin_main._MD2IMG_PLUGIN_INSTANCE
        plugin_main._MD2IMG_PLUGIN_INSTANCE = self.plugin

    async def asyncTearDown(self):
        plugin_main._MD2IMG_PLUGIN_INSTANCE = self.old_plugin
        plugin_shared.USER_PREFERENCES.clear()
        self.temp_dir.cleanup()

    async def test_real_tool_call_obeys_explicit_pc(self):
        event = FakeEvent()
        plugin_shared.set_user_mode_preference(event.get_sender_id(), "pc")

        tool = plugin_main.Md2ImgRenderMarkdownTool()
        payload = json.loads(
            await tool.call(FakeWrapper(event), markdown="# test", mode="mobile", scale=2)
        )

        self.assertEqual(self.plugin.rendered_mode, "pc")
        self.assertEqual(payload["mode"], "pc")

    async def test_normal_markdown_render_obeys_explicit_pc(self):
        event = FakeEvent()
        plugin_shared.set_user_mode_preference(event.get_sender_id(), "pc")
        renderer = FakeRenderPlugin(self.temp_dir.name)

        await renderer._process_text_with_markdown("<md># test</md>", event)

        self.assertEqual(renderer.rendered_mode, "pc")


if __name__ == "__main__":
    unittest.main()
