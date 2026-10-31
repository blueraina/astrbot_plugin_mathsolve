import asyncio
import importlib.util
import inspect
import json
import re
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def _load_pdf_mixin():
    shared = types.ModuleType("shared")
    exported = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "AstrMessageEvent": type("AstrMessageEvent", (), {}),
        "asyncio": asyncio,
        "inspect": inspect,
        "logger": _Logger(),
        "re": re,
        "time": time,
    }
    for name, value in exported.items():
        setattr(shared, name, value)
    shared.__all__ = list(exported)

    previous_shared = sys.modules.get("shared")
    sys.modules["shared"] = shared
    try:
        module_path = Path(__file__).resolve().parents[1] / "pdf.py"
        spec = importlib.util.spec_from_file_location("pdf_model_routing_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PdfMixin
    finally:
        if previous_shared is None:
            sys.modules.pop("shared", None)
        else:
            sys.modules["shared"] = previous_shared


PdfMixin = _load_pdf_mixin()


class _Harness(PdfMixin):
    def __init__(self, config=None):
        self.config = dict(config or {})
        self._pdf_provider_preflight_cache = {}
        self.context = None

    def _cfg(self, key, default=None):
        return self.config.get(key, default)

    async def _pdf_preflight_provider(self, provider_id, purpose=""):
        return True, ""


class PdfModelRoutingTests(unittest.TestCase):
    def test_gpt_keyword_is_case_insensitive_and_matches_inside_text(self):
        for text in ("gpt 解答", "请用 GPT", "Gpt生成", "ChatGPT 来做"):
            self.assertTrue(_Harness._pdf_requests_gpt_model(text))
        self.assertFalse(_Harness._pdf_requests_gpt_model("请用默认模型解答"))

    def test_routed_model_is_first_and_each_slot_keeps_its_compact_switch(self):
        harness = _Harness({
            "pdf_provider_enable_compact_prompt_rules": False,
            "pdf_fallback_provider_id_1": "fallback-one",
            "pdf_fallback_provider_timeout_sec_1": 90,
            "pdf_fallback_provider_enable_compact_prompt_rules_1": True,
        })

        chain = harness._pdf_build_provider_chain(
            "main",
            "current",
            routed_provider_id="gpt-special",
            routed_timeout_sec=120,
            routed_compact_prompt_rules=False,
        )

        self.assertEqual(
            [item["provider_id"] for item in chain],
            ["gpt-special", "main", "fallback-one"],
        )
        self.assertEqual(
            [item["compact_prompt_rules"] for item in chain],
            [False, False, True],
        )

    def test_legacy_global_compact_switch_remains_compatible(self):
        harness = _Harness({"pdf_enable_compact_prompt_rules": False})
        self.assertFalse(
            harness._pdf_compact_enabled_for_slot(
                "pdf_provider_enable_compact_prompt_rules",
            )
        )

    def test_prompt_builder_is_evaluated_per_fallback_model(self):
        harness = _Harness()
        calls = []

        class _Context:
            async def llm_generate(self, chat_provider_id, prompt, image_urls):
                calls.append((chat_provider_id, prompt))
                text = "" if chat_provider_id == "first" else "ok"
                return types.SimpleNamespace(completion_text=text)

        harness.context = _Context()
        chain = [
            {"provider_id": "first", "timeout_sec": 0, "compact_prompt_rules": False},
            {"provider_id": "second", "timeout_sec": 0, "compact_prompt_rules": True},
        ]

        result = asyncio.run(harness._pdf_llm_generate_with_fallback(
            provider_chain=chain,
            prompt=lambda spec: "compact" if spec["compact_prompt_rules"] else "normal",
        ))

        self.assertEqual(result[0:2], ("ok", "second"))
        self.assertEqual(calls, [("first", "normal"), ("second", "compact")])

    def test_compact_rules_can_be_enabled_or_disabled_explicitly(self):
        harness = _Harness()
        compact = harness._build_pdf_latex_prompt("x=1", compact_prompt_rules=True)
        normal = harness._build_pdf_latex_prompt("x=1", compact_prompt_rules=False)

        self.assertIn("其余公式都使用行内公式", compact)
        self.assertNotIn("其余公式都使用行内公式", normal)

    def test_webui_exposes_compact_switch_for_every_pdf_model_slot(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        items = schema["pdf_generation"]["items"]

        self.assertNotIn("pdf_enable_compact_prompt_rules", items)
        self.assertEqual(items["pdf_gpt_provider_id"]["_special"], "select_provider")
        expected_switches = [
            "pdf_provider_enable_compact_prompt_rules",
            "pdf_gpt_provider_enable_compact_prompt_rules",
            *[
                f"pdf_fallback_provider_enable_compact_prompt_rules_{i}"
                for i in range(1, 6)
            ],
        ]
        for key in expected_switches:
            self.assertIn(key, items)
            self.assertEqual(items[key]["type"], "bool")


if __name__ == "__main__":
    unittest.main()
