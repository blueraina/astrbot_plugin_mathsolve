import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading import GradingMixin, parse_command, score_requested, finish_io
from grading_document import GradingError, parse_record, validate_record, document, entry_body, record_chunks
from grading_store import GradingStore, owner_key


def fixture():
    return {"complete": True, "title": "平方根与遗漏条件", "problem_tex": r"解方程 $x^2=4$。另问 $\sqrt{x^2}=x$ 是否恒成立？",
            "steps": [{"work_tex": r"$x=2$。", "comments": [{"kind": "error", "tex": r"遗漏负根 $x=-2$。", "tags": ["平方根", "漏解"]}]},
                      {"work_tex": r"$\sqrt{x^2}=x$ 对所有实数成立。", "comments": [{"kind": "gap", "tex": r"缺少条件 $x\geq 0$；一般应为 $|x|$。", "tags": ["绝对值"]}]},
                      {"work_tex": "第三小问未作答。", "comments": [{"kind": "unattempted", "tex": "未提交作答，不计为错误。", "tags": []}]}],
            "summary_tex": "第一问漏解，第二问缺少非负条件；未作答项单独记录。",
            "reference_tex": r"$x=\pm2$。对所有实数，$\sqrt{x^2}=|x|$。",
            "knowledge_points": ["平方根", "绝对值"], "score": None}


def pdf_bytes(text="test"):
    import fitz
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 60), text)
        return doc.tobytes()


class Event:
    def __init__(self, user="alice", group="group1", mid="1", text="/批改 题目"):
        self.user, self.group, self.message_str = user, group, text
        self.unified_msg_origin = f"bot:GroupMessage:{group}"
        self.message_obj = types.SimpleNamespace(message_id=mid, message=[])
        self.extra = {}
    def get_platform_name(self): return "aiocqhttp"
    def get_sender_id(self): return self.user
    def get_extra(self, key): return self.extra.get(key)
    def set_extra(self, key, value): self.extra[key] = value


class Harness(GradingMixin):
    def __init__(self, root):
        self.DATA_DIR = root
        self.cfg = {}
        self.context = types.SimpleNamespace(get_provider_by_id=lambda selected: None)
        self.calls = 0
        self.fail = False
        self._grading_init()
    def _cfg(self, key, default=None): return self.cfg.get(key, default)
    async def _grading_inputs(self, event, text): return text, []
    async def _pdf_get_event_image_inputs(self, event): return []
    async def _grading_generate(self, event, number, revision, text, images, old=None):
        self.calls += 1
        await asyncio.sleep(0.005)
        if self.fail: raise GradingError("simulated model failure")
        result = fixture()
        if old:
            result["steps"][0]["work_tex"] = r"$x=\pm2$。"
            result["steps"][0]["comments"] = []
        return result, pdf_bytes(f"Question {number} revision {revision}")


class DocumentTests(unittest.TestCase):
    def test_complete_json_and_red_inline_comments(self):
        record = parse_record(json.dumps(fixture()))
        tex = document(entry_body(1, 1, record))
        self.assertIn(r"\color{gradingred}", tex)
        self.assertLess(tex.index(r"$x=2$"), tex.index("遗漏负根"))
        self.assertIn("AI 补充", tex)
        self.assertNotIn("按要求评分", tex)
    def test_malformed_truncated_or_incomplete_output_rejected(self):
        for raw in ['{"complete":true', '{}', '[]']:
            with self.assertRaises(GradingError): parse_record(raw)
    def test_filesystem_macro_and_image_payload_rejected(self):
        for content in [r"\input{/etc/passwd}", r"\csname input\endcsname", r"\def\x{x}",
                        r"\begin{document}", r"^^5cinput", "data:image/png;base64,abc", "50%"]:
            data = fixture();data["problem_tex"] = content
            with self.assertRaises(GradingError, msg=content):validate_record(data)
    def test_scoring_only_on_request(self):
        data = fixture();data["score"] = "6/10"
        self.assertIsNone(validate_record(data)["score"])
        self.assertEqual(validate_record(data, True)["score"], "6/10")
        for text in ["请给我评分", "打个分", "这份作答得多少分"]:self.assertTrue(score_requested(text))
        for text in ["不用评分", "不要打分", "求积分", "几分之一"]:self.assertFalse(score_requested(text))
    def test_numeric_scores_from_provider_are_supported_only_when_requested(self):
        for value in [0, 12, 12.5, 15]:
            data = fixture();data['score'] = value
            self.assertIsNone(validate_record(data)['score'])
            self.assertEqual(validate_record(data, True)['score'], str(value))
        for value in [True, float('nan'), float('inf'), [], {}]:
            data = fixture();data['score'] = value
            with self.assertRaises(GradingError):validate_record(data, True)
        self.assertTrue(score_requested('只批改第二问满分15'))
    def test_double_escaped_newlines_preserve_math_and_safety_checks(self):
        data = fixture()
        data['problem_tex'] = r'第一段\n第二段 $\nabla f\neq 0$，$\nu=1$。\\n'
        record = parse_record(json.dumps(data))
        self.assertEqual(record['problem_tex'], '第一段\n' + r'第二段 $\nabla f\neq 0$，$\nu=1$。\\n')
        data['problem_tex'] = r'第一段\n\input{/etc/passwd}'
        with self.assertRaises(GradingError):validate_record(data)
    def test_standard_proof_end_symbols_are_accepted(self):
        data = fixture()
        data['reference_tex'] = r'证明完毕。$\square$ 或 $\blacksquare$'
        self.assertEqual(validate_record(data)['reference_tex'], data['reference_tex'])
    def test_large_algebra_operators_and_literal_hash_are_preserved(self):
        data = fixture()
        data['reference_tex'] = r'题#1，已转义\#2，换行\\#3；$\bigoplus_{i=1}^n V_i$ 与 $\bigotimes_i W_i$。'
        result = validate_record(data)['reference_tex']
        self.assertEqual(result, r'题\#1，已转义\#2，换行\\\#3；$\bigoplus_{i=1}^n V_i$ 与 $\bigotimes_i W_i$。')
        data['reference_tex'] = r'#\input{/etc/passwd}'
        with self.assertRaises(GradingError): validate_record(data)
    def test_only_complete_bare_math_environments_are_wrapped(self):
        from grading_document import tex_fragment
        bare = r'\begin{aligned} A&=\begin{pmatrix}1&0\\0&1\end{pmatrix}\\B&=0\end{aligned}'
        self.assertEqual(tex_fragment('证明：'+bare+'。'), '证明：'+r'\['+bare+r'\]'+'。')
        for wrapped in [r'\['+bare+r'\]', '$'+bare+'$', '$$'+bare+'$$', r'\('+bare+r'\)',
                        r'\begin{equation}'+bare+r'\end{equation}']:
            self.assertEqual(tex_fragment(wrapped),wrapped)
        multiple = bare+'文字'+r'\begin{cases}x&x>0\\-x&x<0\end{cases}'
        self.assertEqual(tex_fragment(multiple).count(r'\['),2)
        malformed = r'\begin{aligned}x&=1'
        self.assertEqual(tex_fragment(malformed),malformed)
    def test_missing_question_is_rejected_but_unattempted_answer_is_valid(self):
        with self.assertRaises(GradingError):validate_record({'complete':False,'input_available':False})
        data=fixture();data['problem_tex']='[未提供题目]'
        with self.assertRaises(GradingError):validate_record(data)
        data=fixture();data['steps']=[{'work_tex':'未作答','comments':[{'kind':'unattempted','tex':'未提交作答，不计为答错','tags':[]}]}]
        self.assertEqual(validate_record(data)['steps'][0]['comments'][0]['kind'],'unattempted')
    def test_command_boundaries(self):
        self.assertEqual(parse_command("/批改 修改\t12\n修正第二步"), ("modify", 12, "修正第二步"))
        self.assertEqual(parse_command("批改 题1及题2"), ("submit", None, "题1及题2"))
        for text in ["/批改 搜索 -1", "/批改 修改 1", "/批改 搜索 1 多余", "/批改导出"]:
            with self.assertRaises(GradingError):parse_command(text)
    def test_separate_original_and_current_chunks(self):
        original, current = fixture(), fixture()
        current["steps"][0]["work_tex"] = "订正后的作答"
        chunks = record_chunks(7, 2, original, current)
        self.assertEqual({c["kind"] for c in chunks}, {"首次作答", "当前版本"})
        self.assertTrue(all(c["number"] == 7 for c in chunks))
        self.assertTrue(any("$x=2$" in c["text"] for c in chunks if c["kind"] == "首次作答"))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = GradingStore(self.temp.name)
        self.owner = owner_key("aiocqhttp", "alice")
    def tearDown(self):self.temp.cleanup()
    def save(self, number, revision=1, data=None):
        return self.store.publish(self.owner, number, revision, data or fixture(), pdf_bytes(), f"request-{number}-{revision}")
    def test_numbering_one_submission_one_record_and_persistence(self):
        self.save(1);self.save(2)
        loaded = GradingStore(self.temp.name)
        self.assertEqual([r["number"] for r in loaded.rows(self.owner)], [1, 2])
        self.assertIn("另问", loaded.get(self.owner, 1)["record"]["problem_tex"])
    def test_cross_user_denied(self):
        self.save(1)
        with self.assertRaises(GradingError):self.store.get(owner_key("aiocqhttp", "bob"), 1)
        self.assertEqual(self.store.rows(owner_key("aiocqhttp", "bob")), [])
        with self.assertRaises(GradingError):owner_key("aiocqhttp", "unknown")
    def test_modify_preserves_original_history_and_other_question(self):
        first = self.save(1);second = self.save(2)
        other_bytes = (self.store.artifact(second["artifact_dir"]) / "answer.pdf").read_bytes()
        revised = fixture();revised["steps"][0]["work_tex"] = "订正作答"
        new = self.save(1, 2, revised)
        self.assertEqual(new["original"], first["record"])
        self.assertEqual(len(self.store.history(self.owner, 1)), 2)
        self.assertEqual(self.store.get(self.owner, 2), second)
        self.assertEqual((self.store.artifact(second["artifact_dir"]) / "answer.pdf").read_bytes(), other_bytes)
        import fitz
        with fitz.open(self.store.export(self.owner)) as pdf:
            self.assertEqual(len(pdf), 2)
            self.assertEqual(len(pdf.get_toc()), 2)
        tex = self.store.export(self.owner).with_suffix('.tex').read_text()
        self.assertIn("订正作答", tex)
        self.assertIn("第 2 题", tex)
    def test_failed_pdf_leaves_previous_book_and_number_intact(self):
        self.save(1)
        before = self.store.export(self.owner).read_bytes()
        with self.assertRaises(Exception):self.store.publish(self.owner, 2, 1, fixture(), b"not a PDF")
        self.assertEqual(len(self.store.rows(self.owner)), 1)
        self.assertEqual(self.store.export(self.owner).read_bytes(), before)
        self.save(2)
    def test_no_original_image_persisted(self):
        self.save(1)
        self.assertFalse(any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in Path(self.temp.name).rglob('*')))
        for p in Path(self.temp.name).rglob('*.tex'):self.assertNotIn("base64", p.read_text())
    def test_request_dedup_and_stale_revision(self):
        self.save(1)
        self.assertEqual(self.store.request(self.owner, "request-1-1")["number"], 1)
        with self.assertRaises(GradingError):self.save(1)
        self.assertEqual(len(self.store.history(self.owner, 1)), 1)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory();self.plugin = Harness(self.temp.name)
    async def asyncTearDown(self):self.temp.cleanup()
    async def test_same_user_across_groups_and_concurrent_numbering(self):
        first, second = await asyncio.gather(
            self.plugin._grading_mutate(Event(group="a",mid="1"), "submit", None, "题1及题2"),
            self.plugin._grading_mutate(Event(group="b",mid="2"), "submit", None, "另一题"))
        self.assertEqual({first[0]["number"], second[0]["number"]}, {1, 2})
        self.assertEqual(self.plugin._grading_owner(Event(group="a")), self.plugin._grading_owner(Event(group="b")))
    async def test_repeated_agent_call_does_not_duplicate(self):
        event = Event()
        first, _ = await self.plugin._grading_mutate(event,"submit",None,"原话")
        second, duplicate = await self.plugin._grading_mutate(event,"submit",None,"模型重新措辞")
        self.assertTrue(duplicate);self.assertEqual(first["number"], second["number"])
        self.assertEqual(self.plugin.calls, 1)
    async def test_failure_does_not_consume_number(self):
        self.plugin.fail = True
        with self.assertRaises(GradingError):await self.plugin._grading_mutate(Event(),"submit",None,"first")
        self.plugin.fail = False
        row,_ = await self.plugin._grading_mutate(Event(mid="2"),"submit",None,"second")
        self.assertEqual(row["number"], 1)
    async def test_knowledge_base_and_statistics_are_per_user(self):
        await self.plugin._grading_mutate(Event(),"submit",None,"first")
        owner = self.plugin._grading_owner(Event())
        result = await self.plugin._grading_search(owner,"平方根")
        self.assertTrue(result["matches"])
        with self.assertRaises(GradingError):await self.plugin._grading_search(self.plugin._grading_owner(Event(user="bob")),"平方根")
        stats = self.plugin._grading_stats(owner)
        self.assertEqual(stats["comment_counts"]["首次作答"]["error"], 1)
        self.assertEqual(stats["comment_counts"]["首次作答"]["unattempted"], 1)
    async def test_embedding_isolation_model_switch_and_failure_fallback(self):
        await self.plugin._grading_mutate(Event(),"submit",None,"first")
        owner = self.plugin._grading_owner(Event())
        class Provider:
            provider_config = {"model":"v1"}
            fail = False
            def get_dim(self):return 3
            async def get_embedding(self, text):
                if self.fail:raise RuntimeError("offline")
                return [1.0, float("平方根" in text), 0.5]
        provider = Provider()
        self.plugin.context.get_provider_by_id = lambda selected:provider
        self.plugin.cfg["grading_embedding_provider_id"] = "test"
        _, first_key = await self.plugin._grading_embedding()
        result = await self.plugin._grading_search(owner,"平方根")
        self.assertIn("语义", result["mode"])
        self.assertTrue(self.plugin._grading_store.vectors(owner,first_key))
        self.assertFalse(self.plugin._grading_store.vectors(owner_key("aiocqhttp","bob"),first_key))
        provider.provider_config = {"model":"v2"}
        _, second_key = await self.plugin._grading_embedding()
        self.assertNotEqual(first_key,second_key)
        self.assertFalse(self.plugin._grading_store.vectors(owner,second_key))
        provider.fail = True
        result = await self.plugin._grading_search(owner,"平方根")
        self.assertEqual(result["mode"],"关键词检索")
        self.assertTrue(result["matches"])
    async def test_cancelled_mutation_never_writes_partial_record(self):
        started = asyncio.Event()
        async def blocking(*args,**kwargs):
            started.set();await asyncio.sleep(30)
        self.plugin._grading_generate = blocking
        task = asyncio.create_task(self.plugin._grading_mutate(Event(),"submit",None,"question"))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(self.plugin._grading_store.rows(self.plugin._grading_owner(Event())), [])


class ModelProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_label_is_trusted_pinned_and_preserved_per_revision(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = Harness(root)
            plugin.cfg['grading_provider_id'] = 'provider-a'
            selected = []
            providers = {
                'provider-a': types.SimpleNamespace(get_model=lambda: 'model_A'),
                'provider-b': types.SimpleNamespace(get_model=lambda: 'model_B'),
            }
            async def generate(**kwargs):
                selected.append(kwargs['chat_provider_id'])
                plugin.cfg['grading_provider_id'] = 'provider-b'
                result = fixture()
                result['grading_model'] = 'spoofed-model'
                result['grading_provider_id'] = 'spoofed-provider'
                return types.SimpleNamespace(completion_text=json.dumps(result))
            async def compile_tex(tex):
                self.assertNotIn('spoofed', tex)
                return pdf_bytes()
            plugin.context = types.SimpleNamespace(get_provider_by_id=providers.get, llm_generate=generate)
            plugin._compile_local_xelatex = compile_tex
            first, pdf = await GradingMixin._grading_generate(plugin, Event(), 1, 1, '批改', [])
            self.assertEqual((first['grading_provider_id'], first['grading_model']), ('provider-a', 'model_A'))
            owner = plugin._grading_owner(Event())
            old = plugin._grading_store.publish(owner, 1, 1, first, pdf, 'first')
            second, pdf = await GradingMixin._grading_generate(plugin, Event(), 1, 2, '修改', [], old)
            plugin._grading_store.publish(owner, 1, 2, second, pdf, 'second')
            restored = GradingStore(Path(root) / 'grading_portfolio').get(owner, 1)
            self.assertEqual(selected, ['provider-a', 'provider-b'])
            self.assertEqual(restored['original']['grading_model'], 'model_A')
            self.assertEqual(restored['record']['grading_model'], 'model_B')
            tex = plugin._grading_store.export(owner).with_suffix('.tex').read_text()
            self.assertIn(r'批改模型：model\_B', tex)
            self.assertNotIn(r'批改模型：model\_A', tex)

    async def test_missing_metadata_is_not_replaced_by_current_configuration(self):
        tex = entry_body(1, 1, fixture())
        self.assertIn('批改模型：未记录', tex)
        record = fixture()
        record['grading_model'] = r'model_A%&'
        self.assertIn(r'批改模型：model\_A\%\&', entry_body(1, 1, record))


if __name__ == '__main__':unittest.main()
