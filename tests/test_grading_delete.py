"""Deletion must remove derived data without renumbering or crossing accounts."""
import asyncio
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from test_grading import Event, Harness, fixture, pdf_bytes
from grading import parse_command
from grading_document import GradingError
from grading_store import GradingStore, owner_key


class DeleteStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = GradingStore(self.temp.name)
        self.owner = owner_key('aiocqhttp', 'alice')
    def tearDown(self):
        self.temp.cleanup()
    def save(self, number, revision=1, owner=None):
        data = fixture()
        data['title'] = f'Unique question {number}'
        return self.store.publish(owner or self.owner, number, revision, data,
                                  pdf_bytes(f'Question {number} revision {revision}'), f'{number}-{revision}')

    def test_delete_middle_rebuilds_pdf_tex_and_removes_all_revisions(self):
        self.save(1); self.save(2); self.save(2, 2); third = self.save(3)
        other = owner_key('aiocqhttp', 'bob')
        other_row = self.save(1, owner=other)
        other_book = self.store.export(other).read_bytes()
        self.store.save_vector(self.owner, 'model', 'stale', [1])
        self.store.save_vector(other, 'model', 'other', [2])
        result = self.store.delete(self.owner, 2)
        self.assertEqual(result, {'duplicate': False, 'remaining': 2})
        self.assertEqual([r['number'] for r in self.store.rows(self.owner)], [1, 3])
        self.assertEqual(self.store.get(self.owner, 3), third)
        for method in [self.store.get, self.store.history]:
            with self.assertRaises(GradingError): method(self.owner, 2)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM revisions WHERE owner=? AND number=2', (self.owner,)).fetchone()[0], 0)
        self.assertFalse((self.store._user(self.owner) / 'entries/000002').exists())
        self.assertEqual(len(list((self.store._user(self.owner) / 'books').iterdir())), 1)
        self.assertFalse(self.store.vectors(self.owner, 'model'))
        self.assertEqual(self.store.vectors(other, 'model'), {'other': [2]})
        self.assertEqual(self.store.get(other, 1), other_row)
        self.assertEqual(self.store.export(other).read_bytes(), other_book)
        import fitz
        with fitz.open(self.store.export(self.owner)) as pdf:
            self.assertEqual(len(pdf), 2)
            self.assertEqual([item[2] for item in pdf.get_toc()], [1, 2])
            self.assertIn('Unique question 3', pdf.get_toc()[1][1])
            self.assertNotIn('Question 2', ''.join(page.get_text() for page in pdf))
        tex = self.store.export(self.owner).with_suffix('.tex').read_text()
        self.assertIn('Unique question 3', tex)
        self.assertNotIn('Unique question 2', tex)

    def test_last_and_highest_numbers_are_never_reused_after_reload(self):
        self.save(1); self.save(2)
        self.store.delete(self.owner, 2)
        self.store = GradingStore(self.temp.name)
        self.assertEqual(self.store.next_number(self.owner), 3)
        self.store.delete(self.owner, 1)
        with self.assertRaises(GradingError): self.store.export(self.owner)
        self.assertFalse(list((self.store._user(self.owner) / 'books').iterdir()))
        self.store = GradingStore(self.temp.name)
        self.assertEqual(self.store.next_number(self.owner), 3)
        self.save(3)
        self.assertEqual([r['number'] for r in self.store.rows(self.owner)], [3])

    def test_old_database_migrates_number_counter_without_changing_records(self):
        self.save(1); self.save(2)
        before = self.store.rows(self.owner)
        with self.store.connect() as db: db.execute('DROP TABLE sequences')
        migrated = GradingStore(self.temp.name)
        self.assertEqual(migrated.rows(self.owner), before)
        migrated.delete(self.owner, 2)
        self.assertEqual(migrated.next_number(self.owner), 3)

    def test_missing_or_other_user_number_cannot_delete_anything(self):
        self.save(1)
        before = self.store.rows(self.owner)
        for owner, number in [(self.owner, 99), (owner_key('aiocqhttp', 'bob'), 1), (self.owner, True)]:
            with self.assertRaises(GradingError): self.store.delete(owner, number)
        self.assertEqual(self.store.rows(self.owner), before)

    def test_repeat_delete_is_idempotent_and_preserves_submit_receipt(self):
        self.save(1)
        self.store.delete(self.owner, 1)
        version = self.store.version(self.owner)
        self.assertTrue(self.store.delete(self.owner, 1)['duplicate'])
        self.assertEqual(self.store.version(self.owner), version)
        self.assertEqual(self.store.request(self.owner, '1-1')['number'], 1)

    def test_failed_rebuild_leaves_record_history_vectors_and_book_intact(self):
        self.save(1); self.save(2)
        before = self.store.rows(self.owner)
        book = self.store.export(self.owner)
        self.store.save_vector(self.owner, 'model', 'old', [1])
        with patch.object(self.store, '_build_book', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.store.delete(self.owner, 1)
        self.assertEqual(self.store.rows(self.owner), before)
        self.assertEqual(self.store.export(self.owner), book)
        self.assertEqual(len(self.store.history(self.owner, 1)), 1)
        self.assertEqual(self.store.vectors(self.owner, 'model'), {'old': [1]})
        self.assertEqual(len(list(book.parent.parent.iterdir())), 2)

    def test_cleanup_failure_is_retried_after_reload(self):
        row = self.save(1)
        with patch('grading_store.shutil.rmtree', side_effect=OSError('busy')):
            self.store.delete(self.owner, 1)
        self.assertFalse(self.store.rows(self.owner))
        self.assertTrue(self.store.artifact(row['artifact_dir']).exists())
        GradingStore(self.temp.name)
        self.assertFalse(self.store.artifact(row['artifact_dir']).exists())
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM artifact_cleanup').fetchone()[0], 0)

    def test_stale_embedding_write_after_delete_is_discarded(self):
        self.save(1)
        version, _ = self.store.version(self.owner)
        self.store.delete(self.owner, 1)
        self.assertFalse(self.store.save_vector(self.owner, 'model', 'old', [1], version))
        self.assertEqual(self.store.vectors(self.owner, 'model'), {})

    def test_parser_requires_exact_single_positive_number(self):
        self.assertEqual(parse_command('/批改 删除\t2'), ('delete', 2, ''))
        for text in ['/批改 删除', '/批改 删除 0', '/批改 删除 -1', '/批改 删除 1 2', '/批改 删除 全部']:
            with self.assertRaises(GradingError): parse_command(text)


class DeleteWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.plugin = Harness(self.temp.name)
        self.owner = self.plugin._grading_owner(Event())
    async def asyncTearDown(self): self.temp.cleanup()

    async def test_command_deletes_without_model_call_and_keeps_next_number(self):
        event = Event()
        await self.plugin._grading_mutate(event, 'submit', None, 'first')
        calls = self.plugin.calls
        result = await self.plugin._grading_run(Event(group='other'), 'delete', 1)
        self.assertTrue(result['deleted'])
        self.assertEqual(self.plugin.calls, calls)
        self.assertEqual(self.plugin._grading_stats(self.owner)['total_entries'], 0)
        self.assertFalse(self.plugin._grading_chunks(self.owner))
        with self.assertRaises(GradingError): await self.plugin._grading_mutate(event, 'submit', None, 'replay')
        row, _ = await self.plugin._grading_mutate(Event(mid='2'), 'submit', None, 'new')
        self.assertEqual(row['number'], 2)

    async def test_index_in_flight_cannot_recreate_deleted_vectors(self):
        await self.plugin._grading_mutate(Event(), 'submit', None, 'first')
        started, release = asyncio.Event(), asyncio.Event()
        async def embed(text):
            started.set(); await release.wait(); return [1, 2]
        task = asyncio.create_task(self.plugin._grading_index_owner(self.owner, types.SimpleNamespace(get_embedding=embed), 'model'))
        await started.wait()
        await self.plugin._grading_run(Event(), 'delete', 1)
        release.set(); await task
        self.assertFalse(self.plugin._grading_store.vectors(self.owner, 'model'))

    async def test_search_in_flight_refetches_records_after_delete(self):
        await self.plugin._grading_mutate(Event(), 'submit', None, 'first')
        await self.plugin._grading_mutate(Event(mid='2'), 'submit', None, 'second')
        started, release = asyncio.Event(), asyncio.Event()
        async def embed(text):
            if text == '平方根': started.set(); await release.wait()
            return [1, 2]
        self.plugin.cfg['grading_embedding_provider_id'] = 'test'
        self.plugin.context.get_provider_by_id = lambda _: types.SimpleNamespace(get_embedding=embed)
        task = asyncio.create_task(self.plugin._grading_search(self.owner, '平方根'))
        await started.wait()
        await self.plugin._grading_run(Event(), 'delete', 1)
        release.set(); found = await task
        self.assertEqual(found['total_entries'], 1)
        self.assertEqual({item['number'] for item in found['matches']}, {2})

    async def test_cancelled_delete_waits_for_commit_before_releasing_owner_lock(self):
        await self.plugin._grading_mutate(Event(), 'submit', None, 'first')
        started, release = threading.Event(), threading.Event()
        original = self.plugin._grading_store.delete
        def blocked(*args):
            started.set()
            if not release.wait(5): raise RuntimeError('test release timed out')
            return original(*args)
        with patch.object(self.plugin._grading_store, 'delete', blocked):
            task = asyncio.create_task(self.plugin._grading_run(Event(), 'delete', 1))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            await asyncio.sleep(0)
            self.assertTrue(self.plugin._grading_lock(self.owner).locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(self.plugin._grading_lock(self.owner).locked())
        self.assertFalse(self.plugin._grading_store.rows(self.owner))
        self.assertEqual(self.plugin._grading_store.next_number(self.owner), 2)
