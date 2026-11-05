"""Tests for the priority queue, cache review serialization and reset rules."""

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

if "core" not in sys.modules:
    core = types.ModuleType("core")

    class _QuotaExhausted(RuntimeError):
        def __init__(self, message, *, retry_after=None):
            super().__init__(message)
            self.retry_after = retry_after

    class _ModelUnavailable(RuntimeError):
        def __init__(self, model, *, retry_after=None):
            super().__init__(model)
            self.model = model
            self.retry_after = retry_after

    core._QuotaExhausted = _QuotaExhausted
    core._ModelUnavailable = _ModelUnavailable
    core.apply_opencc = lambda text, lang, zh_format: text
    core.transcribe = lambda *args, **kwargs: ""
    core.refine = lambda *args, **kwargs: ""
    core.warm_up = lambda: None
    sys.modules["core"] = core

import config  # noqa: E402
import core  # noqa: E402
import state  # noqa: E402
from cache import CacheStore, sha256_text  # noqa: E402
from jobs import JobManager  # noqa: E402
from models import JobRecord, TranscriptionJob, UserSession  # noqa: E402


class FakeBot:
    def __init__(self):
        self.messages = []
        self.documents = []

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        self.messages.append((chat_id, text, reply_markup))
        return types.SimpleNamespace(message_id=len(self.messages))

    async def send_document(self, chat_id, document, **kwargs):
        self.documents.append((chat_id, document))
        return None


def make_job(job_id, priority, sequence, user_id=1):
    return TranscriptionJob(
        job_id=job_id,
        user_id=user_id,
        chat_id=user_id,
        file_path=f"/tmp/{job_id}",
        file_name=f"{job_id}.mp3",
        md5="a" * 32,
        sha256_key=b"k" * 32,
        priority=priority,
        completion_seq=sequence,
        lang="English",
        fmt="Text",
        zh_format=None,
        en_prompt="",
        refine=False,
        c2w=False,
        char_limit=30,
    )


class PriorityQueueTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        config.UPLOAD_DIR = root / "uploads"
        config.UPLOAD_DIR.mkdir()
        config.RAW_CACHE_DIR = root / "raw"
        config.REFINE_CACHE_DIR = root / "refine"
        config.RUNTIME_STATE_FILE = root / "runtime.json"
        state.cache_store = CacheStore(
            config.RAW_CACHE_DIR, config.REFINE_CACHE_DIR
        )
        state.sessions.clear()
        state.jobs_by_id.clear()
        state.sessions[1] = UserSession(user_id=1, chat_id=1)
        self.session = state.sessions[1]

    def tearDown(self):
        state.sessions.clear()
        state.jobs_by_id.clear()
        self.tempdir.cleanup()

    def test_lowest_priority_runs_first_and_ties_use_completion_order(self):
        jobs = [
            make_job("a", priority=2, sequence=1),
            make_job("b", priority=1, sequence=2),
            make_job("c", priority=1, sequence=3),
        ]
        for job in jobs:
            state.jobs_by_id[job.job_id] = JobRecord(job=job, status="queued")
        self.session.active_count = len(jobs)

        async def run():
            manager = JobManager(FakeBot())
            order = []
            done = asyncio.Event()

            async def fake_process(record):
                order.append(record.job.job_id)
                state.remove_job(record.job.job_id)
                state.complete_active(record)
                if len(order) == len(jobs):
                    done.set()

            manager._process_transcription = fake_process
            for job in jobs:
                await manager.enqueue_transcription(job)
            await manager.start()
            try:
                await asyncio.wait_for(done.wait(), timeout=2)
            finally:
                await manager.stop()
            return order

        self.assertEqual(asyncio.run(run()), ["b", "c", "a"])

    def test_priorities_are_per_user_and_start_at_one(self):
        self.assertEqual(
            state.assign_batch_priorities(self.session, 5), [1, 2, 3, 4, 5]
        )
        other = UserSession(user_id=2, chat_id=2)
        state.sessions[2] = other
        self.assertEqual(state.assign_batch_priorities(other, 1), [1])

    def test_priority_resets_only_when_all_active_work_finishes(self):
        priorities = state.assign_batch_priorities(self.session, 2)
        self.assertEqual(priorities, [1, 2])
        self.assertEqual(self.session.next_priority, 3)
        self.assertEqual(self.session.active_count, 2)

        records = []
        for index, priority in enumerate(priorities):
            job = make_job(f"j{index}", priority, index + 1)
            record = JobRecord(job=job, status="processing")
            state.jobs_by_id[job.job_id] = record
            records.append(record)

        state.complete_active(records[0])
        self.assertEqual(self.session.active_count, 1)
        self.assertEqual(self.session.next_priority, 3)

        state.complete_active(records[1])
        self.assertEqual(self.session.active_count, 0)
        self.assertEqual(self.session.next_priority, 1)

        state.complete_active(records[1])
        self.assertEqual(self.session.active_count, 0)

        self.assertEqual(
            state.assign_batch_priorities(self.session, 1), [1]
        )

    def test_cache_reviews_are_sent_one_at_a_time(self):
        bot = FakeBot()
        records = []
        for index, status in enumerate(("raw_cache", "raw_cache", "refine_cache")):
            job = make_job(f"r{index}", priority=index + 1, sequence=index + 1)
            record = JobRecord(job=job, status=status)
            state.jobs_by_id[job.job_id] = record
            records.append(record)

        async def run():
            manager = JobManager(bot)
            await manager.enqueue_review(1, records[0].job.job_id, "raw")
            await manager.enqueue_review(1, records[1].job.job_id, "raw")
            self.assertEqual(len(bot.messages), 1)
            self.assertTrue(
                manager.is_active_review(1, records[0].job.job_id)
            )
            self.assertTrue(
                manager.begin_review(1, records[0].job.job_id)
            )
            self.assertFalse(
                manager.begin_review(1, records[0].job.job_id)
            )

            await manager.enqueue_review(1, records[2].job.job_id, "refine")
            self.assertEqual(len(bot.messages), 1)

            await manager.finish_review(1)
            self.assertEqual(len(bot.messages), 2)
            self.assertTrue(
                manager.is_active_review(1, records[1].job.job_id)
            )

            self.assertTrue(manager.begin_review(1, records[1].job.job_id))
            await manager.finish_review(1)
            self.assertEqual(len(bot.messages), 3)
            self.assertTrue(
                manager.is_active_review(1, records[2].job.job_id)
            )

        asyncio.run(run())

        for _, text, _ in bot.messages:
            self.assertNotIn("(priority", text)

    def test_resume_requeues_processing_job_with_upload(self):
        bot = FakeBot()
        job = make_job("resume", priority=1, sequence=1)
        path = config.UPLOAD_DIR / "resume.mp3"
        path.write_bytes(b"audio")
        job.file_path = str(path)
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="processing", counted_active=True
        )
        self.session.active_count = 1

        async def run():
            manager = JobManager(bot)
            await manager.resume()
            return manager

        manager = asyncio.run(run())

        self.assertEqual(state.jobs_by_id["resume"].status, "queued")
        self.assertEqual(manager.transcription_queue.qsize(), 1)
        self.assertTrue(
            any("re-queuing" in message[1] for message in bot.messages)
        )

    def test_resume_drops_job_when_file_and_cache_are_missing(self):
        bot = FakeBot()
        job = make_job("lost", priority=1, sequence=1)
        job.file_path = str(config.UPLOAD_DIR / "missing.mp3")
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="processing", counted_active=True
        )
        self.session.active_count = 1

        async def run():
            manager = JobManager(bot)
            await manager.resume()

        asyncio.run(run())

        self.assertNotIn("lost", state.jobs_by_id)
        self.assertEqual(self.session.active_count, 0)
        self.assertTrue(
            any("no longer available" in message[1] for message in bot.messages)
        )

    def test_resume_rebuilds_raw_cache_review(self):
        bot = FakeBot()
        job = make_job("cached", priority=1, sequence=1)
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "cached text"
        )
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="raw_cache", counted_active=False
        )

        async def run():
            manager = JobManager(bot)
            await manager.resume()
            return manager

        manager = asyncio.run(run())

        self.assertEqual(
            state.jobs_by_id["cached"].raw_text_pre, "cached text"
        )
        self.assertTrue(manager.is_active_review(1, "cached"))
        self.assertTrue(
            any("Existing transcription" in message[1] for message in bot.messages)
        )

    def test_resume_delivers_completed_refinement_from_cache(self):
        bot = FakeBot()
        job = make_job("refined", priority=1, sequence=1)
        job.refine = True
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "raw text"
        )
        state.cache_store.save_refinement(
            "raw text",
            lang=job.lang,
            fmt=job.fmt,
            zh_format=job.zh_format,
            cantonese_to_written=job.c2w,
            srt_char_limit=None,
            refined_text="refined text",
        )
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="refining", counted_active=False
        )

        async def run():
            manager = JobManager(bot)
            await manager.resume()

        asyncio.run(run())

        self.assertNotIn("refined", state.jobs_by_id)
        self.assertEqual(len(bot.documents), 1)


    def test_transcription_progress_names_the_file(self):
        bot = FakeBot()
        job = make_job("voice-note", priority=1, sequence=1)
        path = config.UPLOAD_DIR / "voice-note.mp3"
        path.write_bytes(b"audio")
        job.file_path = str(path)
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "transcribe", return_value="raw text"):
                await manager._process_transcription(record)

        asyncio.run(run())

        progress = [text for _, text, _ in bot.messages]
        self.assertIn("Transcribing voice-note.mp3...", progress)

    def test_transcription_completion_is_logged(self):
        bot = FakeBot()
        job = make_job("voice-note", priority=1, sequence=1)
        path = config.UPLOAD_DIR / "voice-note.mp3"
        path.write_bytes(b"audio")
        job.file_path = str(path)
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "transcribe", return_value="hello world"):
                await manager._process_transcription(record)

        with self.assertLogs("jobs", level="INFO") as logs:
            asyncio.run(run())

        completed = "\n".join(logs.output)
        self.assertIn("Transcription job started: job_id=voice-note", completed)
        self.assertIn("Transcription job voice-note completed", completed)
        self.assertIn("file=voice-note.mp3", completed)
        self.assertIn("elapsed=", completed)
        self.assertIn("chars=11", completed)

    def test_text_upload_is_read_and_refined_without_transcribing(self):
        bot = FakeBot()
        job = make_job("text", priority=1, sequence=1)
        job.source = "text"
        job.refine = True
        job.file_name = "notes.txt"
        job.fmt = "Text"
        path = config.UPLOAD_DIR / "notes.txt"
        path.write_text("hello\n", encoding="utf-8")
        job.file_path = str(path)
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record
        seen = []

        async def run():
            manager = JobManager(bot)

            async def fake_after(rec):
                seen.append((rec.raw_text_pre, rec.raw_text_display))

            manager.after_raw = fake_after
            with mock.patch.object(
                core, "transcribe", side_effect=AssertionError("no transcription")
            ):
                await manager._process_transcription(record)

        asyncio.run(run())

        self.assertEqual(seen, [("hello\n", "hello\n")])
        self.assertFalse(path.exists())
        self.assertEqual(bot.documents, [])
        cached = state.cache_store.load_raw(
            job.md5, job.lang, job.fmt, job.sha256_key
        )
        self.assertEqual(cached["text"], "hello\n")
        self.assertTrue(
            any("Reading notes.txt..." in message[1] for message in bot.messages)
        )

    def test_text_upload_uses_cached_text_when_file_is_missing(self):
        bot = FakeBot()
        job = make_job("cached-text", priority=1, sequence=1)
        job.source = "text"
        job.refine = True
        job.file_name = "notes.txt"
        job.fmt = "Text"
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "cached text"
        )
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record
        seen = []

        async def run():
            manager = JobManager(bot)

            async def fake_after(rec):
                seen.append((rec.raw_text_pre, rec.raw_text_display))

            manager.after_raw = fake_after
            await manager._process_transcription(record)

        asyncio.run(run())

        self.assertEqual(seen, [("cached text", "cached text")])
        self.assertFalse(
            any("Using the cached text" in message[1] for message in bot.messages)
        )

    def test_empty_text_upload_is_rejected(self):
        bot = FakeBot()
        job = make_job("empty-text", priority=1, sequence=1)
        job.source = "text"
        job.refine = True
        job.file_name = "notes.txt"
        job.fmt = "Text"
        path = config.UPLOAD_DIR / "notes.txt"
        path.write_text("   \n", encoding="utf-8")
        job.file_path = str(path)
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record
        self.session.active_count = 1

        async def run():
            manager = JobManager(bot)
            await manager._process_transcription(record)

        asyncio.run(run())

        self.assertNotIn(job.job_id, state.jobs_by_id)
        self.assertEqual(self.session.active_count, 0)
        self.assertFalse(path.exists())
        self.assertIn("is empty", bot.messages[-1][1])

    def test_invalid_srt_upload_is_rejected(self):
        bot = FakeBot()
        job = make_job("bad-srt", priority=1, sequence=1)
        job.source = "text"
        job.refine = True
        job.file_name = "subs.srt"
        job.fmt = "SRT"
        path = config.UPLOAD_DIR / "subs.srt"
        path.write_text("not srt", encoding="utf-8")
        job.file_path = str(path)
        record = JobRecord(job=job, status="processing", counted_active=True)
        state.jobs_by_id[job.job_id] = record

        async def run():
            manager = JobManager(bot)
            await manager._process_transcription(record)

        asyncio.run(run())

        self.assertNotIn(job.job_id, state.jobs_by_id)
        self.assertFalse(path.exists())
        self.assertIn("not valid SRT", bot.messages[-1][1])

    def test_refinement_progress_and_logs_name_the_job(self):
        bot = FakeBot()
        record = self._failed_refinement_record("clip")
        record.job.file_name = "clip.mp3"
        record.refinement.file_name = "clip.mp3"
        refine = mock.Mock(return_value="refined")

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "refine", refine):
                await manager.start_refinement(record, record.refinement)
                for task in list(manager._refine_tasks):
                    await task

        with self.assertLogs("jobs", level="INFO") as logs:
            asyncio.run(run())

        progress = [text for _, text, _ in bot.messages]
        self.assertIn("Refining clip.mp3...", progress)
        output = "\n".join(logs.output)
        self.assertIn("Refinement job started: job_id=clip", output)
        self.assertIn("Refinement job completed: job_id=clip", output)
        self.assertIn("chars=7", output)
        self.assertEqual(refine.call_args.kwargs["job_id"], "clip")
        self.assertIs(
            refine.call_args.kwargs["cache_store"], state.cache_store
        )

    def test_transcription_failure_names_the_file(self):
        bot = FakeBot()
        job = make_job("broken", priority=1, sequence=1)
        path = config.UPLOAD_DIR / "broken.mp3"
        path.write_bytes(b"audio")
        job.file_path = str(path)
        record = JobRecord(job=job, status="queued", counted_active=True)
        state.jobs_by_id[job.job_id] = record
        self.session.active_count = 1

        async def run():
            manager = JobManager(bot)
            await manager.enqueue_transcription(job)
            await manager.start()
            await manager.transcription_queue.join()
            await manager.stop()

        def boom(*args, **kwargs):
            raise RuntimeError("bad container")

        with mock.patch.object(core, "transcribe", side_effect=boom):
            asyncio.run(run())

        failed = [text for _, text, _ in bot.messages if "failed" in text]
        self.assertTrue(failed)
        self.assertIn("Transcription failed for broken.mp3", failed[-1])
        self.assertIn("bad container", failed[-1])

    def test_empty_raw_transcription_names_the_file(self):
        bot = FakeBot()
        job = make_job("silent", priority=1, sequence=1)
        job.refine = True
        record = JobRecord(job=job, status="processing", counted_active=True)
        record.raw_text_pre = "   "
        state.jobs_by_id[job.job_id] = record

        async def run():
            manager = JobManager(bot)
            await manager.after_raw(record)

        asyncio.run(run())

        messages = [text for _, text, _ in bot.messages]
        self.assertTrue(
            any(
                "Raw transcription for silent.mp3 is empty" in text
                for text in messages
            )
        )

    def test_after_raw_uses_converted_text_for_refinement_cache(self):
        bot = FakeBot()
        job = make_job("converted", priority=1, sequence=1)
        job.refine = True
        record = JobRecord(job=job, status="processing", counted_active=True)
        record.raw_text_pre = "raw text"
        record.raw_text_display = "converted text"
        state.jobs_by_id[job.job_id] = record
        state.cache_store.save_refinement(
            "converted text",
            lang=job.lang,
            fmt=job.fmt,
            zh_format=job.zh_format,
            cantonese_to_written=job.c2w,
            srt_char_limit=None,
            refined_text="refined text",
        )

        async def run():
            manager = JobManager(bot)
            await manager.after_raw(record)

        asyncio.run(run())

        self.assertEqual(state.jobs_by_id[job.job_id].status, "refine_cache")

    def _failed_refinement_record(self, job_id="retry"):
        """Builds a job that already has raw text cached and is refining."""
        from models import RefinementJob

        job = make_job(job_id, priority=1, sequence=1)
        job.refine = True
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "raw text"
        )
        record = JobRecord(job=job, status="refining", counted_active=True)
        refinement = RefinementJob(
            job_id=job.job_id,
            user_id=job.user_id,
            chat_id=job.chat_id,
            file_name=job.file_name,
            raw_text_pre="raw text",
            raw_text_display="raw text",
            lang=job.lang,
            fmt=job.fmt,
            zh_format=job.zh_format,
            c2w=job.c2w,
            char_limit=job.char_limit,
        )
        record.refinement = refinement
        state.jobs_by_id[job.job_id] = record
        return record

    def test_refinement_failure_keeps_job_and_offers_retry(self):
        bot = FakeBot()
        record = self._failed_refinement_record("boom")
        refinement = record.refinement

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "refine", side_effect=RuntimeError("quota")):
                await manager._run_refinement(refinement.job_id, refinement)

        asyncio.run(run())

        self.assertIn("boom", state.jobs_by_id)
        self.assertEqual(state.jobs_by_id["boom"].status, "refine_failed")
        self.assertIsNotNone(state.jobs_by_id["boom"].refinement)
        self.assertEqual(state.jobs_by_id["boom"].retry_message_id, 1)
        self.assertEqual(len(bot.messages), 1)
        _, text, markup = bot.messages[0]
        self.assertIn("AI refinement failed", text)
        self.assertIn("quota", text)
        self.assertIsNotNone(markup)
        button = markup.inline_keyboard[0][0]
        self.assertEqual(button.text, "Retry")
        self.assertEqual(button.callback_data, "tb:refine:retry:boom")
        self.assertEqual(bot.documents, [])

    def test_quota_failure_shows_reset_time_and_retry(self):
        bot = FakeBot()
        record = self._failed_refinement_record("quota")
        refinement = record.refinement
        error = core._QuotaExhausted(
            "Gemini quota is exhausted.", retry_after=3600
        )

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "refine", side_effect=error):
                await manager._run_refinement(refinement.job_id, refinement)

        asyncio.run(run())

        self.assertEqual(state.jobs_by_id["quota"].status, "refine_failed")
        self.assertIsNotNone(state.jobs_by_id["quota"].retry_message_id)
        self.assertEqual(len(bot.messages), 1)
        _, text, markup = bot.messages[0]
        self.assertIn("Gemini quota is exhausted", text)
        self.assertIn("UTC", text)
        self.assertIsNotNone(markup)
        button = markup.inline_keyboard[0][0]
        self.assertEqual(button.callback_data, "tb:refine:retry:quota")

    def test_model_cooldown_shows_reset_time_and_retry(self):
        bot = FakeBot()
        record = self._failed_refinement_record("cooldown")
        refinement = record.refinement
        error = core._ModelUnavailable("gemini-3.8-flash", retry_after=60)

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "refine", side_effect=error):
                await manager._run_refinement(refinement.job_id, refinement)

        asyncio.run(run())

        self.assertEqual(state.jobs_by_id["cooldown"].status, "refine_failed")
        self.assertIsNotNone(state.jobs_by_id["cooldown"].retry_message_id)
        self.assertEqual(len(bot.messages), 1)
        _, text, markup = bot.messages[0]
        self.assertIn("temporarily unavailable", text)
        self.assertIn("UTC", text)
        self.assertIsNotNone(markup)
        button = markup.inline_keyboard[0][0]
        self.assertEqual(button.callback_data, "tb:refine:retry:cooldown")

    def test_retry_refinement_reruns_and_delivers(self):
        bot = FakeBot()
        record = self._failed_refinement_record("again")
        refinement = record.refinement

        async def run():
            manager = JobManager(bot)
            with mock.patch.object(core, "refine", side_effect=RuntimeError("down")):
                await manager._run_refinement(refinement.job_id, refinement)
            self.assertEqual(state.jobs_by_id["again"].status, "refine_failed")

            with mock.patch.object(core, "refine", return_value="refined text"):
                message = await manager.retry_refinement("again")
                for task in list(manager._refine_tasks):
                    await task
            return message

        message = asyncio.run(run())

        self.assertIn("Retrying", message)
        self.assertNotIn("again", state.jobs_by_id)
        self.assertEqual(len(bot.documents), 1)

    def test_retry_rejected_when_refinement_is_not_failed(self):
        bot = FakeBot()
        job = make_job("active", priority=1, sequence=1)
        state.jobs_by_id[job.job_id] = JobRecord(job=job, status="refining")

        async def run():
            manager = JobManager(bot)
            active = await manager.retry_refinement("active")
            missing = await manager.retry_refinement("nope")
            return active, missing

        active, missing = asyncio.run(run())

        self.assertIn("can no longer be retried", active)
        self.assertIn("can no longer be retried", missing)
        self.assertEqual(state.jobs_by_id["active"].status, "refining")
        self.assertEqual(bot.messages, [])

    def test_resume_skips_retry_prompt_already_sent(self):
        bot = FakeBot()
        job = make_job("notified", priority=1, sequence=1)
        job.refine = True
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "raw text"
        )
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job,
            status="refine_failed",
            counted_active=False,
            retry_message_id=42,
        )

        async def run():
            manager = JobManager(bot)
            await manager.resume()

        asyncio.run(run())

        self.assertIn("notified", state.jobs_by_id)
        self.assertEqual(
            state.jobs_by_id["notified"].status, "refine_failed"
        )
        self.assertIsNotNone(state.jobs_by_id["notified"].refinement)
        self.assertEqual(
            state.jobs_by_id["notified"].retry_message_id, 42
        )
        self.assertEqual(bot.messages, [])

    def test_resume_reoffers_retry_for_failed_refinement(self):
        bot = FakeBot()
        job = make_job("stuck", priority=1, sequence=1)
        job.refine = True
        state.cache_store.save_raw(
            job.md5, job.lang, job.fmt, job.sha256_key, "raw text"
        )
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="refine_failed", counted_active=False
        )

        async def run():
            manager = JobManager(bot)
            await manager.resume()
            return manager

        manager = asyncio.run(run())

        self.assertIn("stuck", state.jobs_by_id)
        self.assertEqual(state.jobs_by_id["stuck"].status, "refine_failed")
        self.assertIsNotNone(state.jobs_by_id["stuck"].refinement)
        _, text, markup = bot.messages[-1]
        self.assertIn("failed earlier", text)
        self.assertEqual(
            markup.inline_keyboard[0][0].callback_data, "tb:refine:retry:stuck"
        )


if __name__ == "__main__":
    unittest.main()
