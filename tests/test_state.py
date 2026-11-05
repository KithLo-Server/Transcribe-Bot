"""Tests for runtime state persistence, cleanup protection and recovery."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
import config
import state
from cache import sha256_text
from models import JobRecord, TranscriptionJob, UploadedFile, UserSession


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        config.DATA_DIR = root
        config.UPLOAD_DIR = root / "uploads"
        config.CACHE_DIR = root / "cache"
        config.RAW_CACHE_DIR = config.CACHE_DIR / "raw"
        config.REFINE_CACHE_DIR = config.CACHE_DIR / "refine"
        config.STATE_DIR = root / "state"
        config.AUTH_STATE_FILE = config.STATE_DIR / "auth.json"
        config.RUNTIME_STATE_FILE = config.STATE_DIR / "runtime.json"
        config.SECURITY_QUESTIONS_FILE = root / "questions.json"
        config.SECURITY_QUESTIONS_FILE.write_text(
            json.dumps([{"question": "Q?", "answers": ["A"]}]),
            encoding="utf-8",
        )
        state.sessions.clear()
        state.jobs_by_id.clear()
        state._next_completion_seq = 1

    def tearDown(self):
        state.sessions.clear()
        state.jobs_by_id.clear()
        self.tempdir.cleanup()

    def _job(self, job_id="job1", sequence=1):
        return TranscriptionJob(
            job_id=job_id,
            user_id=1,
            chat_id=10,
            file_path=str(config.UPLOAD_DIR / f"{job_id}.mp3"),
            file_name=f"{job_id}.mp3",
            md5="a" * 32,
            sha256_key=sha256_text(job_id),
            priority=1,
            completion_seq=sequence,
            lang="English",
            fmt="Text",
            zh_format=None,
            en_prompt="",
            refine=False,
            c2w=False,
            char_limit=30,
        )

    def test_runtime_round_trip_preserves_jobs_sessions_and_sequence(self):
        state.init_state()
        session = state.get_session(1, 10)
        session.stage = "await_language"
        (config.UPLOAD_DIR / "waiting.mp3").write_bytes(b"audio")
        session.batch_files.append(
            UploadedFile(
                file_path=str(config.UPLOAD_DIR / "waiting.mp3"),
                file_name="waiting.mp3",
                md5="b" * 32,
                sha256_key=sha256_text("waiting"),
                source="text",
            )
        )
        job = self._job()
        job.source = "text"
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job,
            status="queued",
            counted_active=True,
            retry_message_id=42,
        )
        state._next_completion_seq = 7
        state.save_runtime()

        state.sessions.clear()
        state.jobs_by_id.clear()
        state._next_completion_seq = 1
        state.init_state()

        self.assertIn(1, state.sessions)
        self.assertEqual(len(state.sessions[1].batch_files), 1)
        self.assertEqual(state.sessions[1].batch_files[0].source, "text")
        loaded = state.jobs_by_id["job1"]
        self.assertEqual(loaded.status, "queued")
        self.assertEqual(loaded.retry_message_id, 42)
        self.assertEqual(loaded.job.sha256_key, sha256_text("job1"))
        self.assertEqual(loaded.job.source, "text")
        self.assertEqual(state.new_completion_seq(), 7)

    def test_manual_refinement_session_survives_reload(self):
        state.init_state()
        session = state.get_session(1, 10)
        session.stage = "manual_await_response"
        session.manual_refine = True
        session.manual_name = "notes.txt"
        session.manual_lang = "English"
        session.manual_fmt = "Text"
        session.manual_c2w = False
        session.manual_char_limit = 42
        session.manual_steps = [
            {
                "kind": "Text",
                "text": "first",
                "heading_context": "## One",
                "starts_with_heading": True,
            },
            {"kind": "Text", "text": "second"},
        ]
        session.manual_step = 1
        session.manual_outputs = ["refined first"]
        state.save_runtime()

        state.sessions.clear()
        state.jobs_by_id.clear()
        state.init_state()

        loaded = state.sessions[1]
        self.assertEqual(loaded.stage, "manual_await_response")
        self.assertTrue(loaded.manual_refine)
        self.assertEqual(loaded.manual_name, "notes.txt")
        self.assertEqual(loaded.manual_char_limit, 42)
        self.assertEqual(len(loaded.manual_steps), 2)
        self.assertEqual(
            loaded.manual_steps[0],
            {
                "kind": "Text",
                "text": "first",
                "heading_context": "## One",
                "starts_with_heading": True,
            },
        )
        self.assertEqual(loaded.manual_step, 1)
        self.assertEqual(loaded.manual_outputs, ["refined first"])

    def test_invalid_manual_session_is_reset(self):
        state.init_state()
        session = state.get_session(1, 10)
        session.stage = "manual_await_response"
        session.manual_refine = True
        session.manual_steps = [{"kind": "Text", "text": "only"}]
        session.manual_step = 3
        state.save_runtime()

        state.sessions.clear()
        state.init_state()

        self.assertEqual(state.sessions[1].stage, "await_upload")
        self.assertFalse(state.sessions[1].manual_refine)
        self.assertEqual(state.sessions[1].manual_steps, [])

    def test_stale_cleanup_protects_referenced_uploads(self):
        state.init_state()
        referenced = config.UPLOAD_DIR / "referenced.mp3"
        orphan = config.UPLOAD_DIR / "orphan.mp3"
        referenced.write_bytes(b"keep")
        orphan.write_bytes(b"drop")
        old = time.time() - config.STALE_UPLOAD_SECONDS - 60
        os.utime(referenced, (old, old))
        os.utime(orphan, (old, old))

        job = self._job()
        job.file_path = str(referenced)
        state.jobs_by_id[job.job_id] = JobRecord(
            job=job, status="queued", counted_active=True
        )
        state.save_runtime()

        state.init_state()

        self.assertTrue(referenced.exists())
        self.assertFalse(orphan.exists())

    def test_missing_wizard_files_reset_the_session(self):
        state.init_state()
        session = state.get_session(1, 10)
        session.stage = "await_language"
        session.lang = "English"
        session.batch_files.append(
            UploadedFile(
                file_path=str(config.UPLOAD_DIR / "missing.mp3"),
                file_name="missing.mp3",
                md5="c" * 32,
                sha256_key=sha256_text("missing"),
            )
        )
        state.save_runtime()

        state.init_state()

        self.assertEqual(state.sessions[1].stage, "await_upload")
        self.assertEqual(state.sessions[1].batch_files, [])

    def test_reconcile_rebuilds_active_counts_and_priorities(self):
        state.init_state()
        session = UserSession(
            user_id=1,
            chat_id=10,
            active_count=99,
            next_priority=1,
        )
        state.sessions[1] = session
        for index in range(2):
            job = self._job(f"j{index}", index + 1)
            job.priority = index + 1
            state.jobs_by_id[job.job_id] = JobRecord(
                job=job, status="queued", counted_active=True
            )
        state._reconcile_sessions()

        self.assertEqual(state.sessions[1].active_count, 2)
        self.assertEqual(state.sessions[1].next_priority, 3)


if __name__ == "__main__":
    unittest.main()
