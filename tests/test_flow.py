"""Flow tests for batched uploads, wizard state and priority assignment."""

import asyncio
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

# The real core imports torch/funasr; these tests only need its interfaces.
if "core" not in sys.modules:
    core = types.ModuleType("core")
    core.apply_opencc = lambda text, lang, zh_format: text
    core.transcribe = lambda *args, **kwargs: ""
    core.refine = lambda *args, **kwargs: ""
    core.warm_up = lambda: None
    sys.modules["core"] = core

import config  # noqa: E402
import handlers  # noqa: E402
import manual  # noqa: E402
import state  # noqa: E402
from cache import sha256_text  # noqa: E402
from models import JobRecord, TranscriptionJob, UploadedFile  # noqa: E402


class FakeBot:
    def __init__(self):
        self.messages = []
        self.documents = []

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        self.messages.append((chat_id, text, reply_markup))
        return None

    async def send_document(self, chat_id, document, **kwargs):
        self.documents.append((chat_id, document))
        return None

    async def send_chat_action(self, chat_id, action):
        return None


class FakeMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []
        self.voice = None
        self.video = None
        self.audio = None
        self.effective_attachment = None

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        return None


class FakeUser:
    def __init__(self, user_id=1):
        self.id = user_id


class FakeChat:
    def __init__(self, chat_id=1):
        self.id = chat_id


class FakeUpdate:
    def __init__(self, text=""):
        self.effective_user = FakeUser()
        self.effective_chat = FakeChat()
        self.message = FakeMessage(text)
        self.callback_query = None


class FakeMessageWithEdit(FakeMessage):
    def __init__(self, text=""):
        super().__init__(text)
        self.edits = []

    async def edit_message_text(self, text, **kwargs):
        self.edits.append((text, kwargs))
        return None

    async def edit_message_reply_markup(self, reply_markup=None):
        return None


class FakeCallbackQuery:
    def __init__(self, data: str):
        self.data = data
        self.message = FakeMessageWithEdit()
        self.answers = 0
        self.markup_edits = []

    async def answer(self, *args, **kwargs):
        self.answers += 1

    async def edit_message_text(self, text, **kwargs):
        self.message.edits.append((text, kwargs))

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_edits.append(reply_markup)


class FakeContext:
    def __init__(self, bot):
        self.bot = bot


class FakeJobs:
    def __init__(self):
        self.enqueued = []
        self.reviews = []

    async def enqueue_transcription(self, job):
        self.enqueued.append(job)
        return len(self.enqueued)

    async def enqueue_review(self, user_id, job_id, kind):
        self.reviews.append((user_id, job_id, kind))


class FlowTests(unittest.TestCase):
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
        config.TRANSCRIBE_ADMIN_USER_ID = None
        state.sessions.clear()
        state.jobs_by_id.clear()
        state.init_state()
        state.jobs = FakeJobs()
        self.bot = FakeBot()

    def tearDown(self):
        self.tempdir.cleanup()

    def _file(self, name="audio.mp3", md5="a" * 32, key=None, source="media"):
        path = config.UPLOAD_DIR / name
        path.write_bytes(b"audio")
        return UploadedFile(
            file_path=str(path),
            file_name=name,
            md5=md5,
            sha256_key=key or sha256_text(name),
            source=source,
        )

    def _session(self):
        session = state.get_session(1, 1)
        state.clear_wizard(session)
        return session

    def _job(self, job_id, user_id=1, status="queued", sequence=1):
        job = TranscriptionJob(
            job_id=job_id,
            user_id=user_id,
            chat_id=user_id,
            file_path=f"/tmp/{job_id}",
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
        return JobRecord(job=job, status=status)

    def _status_text(self, update):
        return "\n".join(text for text, _ in update.message.replies)

    def test_status_lists_only_the_users_jobs(self):
        state.auth._authenticated.add(1)
        state.jobs_by_id["mine"] = self._job(
            "mine", user_id=1, status="processing", sequence=1
        )
        state.jobs_by_id["other"] = self._job(
            "other", user_id=2, status="refining", sequence=2
        )

        update = FakeUpdate("/status")
        asyncio.run(handlers.cmd_status(update, FakeContext(self.bot)))

        text = self._status_text(update)
        self.assertIn("Your active jobs (1):", text)
        self.assertIn("mine.mp3", text)
        self.assertIn("Transcribing", text)
        self.assertIn("job mine", text)
        self.assertNotIn("other.mp3", text)

    def test_status_admin_sees_every_job_with_user_ids(self):
        config.TRANSCRIBE_ADMIN_USER_ID = 1
        state.jobs_by_id["mine"] = self._job(
            "mine", user_id=1, status="queued", sequence=1
        )
        state.jobs_by_id["other"] = self._job(
            "other", user_id=2, status="refine_failed", sequence=2
        )

        update = FakeUpdate("/status")
        asyncio.run(handlers.cmd_status(update, FakeContext(self.bot)))

        text = self._status_text(update)
        self.assertIn("All active jobs (2):", text)
        self.assertIn("[user 1] mine.mp3", text)
        self.assertIn("[user 2] other.mp3", text)
        self.assertIn("Refinement failed (retry available)", text)

    def test_status_requires_authentication_for_non_admins(self):
        state.jobs_by_id["mine"] = self._job("mine", user_id=1)

        update = FakeUpdate("/status")
        asyncio.run(handlers.cmd_status(update, FakeContext(self.bot)))

        text = self._status_text(update)
        self.assertIn("authenticate with /start", text)
        self.assertNotIn("mine.mp3", text)

    def test_status_reports_no_active_jobs(self):
        state.auth._authenticated.add(1)

        update = FakeUpdate("/status")
        asyncio.run(handlers.cmd_status(update, FakeContext(self.bot)))

        self.assertIn("You have no active jobs.", self._status_text(update))

    def test_first_upload_starts_batch(self):
        session = self._session()
        started = handlers._register_upload(session, self._file())
        self.assertEqual(started, "started")
        self.assertEqual(session.stage, "await_language")
        self.assertEqual(len(session.batch_files), 1)

    def test_later_upload_joins_active_batch(self):
        session = self._session()
        session.stage = "await_language"
        session.batch_files.append(self._file("one.mp3"))
        started = handlers._register_upload(session, self._file("two.mp3"))
        self.assertEqual(started, "added")
        self.assertEqual(session.stage, "await_language")
        self.assertEqual(len(session.batch_files), 2)

    def test_finalize_applies_settings_to_whole_batch(self):
        session = self._session()
        files = [self._file("one.mp3", "1" * 32), self._file("two.mp3", "2" * 32)]
        session.batch_files.extend(files)
        session.stage = "await_refine"
        session.lang = "Cantonese"
        session.fmt = "SRT"
        session.zh_format = "Traditional Chinese"
        session.refine = False

        asyncio.run(handlers._finalize(self.bot, 1, session))

        self.assertEqual(len(state.jobs.enqueued), 2)
        self.assertEqual(
            sorted(job.priority for job in state.jobs.enqueued), [1, 2]
        )
        self.assertEqual(session.active_count, 2)
        self.assertEqual(session.next_priority, 3)
        self.assertEqual(session.batch_files, [])
        self.assertEqual(session.stage, "await_upload")
        self.assertEqual(len(state.jobs_by_id), 2)

    def test_upload_after_finalize_starts_a_new_wizard(self):
        session = self._session()
        session.batch_files.append(self._file("first.mp3"))
        session.stage = "await_refine"
        session.lang = "English"
        session.fmt = "Text"
        asyncio.run(handlers._finalize(self.bot, 1, session))
        self.assertEqual(session.stage, "await_upload")

        started = handlers._register_upload(
            session, self._file("second.mp3", "2" * 32)
        )
        self.assertEqual(started, "started")
        self.assertEqual(session.stage, "await_language")
        self.assertEqual(len(session.batch_files), 1)

    def test_mixed_media_and_text_upload_is_rejected(self):
        session = self._session()
        session.stage = "await_language"
        session.batch_files.append(self._file("one.mp3"))

        result = handlers._register_upload(
            session, self._file("notes.txt", "2" * 32, source="text")
        )

        self.assertEqual(result, "mixed")
        self.assertEqual(len(session.batch_files), 1)
        self.assertEqual(session.batch_files[0].source, "media")

    def test_text_upload_skips_format_and_refine_questions(self):
        state.auth._authenticated.add(1)
        session = self._session()
        session.batch_files.append(
            self._file("notes.txt", "3" * 32, source="text")
        )
        session.stage = "await_language"
        update = self._callback_update("tb:lang:mandarin")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_confirm")
        self.assertTrue(session.refine)
        self.assertIsNone(session.fmt)
        self.assertIsNone(session.zh_format)
        text, _ = update.callback_query.message.edits[-1]
        self.assertIn("Output format: Text (from file extension)", text)
        self.assertIn("AI refinement: Yes (required for text uploads)", text)
        self.assertNotIn("Chinese format", text)
        self.assertNotIn("Initial prompt", text)

    def test_srt_upload_asks_for_character_limit(self):
        state.auth._authenticated.add(1)
        session = self._session()
        session.batch_files.append(
            self._file("subs.srt", "4" * 32, source="text")
        )
        session.stage = "await_language"
        update = self._callback_update("tb:lang:english")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_char_limit")

        text_update = FakeUpdate("30")
        asyncio.run(handlers.handle_text(text_update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_confirm")
        self.assertEqual(session.char_limit, 30)
        summary, _ = text_update.message.replies[-1]
        self.assertIn("Characters per subtitle line: 30", summary)

    def test_finalize_text_batch_infers_formats_and_enqueues(self):
        session = self._session()
        session.batch_files.extend(
            [
                self._file("one.txt", "5" * 32, source="text"),
                self._file("two.srt", "6" * 32, source="text"),
            ]
        )
        session.stage = "await_confirm"
        session.lang = "English"
        session.fmt = None
        session.refine = True
        session.zh_format = None

        asyncio.run(handlers._finalize(self.bot, 1, session))

        jobs = state.jobs.enqueued
        self.assertEqual([job.fmt for job in jobs], ["Text", "SRT"])
        self.assertTrue(all(job.source == "text" for job in jobs))
        self.assertTrue(all(job.refine for job in jobs))
        self.assertTrue(all(job.zh_format is None for job in jobs))
        self.assertEqual(state.jobs.reviews, [])

    def test_finalize_cache_hit_uses_review_queue(self):
        key = sha256_text("audio")
        session = self._session()
        uploaded = self._file("cached.mp3", "c" * 32, key)
        state.cache_store.save_raw(
            uploaded.md5, "Cantonese", "SRT", key, "cached text"
        )
        session.batch_files.append(uploaded)
        session.stage = "await_refine"
        session.lang = "Cantonese"
        session.fmt = "SRT"
        session.zh_format = "Traditional Chinese"
        session.refine = False

        asyncio.run(handlers._finalize(self.bot, 1, session))

        self.assertEqual(len(state.jobs.enqueued), 0)
        self.assertEqual(len(state.jobs.reviews), 1)
        _, job_id, kind = state.jobs.reviews[0]
        self.assertEqual(kind, "raw")
        self.assertEqual(state.jobs_by_id[job_id].status, "raw_cache")

    def _callback_update(self, data: str) -> FakeUpdate:
        update = FakeUpdate()
        update.callback_query = FakeCallbackQuery(data)
        return update

    def _ready_session(self):
        session = self._session()
        session.batch_files.extend(
            [self._file("one.mp3", "1" * 32), self._file("two.mp3", "2" * 32)]
        )
        session.stage = "await_refine"
        session.lang = "English"
        session.fmt = "Text"
        return session

    def test_refine_no_prompts_for_confirmation(self):
        state.auth._authenticated.add(1)
        session = self._ready_session()
        update = self._callback_update("tb:refine:no")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_confirm")
        self.assertEqual(state.jobs.enqueued, [])
        self.assertEqual(state.jobs_by_id, {})
        text, kwargs = update.callback_query.message.edits[-1]
        self.assertIn("Files: 2", text)
        self.assertIn("AI refinement: No", text)
        labels = [
            button.text
            for row in kwargs["reply_markup"].inline_keyboard
            for button in row
        ]
        self.assertEqual(labels, ["Confirm", "Edit", "Cancel"])

    def test_confirm_yes_submits_the_batch(self):
        state.auth._authenticated.add(1)
        session = self._ready_session()
        session.stage = "await_confirm"
        update = self._callback_update("tb:confirm:yes")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(len(state.jobs.enqueued), 2)
        self.assertEqual(session.stage, "await_upload")
        self.assertEqual(session.batch_files, [])
        self.assertIn(None, update.callback_query.markup_edits)
        self.assertTrue(
            any("Submitted 2 files." in text for _, text, _ in self.bot.messages)
        )

    def test_confirm_edit_keeps_files_and_restarts_settings(self):
        state.auth._authenticated.add(1)
        session = self._ready_session()
        session.stage = "await_confirm"
        session.refine = True
        session.zh_format = "Traditional Chinese"
        update = self._callback_update("tb:confirm:edit")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_language")
        self.assertEqual(len(session.batch_files), 2)
        self.assertIsNone(session.lang)
        self.assertIsNone(session.fmt)
        self.assertIsNone(session.zh_format)
        self.assertFalse(session.refine)
        self.assertEqual(state.jobs.enqueued, [])
        self.assertIn(None, update.callback_query.markup_edits)
        for uploaded in session.batch_files:
            self.assertTrue(Path(uploaded.file_path).exists())

    def test_confirm_cancel_discards_files_and_resets(self):
        state.auth._authenticated.add(1)
        session = self._ready_session()
        session.stage = "await_confirm"
        paths = [Path(uploaded.file_path) for uploaded in session.batch_files]
        update = self._callback_update("tb:confirm:cancel")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_upload")
        self.assertEqual(session.batch_files, [])
        self.assertEqual(state.jobs.enqueued, [])
        self.assertIn(None, update.callback_query.markup_edits)
        for path in paths:
            self.assertFalse(path.exists())

    def test_confirm_rejected_when_wizard_not_at_review(self):
        state.auth._authenticated.add(1)
        session = self._ready_session()
        update = self._callback_update("tb:confirm:yes")

        asyncio.run(handlers.handle_callback(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_refine")
        self.assertEqual(state.jobs.enqueued, [])
        text, _ = update.callback_query.message.edits[-1]
        self.assertIn("no longer active", text)

    def test_char_limit_text_reply_asks_for_confirmation(self):
        state.auth._authenticated.add(1)
        session = self._session()
        session.batch_files.append(self._file())
        session.stage = "await_char_limit"
        session.lang = "Mandarin"
        session.fmt = "SRT"
        session.refine = True
        update = FakeUpdate("30")

        asyncio.run(handlers.handle_text(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_confirm")
        self.assertEqual(session.char_limit, 30)
        self.assertEqual(state.jobs.enqueued, [])
        text, kwargs = update.message.replies[-1]
        self.assertIn("Characters per subtitle line: 30", text)
        self.assertIn("reply_markup", kwargs)

    def test_start_authenticated_only_replies(self):
        state.auth._authenticated.add(1)
        session = self._session()
        session.stage = "await_language"
        session.batch_files.append(self._file())
        update = FakeUpdate()

        asyncio.run(handlers.cmd_start(update, FakeContext(self.bot)))

        self.assertEqual(
            update.message.replies[-1][0],
            "You are authenticated. Send an audio/video file or a .txt/.srt "
            "document to begin.",
        )
        self.assertEqual(session.stage, "await_language")
        self.assertEqual(len(session.batch_files), 1)

    def test_unauthenticated_upload_creates_no_draft(self):
        update = FakeUpdate()
        update.message.effective_attachment = object()
        session = state.get_session(1, 1)

        asyncio.run(handlers.handle_media(update, FakeContext(self.bot)))

        self.assertEqual(session.batch_files, [])
        self.assertIn("authenticate", update.message.replies[-1][0])

    def test_start_uses_traditional_chinese_from_telegram_locale(self):
        state.auth._authenticated.add(1)
        update = FakeUpdate()
        update.effective_user.language_code = "zh-HK"

        asyncio.run(handlers.cmd_start(update, FakeContext(self.bot)))

        self.assertIn("您已通過驗證", update.message.replies[-1][0])

    def test_finalize_persists_locale_on_jobs(self):
        session = self._session()
        session.locale = "zh-Hans"
        session.lang = "English"
        session.fmt = "Text"
        session.batch_files.append(self._file())

        asyncio.run(handlers._finalize(self.bot, 1, session))

        self.assertEqual(state.jobs.enqueued[0].locale, "zh-Hans")
        self.assertEqual(
            self.bot.messages[-1][1], "已提交 1 个文件。"
        )

    def test_refine_requires_authentication(self):
        update = FakeUpdate("/refine")

        asyncio.run(handlers.cmd_refine(update, FakeContext(self.bot)))

        self.assertIn("authenticate", update.message.replies[-1][0])

    def test_refine_starts_manual_upload_wizard(self):
        state.auth._authenticated.add(1)
        session = self._session()
        session.stage = "await_language"
        update = FakeUpdate("/refine")

        asyncio.run(handlers.cmd_refine(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, handlers.MANUAL_UPLOAD_STAGE)
        self.assertTrue(session.manual_refine)
        self.assertIn("Manual AI refinement", update.message.replies[-1][0])

    def test_manual_wizard_accepts_one_text_file_only(self):
        session = self._session()
        session.stage = handlers.MANUAL_UPLOAD_STAGE
        session.manual_refine = True
        uploaded = self._file("notes.txt", "7" * 32, source="text")

        started = handlers._register_upload(session, uploaded)

        self.assertEqual(started, "started")
        self.assertEqual(session.stage, "await_language")
        self.assertTrue(session.manual_refine)
        self.assertEqual(len(session.batch_files), 1)

        second = self._file("more.txt", "8" * 32, source="text")
        self.assertEqual(
            handlers._register_upload(session, second), "manual_single"
        )

    def test_manual_finalize_sends_the_first_prompt(self):
        state.auth._authenticated.add(1)
        session = self._session()
        uploaded = self._file("notes.txt", "9" * 32, source="text")
        uploaded.file_path = str(config.UPLOAD_DIR / "notes.txt")
        Path(uploaded.file_path).write_text(
            "Hello world.", encoding="utf-8"
        )
        session.batch_files = [uploaded]
        session.stage = "await_confirm"
        session.lang = "English"
        session.manual_refine = True
        session.fmt = None
        session.refine = True

        asyncio.run(handlers._finalize(self.bot, 1, session))

        self.assertEqual(state.jobs.enqueued, [])
        self.assertEqual(session.stage, handlers.MANUAL_RESPONSE_STAGE)
        self.assertEqual(len(session.manual_steps), 1)
        self.assertFalse(Path(uploaded.file_path).exists())
        self.assertTrue(
            any("<pre>" in text for _, text, _ in self.bot.messages)
        )
        self.assertIn("Manual refinement prompt 1/1", self.bot.messages[0][1])

    def test_manual_long_text_sends_a_plan_prompt(self):
        state.auth._authenticated.add(1)
        session = self._session()
        uploaded = self._file("notes.txt", "a" * 32, source="text")
        uploaded.file_path = str(config.UPLOAD_DIR / "notes.txt")
        Path(uploaded.file_path).write_text(
            "This is a sentence. " * 800, encoding="utf-8"
        )
        session.batch_files = [uploaded]
        session.stage = "await_confirm"
        session.lang = "English"
        session.manual_refine = True
        session.fmt = None
        session.refine = True

        asyncio.run(handlers._finalize(self.bot, 1, session))

        self.assertEqual(session.stage, handlers.MANUAL_RESPONSE_STAGE)
        self.assertEqual(
            session.manual_steps[0]["kind"], manual.TEXT_PLAN_KIND
        )
        self.assertIn(
            "Manual structure-planning prompt",
            self.bot.messages[0][1],
        )
        self.assertEqual(len(self.bot.documents), 1)
        prompt = self.bot.documents[0][1].input_file_content.decode("utf-8")
        self.assertIn('"start_unit"', prompt)

    def test_manual_plan_response_builds_refinement_prompts(self):
        state.auth._authenticated.add(1)
        session = self._manual_session()
        session.manual_steps = [
            {
                "kind": manual.TEXT_PLAN_KIND,
                "text": (
                    "First sentence. Second sentence. Third sentence."
                ),
            }
        ]
        payload = json.dumps(
            {
                "sections": [
                    {"start_unit": 1, "heading": "## Intro"},
                    {"start_unit": 2, "heading": None},
                ]
            }
        )
        update = FakeUpdate(payload)

        asyncio.run(handlers.handle_text(update, FakeContext(self.bot)))

        self.assertEqual(len(session.manual_steps), 1)
        self.assertEqual(
            session.manual_steps[0]["kind"], manual.TEXT_KIND
        )
        self.assertTrue(session.manual_steps[0]["starts_with_heading"])
        self.assertEqual(session.manual_step, 0)
        self.assertEqual(session.manual_outputs, [])
        self.assertIn(
            "Structure plan accepted", update.message.replies[-1][0]
        )
        self.assertIn(
            "Manual refinement prompt 1/1", self.bot.messages[0][1]
        )

    def test_invalid_manual_plan_falls_back_to_chunking(self):
        state.auth._authenticated.add(1)
        session = self._manual_session()
        session.manual_steps = [
            {
                "kind": manual.TEXT_PLAN_KIND,
                "text": "This is a sentence. " * 1500,
            }
        ]
        update = FakeUpdate("not json")

        asyncio.run(handlers.handle_text(update, FakeContext(self.bot)))

        self.assertIn("automatic chunking", update.message.replies[-1][0])
        self.assertGreater(len(session.manual_steps), 1)
        self.assertEqual(
            session.manual_steps[0]["kind"], manual.TEXT_KIND
        )
        self.assertEqual(session.manual_step, 0)
        self.assertEqual(session.manual_outputs, [])

    def _manual_session(self, text="AI refined text.", kind="Text"):
        session = self._session()
        session.locale = "en"
        session.stage = handlers.MANUAL_RESPONSE_STAGE
        session.manual_refine = True
        session.manual_name = "notes.txt"
        session.manual_lang = "English"
        session.manual_fmt = kind
        session.manual_steps = [{"kind": kind, "text": "source text"}]
        session.manual_step = 0
        session.manual_outputs = []
        return session

    def test_manual_text_response_finishes_and_sends_document(self):
        state.auth._authenticated.add(1)
        session = self._manual_session()
        update = FakeUpdate("AI refined text.")

        asyncio.run(handlers.handle_text(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_upload")
        self.assertEqual(session.manual_steps, [])
        self.assertEqual(len(self.bot.documents), 1)
        self.assertIn("Manual refinement complete", update.message.replies[-1][0])

    def test_manual_txt_response_file_advances(self):
        state.auth._authenticated.add(1)
        session = self._manual_session()
        source = Path(self.tempdir.name) / "reply.txt"
        source.write_text("AI refined text.", encoding="utf-8")

        class Attachment:
            file_name = "reply.txt"
            file_size = len("AI refined text.")

            async def get_file(self):
                return types.SimpleNamespace(file_path=str(source))

        update = FakeUpdate()
        update.message.effective_attachment = Attachment()

        asyncio.run(handlers.handle_media(update, FakeContext(self.bot)))

        self.assertEqual(session.stage, "await_upload")
        self.assertEqual(len(self.bot.documents), 1)

    def test_long_manual_prompt_uses_txt_attachment(self):
        session = self._manual_session()
        session.manual_steps = [{"kind": "Text", "text": "x" * 5000}]

        asyncio.run(handlers._send_manual_prompt(self.bot, session))

        self.assertEqual(len(self.bot.messages), 1)
        self.assertEqual(len(self.bot.documents), 1)

    def test_exceed_splits_the_pending_prompt(self):
        state.auth._authenticated.add(1)
        session = self._manual_session()
        session.manual_steps = [
            {"kind": "Text", "text": "This is a sentence. " * 400}
        ]
        update = FakeUpdate("/exceed")

        asyncio.run(handlers.cmd_exceed(update, FakeContext(self.bot)))

        self.assertGreater(len(session.manual_steps), 1)
        self.assertEqual(session.manual_step, 0)
        self.assertIn("split into", update.message.replies[-1][0])

    def test_exceed_without_pending_prompt_is_rejected(self):
        state.auth._authenticated.add(1)
        session = self._session()
        update = FakeUpdate("/exceed")

        asyncio.run(handlers.cmd_exceed(update, FakeContext(self.bot)))

        self.assertIn("no manual refinement", update.message.replies[-1][0])

    def test_media_upload_copies_and_hashes_in_one_pass(self):
        state.auth._authenticated.add(1)
        payload = b"hello audio"
        source = Path(self.tempdir.name) / "source.mp3"
        source.write_bytes(payload)

        class Attachment:
            file_name = "clip.mp3"
            file_size = len(payload)

            async def get_file(self):
                return types.SimpleNamespace(file_path=str(source))

        update = FakeUpdate()
        update.message.effective_attachment = Attachment()

        asyncio.run(handlers.handle_media(update, FakeContext(self.bot)))

        session = state.get_session(1, 1)
        self.assertEqual(session.stage, "await_language")
        uploaded = session.batch_files[0]
        self.assertEqual(Path(uploaded.file_path).read_bytes(), payload)
        self.assertEqual(uploaded.md5, hashlib.md5(payload).hexdigest())
        self.assertEqual(
            uploaded.sha256_key, hashlib.sha256(payload).digest()
        )
        self.assertIn("Received clip.mp3", update.message.replies[-1][0])


if __name__ == "__main__":
    unittest.main()
