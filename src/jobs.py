"""Priority transcription queue, parallel refinement, and cache reviews."""

import asyncio
import collections
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional, Tuple

import chunking
import config
import core
import delivery
import i18n
import state
import ui
from models import JobRecord, RefinementJob, TranscriptionJob

logger = logging.getLogger(__name__)

# A failed refinement keeps its job record so the user can retry from the chat.
REFINE_FAILED = "refine_failed"


def _remove_file(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not remove %s: %s", path, exc)


def _read_text_upload(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


async def _safe_send(bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:
        logger.warning("Could not send message to %s: %s", chat_id, exc)


async def _safe_send_markup(bot, chat_id: int, text: str, reply_markup):
    try:
        return await bot.send_message(
            chat_id=chat_id, text=text, reply_markup=reply_markup
        )
    except Exception as exc:
        logger.warning("Could not send keyboard message to %s: %s", chat_id, exc)
        await _safe_send(bot, chat_id, text)
        return None


class JobManager:
    def __init__(self, bot):
        self.bot = bot
        self.transcription_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.refine_semaphore = asyncio.Semaphore(config.REFINE_CONCURRENCY)
        self._worker_task: Optional[asyncio.Task] = None
        self._refine_tasks = set()
        self._review_queues: Dict[int, Deque[Tuple[str, str]]] = {}
        self._review_active: Dict[int, str] = {}
        self._review_resolving: Dict[int, str] = {}

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._transcription_worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        for task in list(self._refine_tasks):
            task.cancel()
        if self._refine_tasks:
            await asyncio.gather(*self._refine_tasks, return_exceptions=True)

    async def resume(self) -> None:
        """Rebuilds queues and resumes work left behind by a previous process."""
        for record in list(state.jobs_by_id.values()):
            job_id = record.job.job_id
            try:
                if record.status in ("pending", "queued", "processing"):
                    await self._resume_transcription(record)
                elif record.status == "raw_cache":
                    await self._resume_raw_review(record)
                elif record.status == "refine_cache":
                    await self._resume_refinement_review(record)
                elif record.status == "refining":
                    await self._resume_refinement(record)
                elif record.status == REFINE_FAILED:
                    await self._resume_failed_refinement(record)
                else:
                    logger.warning(
                        "Dropping job %s with unknown status %s",
                        job_id,
                        record.status,
                    )
                    state.remove_job(job_id)
                    state.complete_active(record)
            except Exception:
                logger.exception("Could not resume job %s", job_id)
        state.save_runtime()

    async def _resume_transcription(self, record: JobRecord) -> None:
        job = record.job
        cached = state.cache_store.load_raw(
            job.md5, job.lang, job.fmt, job.sha256_key
        )
        has_file = bool(job.file_path) and os.path.isfile(job.file_path)
        if not cached and not has_file:
            await _safe_send(
                self.bot,
                job.chat_id,
                i18n.t(
                    job.locale,
                    "job.restart_upload_missing",
                    file_name=job.file_name,
                ),
            )
            state.remove_job(job.job_id)
            state.complete_active(record)
            state.save_runtime()
            return

        record.status = "queued"
        state.save_runtime()
        if cached:
            message = i18n.t(
                job.locale,
                "job.restart_resume_cache",
                file_name=job.file_name,
            )
        else:
            message = i18n.t(
                job.locale,
                "job.restart_requeue",
                file_name=job.file_name,
            )
        await _safe_send(self.bot, job.chat_id, message)
        await self.enqueue_transcription(job)

    async def _hydrate_raw_text(self, record: JobRecord) -> bool:
        job = record.job
        cached = state.cache_store.load_raw(
            job.md5, job.lang, job.fmt, job.sha256_key
        )
        if not cached:
            return False
        record.raw_text_pre = cached.get("text", "")
        record.raw_text_display = await asyncio.to_thread(
            core.apply_opencc,
            record.raw_text_pre,
            job.lang,
            job.zh_format,
        )
        return True

    def _refinement_for(self, record: JobRecord) -> RefinementJob:
        job = record.job
        return RefinementJob(
            job_id=job.job_id,
            user_id=job.user_id,
            chat_id=job.chat_id,
            file_name=job.file_name,
            raw_text_pre=record.raw_text_pre or "",
            raw_text_display=record.raw_text_display or record.raw_text_pre or "",
            lang=job.lang,
            fmt=job.fmt,
            zh_format=job.zh_format,
            c2w=job.c2w,
            char_limit=job.char_limit,
            locale=job.locale,
        )

    async def _resume_raw_review(self, record: JobRecord) -> None:
        if not await self._hydrate_raw_text(record):
            await self._resume_transcription(record)
            return
        job = record.job
        record.status = "raw_cache"
        state.save_runtime()
        await _safe_send(
            self.bot,
            job.chat_id,
            i18n.t(
                job.locale,
                "job.restart_cache_review",
                file_name=job.file_name,
            ),
        )
        await self.enqueue_review(job.user_id, job.job_id, "raw")

    async def _resume_refinement_review(self, record: JobRecord) -> None:
        if not await self._hydrate_raw_text(record):
            await self._resume_transcription(record)
            return

        refinement = self._refinement_for(record)
        record.refinement = refinement
        cached = state.cache_store.load_refinement(
            refinement.raw_text_display,
            refinement.lang,
            refinement.fmt,
            refinement.zh_format,
            refinement.c2w,
            refinement.char_limit if refinement.fmt == "SRT" else None,
        )
        if not cached:
            await self.start_refinement(record, refinement)
            return

        job = record.job
        record.status = "refine_cache"
        state.save_runtime()
        await _safe_send(
            self.bot,
            job.chat_id,
            i18n.t(
                job.locale,
                "job.restart_refine_review",
                file_name=job.file_name,
            ),
        )
        await self.enqueue_review(job.user_id, job.job_id, "refine")

    async def _resume_refinement(self, record: JobRecord) -> None:
        if not await self._hydrate_raw_text(record):
            await self._resume_transcription(record)
            return

        refinement = self._refinement_for(record)
        record.refinement = refinement
        cached = state.cache_store.load_refinement(
            refinement.raw_text_display,
            refinement.lang,
            refinement.fmt,
            refinement.zh_format,
            refinement.c2w,
            refinement.char_limit if refinement.fmt == "SRT" else None,
        )
        if not cached:
            await self.start_refinement(record, refinement)
            return

        await _safe_send(
            self.bot,
            refinement.chat_id,
            i18n.t(refinement.locale, "job.restart_send_refinement"),
        )
        try:
            await delivery.send_text_document(
                self.bot,
                refinement.chat_id,
                cached.get("text", ""),
                refinement.fmt,
                refinement.file_name,
                "refined",
                refinement.locale,
            )
        except Exception as exc:
            logger.warning(
                "Could not send resumed refinement: job_id=%s: %s",
                refinement.job_id,
                exc,
            )
            await _safe_send(
                self.bot,
                refinement.chat_id,
                i18n.t(
                    refinement.locale,
                    "job.refinement_send_failed",
                    error=exc,
                ),
            )
        state.remove_job(refinement.job_id)
        state.save_runtime()

    async def _send_failure_prompt(
        self, record: JobRecord, text: str
    ) -> None:
        """Sends the error and Retry prompt, remembering its message id."""
        job = record.job
        message = await _safe_send_markup(
            self.bot,
            job.chat_id,
            text,
            ui.refine_retry_keyboard(job.job_id, job.locale),
        )
        record.retry_message_id = getattr(message, "message_id", None)
        state.save_runtime()

    async def _fail_refinement(
        self, job_id: str, refinement: RefinementJob, text: str
    ) -> None:
        """Marks a refinement failed and offers the Retry button."""
        record = state.jobs_by_id.get(job_id)
        if record is not None:
            record.status = REFINE_FAILED
            record.refinement = refinement
        state.save_runtime()
        if record is not None:
            await self._send_failure_prompt(record, text)
            return
        await _safe_send_markup(
            self.bot,
            refinement.chat_id,
            text,
            ui.refine_retry_keyboard(job_id, refinement.locale),
        )

    @staticmethod
    def _quota_reset_time(retry_after: Optional[float]) -> str:
        seconds = retry_after if retry_after is not None else 0
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return reset_at.strftime("%Y-%m-%d %H:%M UTC")

    async def _resume_failed_refinement(self, record: JobRecord) -> None:
        """Re-offers the Retry button when the failure prompt was not sent."""
        if not await self._hydrate_raw_text(record):
            await self._resume_transcription(record)
            return

        job = record.job
        record.refinement = self._refinement_for(record)
        state.save_runtime()
        if record.retry_message_id is not None:
            return
        await self._send_failure_prompt(
            record,
            i18n.t(
                job.locale,
                "job.refine_failed_earlier",
                file_name=job.file_name,
            ),
        )

    async def retry_refinement(self, job_id: str) -> str:
        """Restarts a failed refinement from the same raw transcription."""
        record = state.jobs_by_id.get(job_id)
        if (
            not record
            or record.status != REFINE_FAILED
            or not record.refinement
        ):
            locale = record.job.locale if record else i18n.DEFAULT_LOCALE
            return i18n.t(locale, "job.retry_unavailable")

        record.retry_message_id = None
        state.save_runtime()
        await self.start_refinement(record, record.refinement)
        return i18n.t(record.job.locale, "job.retrying")

    async def enqueue_transcription(self, job: TranscriptionJob) -> int:
        await self.transcription_queue.put(
            (job.priority, job.completion_seq, job)
        )
        return self.transcription_queue.qsize()

    async def _transcription_worker(self) -> None:
        while True:
            _priority, _sequence, job = await self.transcription_queue.get()
            record = state.jobs_by_id.get(job.job_id)
            try:
                if not record or record.status != "queued":
                    continue
                record.status = "processing"
                state.save_runtime()
                await self._process_transcription(record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Transcription job %s failed", job.job_id)
                _remove_file(job.file_path)
                action = i18n.t(
                    job.locale,
                    "job.action_text_processing"
                    if job.source == "text"
                    else "job.action_transcription",
                )
                await _safe_send(
                    self.bot,
                    job.chat_id,
                    i18n.t(
                        job.locale,
                        "job.action_failed",
                        action=action,
                        file_name=job.file_name,
                        error=exc,
                    ),
                )
                record = state.remove_job(job.job_id)
                if record:
                    state.complete_active(record)
                state.save_runtime()
            finally:
                self.transcription_queue.task_done()

    async def _fail_text_upload(self, record: JobRecord, message: str) -> None:
        await _safe_send(self.bot, record.job.chat_id, message)
        _remove_file(record.job.file_path)
        removed = state.remove_job(record.job.job_id)
        if removed:
            state.complete_active(removed)
        state.save_runtime()

    async def _process_text_upload(self, record: JobRecord, cached) -> None:
        job = record.job
        if cached:
            raw_pre = cached.get("text", "")
        else:
            if not job.file_path or not os.path.isfile(job.file_path):
                await self._fail_text_upload(
                    record,
                    i18n.t(
                        job.locale,
                        "job.text_upload_missing",
                        file_name=job.file_name,
                    ),
                )
                return

            await _safe_send(
                self.bot,
                job.chat_id,
                i18n.t(
                    job.locale, "job.reading", file_name=job.file_name
                ),
            )
            try:
                raw_pre = await asyncio.to_thread(
                    _read_text_upload, job.file_path
                )
            except Exception as exc:
                logger.exception("Could not read text upload %s", job.job_id)
                await self._fail_text_upload(
                    record,
                    i18n.t(
                        job.locale,
                        "job.read_failed",
                        file_name=job.file_name,
                        error=exc,
                    ),
                )
                return

            if not raw_pre.strip():
                await self._fail_text_upload(
                    record,
                    i18n.t(
                        job.locale,
                        "job.text_empty",
                        file_name=job.file_name,
                    ),
                )
                return

            if job.fmt == "SRT" and not chunking.parse_srt(raw_pre):
                await self._fail_text_upload(
                    record,
                    i18n.t(
                        job.locale,
                        "job.invalid_srt",
                        file_name=job.file_name,
                    ),
                )
                return

            state.cache_store.save_raw(
                job.md5, job.lang, job.fmt, job.sha256_key, raw_pre
            )

        _remove_file(job.file_path)
        record.raw_text_pre = raw_pre
        record.raw_text_display = raw_pre
        state.complete_active(record)
        state.save_runtime()
        await self.after_raw(record)

    async def _process_transcription(self, record: JobRecord) -> None:
        job = record.job
        cached = state.cache_store.load_raw(
            job.md5, job.lang, job.fmt, job.sha256_key
        )
        if job.source == "text":
            await self._process_text_upload(record, cached)
            return
        if cached:
            await _safe_send(
                self.bot,
                job.chat_id,
                i18n.t(
                    job.locale,
                    "job.using_cache",
                    file_name=job.file_name,
                ),
            )
            raw_pre = cached.get("text", "")
        else:
            if not job.file_path or not os.path.isfile(job.file_path):
                await _safe_send(
                    self.bot,
                    job.chat_id,
                    i18n.t(
                        job.locale,
                        "job.text_upload_missing",
                        file_name=job.file_name,
                    ),
                )
                removed = state.remove_job(job.job_id)
                if removed:
                    state.complete_active(removed)
                state.save_runtime()
                return

            logger.info(
                "Transcription job started: job_id=%s file=%s lang=%s fmt=%s",
                job.job_id,
                job.file_name,
                job.lang,
                job.fmt,
            )
            await _safe_send(
                self.bot,
                job.chat_id,
                i18n.t(
                    job.locale,
                    "job.transcribing",
                    file_name=job.file_name,
                ),
            )
            started = time.monotonic()
            raw_pre = await asyncio.to_thread(
                core.transcribe,
                job.file_path,
                job.lang,
                job.fmt,
                job.en_prompt,
            )
            state.cache_store.save_raw(
                job.md5, job.lang, job.fmt, job.sha256_key, raw_pre
            )
            logger.info(
                "Transcription job %s completed: file=%s lang=%s fmt=%s "
                "elapsed=%.2fs chars=%d",
                job.job_id,
                job.file_name,
                job.lang,
                job.fmt,
                time.monotonic() - started,
                len(raw_pre),
            )

        raw_display = await asyncio.to_thread(
            core.apply_opencc, raw_pre, job.lang, job.zh_format
        )
        await delivery.send_text_document(
            self.bot,
            job.chat_id,
            raw_display,
            job.fmt,
            job.file_name,
            "raw",
            job.locale,
        )

        _remove_file(job.file_path)
        record.raw_text_pre = raw_pre
        record.raw_text_display = raw_display
        state.complete_active(record)
        state.save_runtime()
        await self.after_raw(record)

    async def after_raw(self, record: JobRecord) -> None:
        job = record.job
        if not job.refine:
            state.remove_job(job.job_id)
            state.save_runtime()
            return

        if not record.raw_text_pre or not record.raw_text_pre.strip():
            await _safe_send(
                self.bot,
                job.chat_id,
                i18n.t(
                    job.locale,
                    "job.raw_empty",
                    file_name=job.file_name,
                ),
            )
            state.remove_job(job.job_id)
            state.save_runtime()
            return

        refinement = self._refinement_for(record)
        record.refinement = refinement

        cached = state.cache_store.load_refinement(
            refinement.raw_text_display,
            refinement.lang,
            refinement.fmt,
            refinement.zh_format,
            refinement.c2w,
            refinement.char_limit if refinement.fmt == "SRT" else None,
        )
        if cached:
            record.status = "refine_cache"
            state.save_runtime()
            await self.enqueue_review(job.user_id, job.job_id, "refine")
            return

        await self.start_refinement(record, refinement)

    async def start_refinement(
        self, record: JobRecord, refinement: RefinementJob
    ) -> None:
        record.refinement = refinement
        record.status = "refining"
        state.save_runtime()
        await _safe_send(
            self.bot,
            refinement.chat_id,
            i18n.t(
                refinement.locale,
                "job.refining",
                file_name=refinement.file_name,
            ),
        )
        task = asyncio.create_task(
            self._run_refinement(refinement.job_id, refinement)
        )
        self._refine_tasks.add(task)
        task.add_done_callback(self._refine_tasks.discard)

    async def _run_refinement(
        self, job_id: str, refinement: RefinementJob
    ) -> None:
        async with self.refine_semaphore:
            logger.info(
                "Refinement job started: job_id=%s file=%s fmt=%s",
                job_id,
                refinement.file_name,
                refinement.fmt,
            )
            started = time.monotonic()
            try:
                refined_text = await asyncio.to_thread(
                    core.refine,
                    refinement.raw_text_display,
                    refinement.lang,
                    refinement.fmt,
                    refinement.c2w,
                    refinement.char_limit,
                    job_id=job_id,
                    cache_store=state.cache_store,
                )
            except asyncio.CancelledError:
                raise
            except core._QuotaExhausted as exc:
                logger.warning(
                    "Refinement job %s hit Gemini quota exhaustion: %s",
                    job_id,
                    exc,
                )
                text = i18n.t(
                    refinement.locale,
                    "job.quota_exhausted",
                    reset_at=self._quota_reset_time(exc.retry_after),
                )
                await self._fail_refinement(job_id, refinement, text)
                return
            except core._ModelUnavailable as exc:
                logger.warning(
                    "Refinement job %s hit a model cooldown: %s", job_id, exc
                )
                text = i18n.t(
                    refinement.locale,
                    "job.model_unavailable",
                    reset_at=self._quota_reset_time(exc.retry_after),
                )
                await self._fail_refinement(job_id, refinement, text)
                return
            except Exception as exc:
                logger.exception("Refinement job %s failed", job_id)
                text = i18n.t(
                    refinement.locale,
                    "job.refine_failed",
                    error=exc,
                )
                await self._fail_refinement(job_id, refinement, text)
                return

            logger.info(
                "Refinement job completed: job_id=%s file=%s elapsed=%.2fs "
                "chars=%d",
                job_id,
                refinement.file_name,
                time.monotonic() - started,
                len(refined_text),
            )

            try:
                state.cache_store.save_refinement(
                    refinement.raw_text_display,
                    refinement.lang,
                    refinement.fmt,
                    refinement.zh_format,
                    refinement.c2w,
                    refinement.char_limit
                    if refinement.fmt == "SRT"
                    else None,
                    refined_text,
                )
            except Exception as exc:
                logger.exception(
                    "Could not save refinement cache: job_id=%s: %s",
                    job_id,
                    exc,
                )

            try:
                await delivery.send_text_document(
                    self.bot,
                    refinement.chat_id,
                    refined_text,
                    refinement.fmt,
                    refinement.file_name,
                    "refined",
                    refinement.locale,
                )
            except Exception as exc:
                logger.warning(
                    "Could not send refined document: job_id=%s: %s",
                    job_id,
                    exc,
                )
                await _safe_send(
                    self.bot,
                    refinement.chat_id,
                    i18n.t(
                        refinement.locale,
                        "job.refinement_send_failed",
                        error=exc,
                    ),
                )

            state.remove_job(job_id)
            state.save_runtime()

    # ----------------- Cache review queue -----------------
    async def enqueue_review(self, user_id: int, job_id: str, kind: str) -> None:
        queue = self._review_queues.setdefault(user_id, collections.deque())
        queue.append((kind, job_id))
        if (
            user_id not in self._review_active
            and user_id not in self._review_resolving
        ):
            await self._send_next_review(user_id)

    def is_active_review(self, user_id: int, job_id: str) -> bool:
        return self._review_active.get(user_id) == job_id

    def begin_review(self, user_id: int, job_id: str) -> bool:
        """Claims the active review so double clicks cannot advance it twice."""
        if self._review_active.get(user_id) != job_id:
            return False
        if user_id in self._review_resolving:
            return False
        self._review_resolving[user_id] = job_id
        return True

    async def finish_review(self, user_id: int) -> None:
        self._review_resolving.pop(user_id, None)
        queue = self._review_queues.get(user_id)
        if queue:
            queue.popleft()
        self._review_active.pop(user_id, None)
        await self._send_next_review(user_id)

    async def _send_next_review(self, user_id: int) -> None:
        if user_id in self._review_resolving:
            return
        queue = self._review_queues.get(user_id)
        while queue:
            kind, job_id = queue[0]
            record = state.jobs_by_id.get(job_id)
            expected = "raw_cache" if kind == "raw" else "refine_cache"
            if not record or record.status != expected:
                queue.popleft()
                continue

            self._review_active[user_id] = job_id
            job = record.job
            if kind == "raw":
                text = i18n.t(
                    job.locale,
                    "job.cache_existing_raw",
                    file_name=job.file_name,
                )
                markup = ui.raw_cache_keyboard(job_id, job.locale)
            else:
                text = i18n.t(
                    job.locale,
                    "job.cache_existing_refine",
                    file_name=job.file_name,
                )
                markup = ui.refine_cache_keyboard(job_id, job.locale)

            try:
                await self.bot.send_message(
                    chat_id=job.chat_id, text=text, reply_markup=markup
                )
                return
            except Exception as exc:
                logger.warning(
                    "Could not send cache review prompt: job_id=%s: %s",
                    job_id,
                    exc,
                )
                queue.popleft()
                self._review_active.pop(user_id, None)

        self._review_active.pop(user_id, None)
        if not queue:
            self._review_queues.pop(user_id, None)

    async def resolve_raw_cache(self, job_id: str, use_existing: bool) -> str:
        record = state.jobs_by_id.get(job_id)
        if not record or record.status != "raw_cache":
            return i18n.t(
                record.job.locale, "job.raw_cache_inactive"
            )

        job = record.job
        if not use_existing:
            record.status = "queued"
            state.save_runtime()
            await self.enqueue_transcription(job)
            return i18n.t(job.locale, "job.re_transcribing")

        raw_pre = record.raw_text_pre or ""
        raw_display = await asyncio.to_thread(
            core.apply_opencc, raw_pre, job.lang, job.zh_format
        )
        await delivery.send_text_document(
            self.bot,
            job.chat_id,
            raw_display,
            job.fmt,
            job.file_name,
            "raw",
            job.locale,
        )
        _remove_file(job.file_path)
        record.raw_text_pre = raw_pre
        record.raw_text_display = raw_display
        state.complete_active(record)
        state.save_runtime()
        await self.after_raw(record)
        return i18n.t(job.locale, "job.using_existing_raw")

    async def resolve_refinement_cache(
        self, job_id: str, use_existing: bool
    ) -> str:
        record = state.jobs_by_id.get(job_id)
        if (
            not record
            or record.status != "refine_cache"
            or not record.refinement
        ):
            return i18n.t(
                record.job.locale, "job.refine_cache_inactive"
            )

        refinement = record.refinement
        if not use_existing:
            await self.start_refinement(record, refinement)
            return i18n.t(
                refinement.locale, "job.starting_new_refinement"
            )

        cached = state.cache_store.load_refinement(
            refinement.raw_text_display,
            refinement.lang,
            refinement.fmt,
            refinement.zh_format,
            refinement.c2w,
            refinement.char_limit if refinement.fmt == "SRT" else None,
        )
        if not cached:
            await self.start_refinement(record, refinement)
            return i18n.t(
                refinement.locale, "job.cache_unreadable"
            )

        await delivery.send_text_document(
            self.bot,
            refinement.chat_id,
            cached.get("text", ""),
            refinement.fmt,
            refinement.file_name,
            "refined",
            refinement.locale,
        )
        state.remove_job(job_id)
        state.save_runtime()
        return i18n.t(
            refinement.locale, "job.using_existing_refinement"
        )
