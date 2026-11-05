"""Authentication, batched uploads, settings wizard and cache callbacks."""

import asyncio
import html
import logging
import os
import uuid
from pathlib import Path

from telegram import InputFile, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

import chunking
import config
import delivery
import i18n
import manual
import state
import ui
from cache import copy_and_hash, hash_file
from models import JobRecord, TranscriptionJob, UploadedFile

logger = logging.getLogger(__name__)

WIZARD_STAGES = {
    "await_language",
    "await_format",
    "await_zh",
    "await_prompt",
    "await_refine",
    "await_c2w",
    "await_char_limit",
    "await_confirm",
}

TEXT_SUFFIXES = {".txt", ".srt"}
MANUAL_UPLOAD_STAGE = "await_manual_upload"
MANUAL_RESPONSE_STAGE = "manual_await_response"
_TELEGRAM_MESSAGE_LIMIT = 3800

STATUS_KEYS = {
    "pending": "status.pending",
    "queued": "status.queued",
    "processing": "status.processing",
    "raw_cache": "status.raw_cache",
    "refine_cache": "status.refine_cache",
    "refining": "status.refining",
    "refine_failed": "status.refine_failed",
}


def _status_label(locale: str, status: str) -> str:
    key = STATUS_KEYS.get(status)
    if key:
        return i18n.t(locale, key)
    return status.replace("_", " ").title()


def _status_lines(records, locale: str, include_user: bool = False) -> list[str]:
    lines = []
    for index, record in enumerate(records, start=1):
        job = record.job
        user_prefix = (
            i18n.t(locale, "status.user_prefix", user_id=job.user_id)
            if include_user
            else ""
        )
        lines.append(
            i18n.t(
                locale,
                "status.line",
                index=index,
                user_prefix=user_prefix,
                file_name=job.file_name,
                status=_status_label(locale, record.status),
                lang=i18n.display_language(locale, job.lang),
                fmt=i18n.display_format(locale, job.fmt),
                job_id=job.job_id,
            )
        )
    return lines


async def _send_status_chunks(message, header: str, lines: list[str]) -> None:
    """Sends status lines without exceeding Telegram's message size limit."""
    chunks = []
    current = header
    for line in lines:
        candidate = f"{current}\n{line}"
        if len(candidate) > 4000 and current != header:
            chunks.append(current)
            current = line
        else:
            current = candidate
    chunks.append(current)
    for chunk in chunks:
        await message.reply_text(chunk)


def _source_for_name(name: str) -> str:
    return "text" if Path(name).suffix.lower() in TEXT_SUFFIXES else "media"


def _is_text_batch(session) -> bool:
    return bool(session.batch_files) and session.batch_files[0].source == "text"


def _upload_format(uploaded: UploadedFile) -> str:
    return "SRT" if Path(uploaded.file_name).suffix.lower() == ".srt" else "Text"


def _text_batch_needs_char_limit(session) -> bool:
    return any(
        _upload_format(uploaded) == "SRT"
        for uploaded in session.batch_files
        if uploaded.source == "text"
    )


def _user_id(update: Update) -> int:
    return update.effective_user.id


def _chat_id(update: Update) -> int:
    return update.effective_chat.id


def _locale(update: Update) -> str:
    return i18n.locale_for_update(update)


def _lock_text(locale: str, seconds: int) -> str:
    minutes = max(1, (int(seconds) + 59) // 60)
    return i18n.t(locale, "auth.locked", minutes=minutes)


def _question_text(locale: str, result, prefix: str = "") -> str:
    return i18n.t(
        locale,
        "auth.question",
        prefix=prefix,
        number=result.question_number,
        total=result.question_total,
        question=result.question,
    )


async def _edit(query, text: str, reply_markup=None) -> None:
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except Exception:
        if query.message:
            await query.message.reply_text(text, reply_markup=reply_markup)
            return
    if reply_markup is None:
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass


async def _clear_markup(query) -> None:
    """Removes the inline keyboard from the message behind a callback."""
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        logger.debug("Could not clear the message keyboard", exc_info=True)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    locale = _locale(update)

    if state.auth.is_authenticated(user_id):
        await update.message.reply_text(
            i18n.t(locale, "auth.authenticated")
        )
        return

    locked_for = state.auth.is_locked(user_id)
    if locked_for:
        await update.message.reply_text(_lock_text(locale, locked_for))
        return

    session = state.get_session(user_id, _chat_id(update))
    session.locale = locale
    state.discard_wizard(session)
    state.save_runtime()
    result = state.auth.start(user_id)
    if result.status == "locked":
        await update.message.reply_text(
            _lock_text(locale, result.lock_remaining)
        )
        return
    await update.message.reply_text(_question_text(locale, result))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    locale = _locale(update)
    is_admin = (
        config.TRANSCRIBE_ADMIN_USER_ID is not None
        and user_id == config.TRANSCRIBE_ADMIN_USER_ID
    )
    if not is_admin and not state.auth.is_authenticated(user_id):
        await update.message.reply_text(
            i18n.t(locale, "auth.status_required")
        )
        return

    records = list(state.jobs_by_id.values())
    if not is_admin:
        records = [
            record for record in records if record.job.user_id == user_id
        ]
    records.sort(
        key=lambda record: (
            record.job.completion_seq,
            record.job.job_id,
        )
    )

    if not records:
        text = i18n.t(
            locale,
            "status.none_admin" if is_admin else "status.none_user",
        )
        await update.message.reply_text(text)
        return

    if is_admin:
        header = i18n.t(
            locale, "status.header_admin", count=len(records)
        )
    else:
        header = i18n.t(
            locale, "status.header_user", count=len(records)
        )
    await _send_status_chunks(
        update.message,
        header,
        _status_lines(records, locale, include_user=is_admin),
    )


async def cmd_refine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    chat_id = _chat_id(update)
    locale = _locale(update)

    if not state.auth.is_authenticated(user_id):
        await update.message.reply_text(
            i18n.t(locale, "auth.upload_required")
        )
        return

    session = state.get_session(user_id, chat_id)
    session.locale = locale
    state.discard_wizard(session)
    session.manual_refine = True
    session.stage = MANUAL_UPLOAD_STAGE
    state.save_runtime()
    await update.message.reply_text(
        i18n.t(locale, "manual.upload_prompt")
    )


async def cmd_exceed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    chat_id = _chat_id(update)
    locale = _locale(update)

    if not state.auth.is_authenticated(user_id):
        await update.message.reply_text(
            i18n.t(locale, "auth.upload_required")
        )
        return

    session = state.get_session(user_id, chat_id)
    session.locale = locale
    if (
        session.stage != MANUAL_RESPONSE_STAGE
        or not session.manual_steps
        or session.manual_step >= len(session.manual_steps)
    ):
        await update.message.reply_text(
            i18n.t(locale, "manual.exceed_unavailable")
        )
        return

    step = session.manual_steps[session.manual_step]
    pieces = manual.split_step(step)
    if len(pieces) <= 1:
        await update.message.reply_text(
            i18n.t(locale, "manual.exceed_impossible")
        )
        return

    session.manual_steps[session.manual_step : session.manual_step + 1] = (
        pieces
    )
    state.save_runtime()
    await update.message.reply_text(
        i18n.t(locale, "manual.exceed_split", count=len(pieces))
    )
    await _send_manual_prompt(context.bot, session)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    chat_id = _chat_id(update)
    locale = _locale(update)
    text = (update.message.text or "").strip()

    if not state.auth.is_authenticated(user_id):
        result = state.auth.submit_answer(user_id, text)
        if result.status == "locked":
            await update.message.reply_text(
                _lock_text(locale, result.lock_remaining)
            )
        elif result.status == "authenticated":
            session = state.get_session(user_id, chat_id)
            session.locale = locale
            state.discard_wizard(session)
            await update.message.reply_text(
                i18n.t(locale, "auth.correct_authenticated")
            )
        else:
            if result.accepted:
                prefix = i18n.t(locale, "auth.correct_prefix")
            else:
                prefix = i18n.t(
                    locale,
                    "auth.incorrect_prefix",
                    attempts=result.attempts_remaining,
                )
            await update.message.reply_text(
                _question_text(locale, result, prefix)
            )
        return

    session = state.get_session(user_id, chat_id)
    session.locale = locale

    if session.stage == MANUAL_RESPONSE_STAGE:
        await _accept_manual_response(update, context.bot, session, text)
        return

    if session.stage == MANUAL_UPLOAD_STAGE:
        await update.message.reply_text(
            i18n.t(locale, "manual.upload_prompt")
        )
        return

    if session.stage == "await_prompt":
        if not text:
            await update.message.reply_text(
                i18n.t(locale, "text.prompt_or_skip"),
                reply_markup=ui.prompt_keyboard(locale),
            )
            return
        session.en_prompt = text[:1000]
        session.stage = "await_refine"
        state.save_runtime()
        await update.message.reply_text(
            i18n.t(locale, "wizard.use_refinement"),
            reply_markup=ui.refine_keyboard(locale),
        )
        return

    if session.stage == "await_char_limit":
        try:
            value = int(text)
        except ValueError:
            value = 0
        if not 10 <= value <= 100:
            await update.message.reply_text(
                i18n.t(locale, "wizard.char_limit_invalid")
            )
            return
        session.char_limit = value
        state.save_runtime()
        await _request_confirm_text(update.message, session)
        return

    if session.stage in WIZARD_STAGES:
        await update.message.reply_text(
            i18n.t(locale, "wizard.use_buttons")
        )
        return

    await update.message.reply_text(i18n.t(locale, "wizard.send_file"))


def _attachment_name(update: Update, attachment) -> str:
    name = getattr(attachment, "file_name", None)
    if name:
        return name
    if update.message.voice:
        return "voice.ogg"
    if update.message.video:
        return "video.mp4"
    if update.message.audio:
        return "audio.m4a"
    return "document.bin"


def _register_upload(session, uploaded: UploadedFile) -> str:
    """Adds an upload to the batch, starting a wizard or rejecting a mix."""
    if session.stage == MANUAL_UPLOAD_STAGE:
        if uploaded.source != "text":
            return "manual_text"
        state.discard_wizard(session)
        session.manual_refine = True
        session.batch_files = [uploaded]
        session.stage = "await_language"
        session.refine = True
        session.zh_format = None
        session.en_prompt = ""
        state.save_runtime()
        return "started"

    if session.manual_refine:
        return "manual_single"

    if session.stage in WIZARD_STAGES:
        if session.batch_files and session.batch_files[0].source != uploaded.source:
            return "mixed"
        session.batch_files.append(uploaded)
        if uploaded.source == "text":
            session.refine = True
            session.zh_format = None
            session.en_prompt = ""
        state.save_runtime()
        return "added"

    state.discard_wizard(session)
    session.batch_files.append(uploaded)
    session.stage = "await_language"
    if uploaded.source == "text":
        session.refine = True
        session.zh_format = None
        session.en_prompt = ""
    state.save_runtime()
    return "started"


async def _download_attachment(attachment, path: Path) -> tuple:
    telegram_file = await attachment.get_file()
    local_path = getattr(telegram_file, "file_path", None)
    if local_path and os.path.isfile(local_path):
        return await asyncio.to_thread(
            copy_and_hash, local_path, str(path)
        )
    await telegram_file.download_to_drive(custom_path=str(path))
    return await asyncio.to_thread(hash_file, str(path))


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = _user_id(update)
    chat_id = _chat_id(update)
    locale = _locale(update)

    if not state.auth.is_authenticated(user_id):
        await update.message.reply_text(
            i18n.t(locale, "auth.upload_required")
        )
        return

    attachment = update.message.effective_attachment
    if isinstance(attachment, (list, tuple)):
        attachment = attachment[-1] if attachment else None
    if attachment is None:
        await update.message.reply_text(
            i18n.t(locale, "upload.could_not_read")
        )
        return

    file_size = getattr(attachment, "file_size", None) or 0
    if file_size > config.MAX_UPLOAD_BYTES:
        await update.message.reply_text(
            i18n.t(locale, "upload.too_large")
        )
        return

    session = state.get_session(user_id, chat_id)
    session.locale = locale
    original_name = _attachment_name(update, attachment)
    source = _source_for_name(original_name)

    if session.stage == MANUAL_RESPONSE_STAGE:
        if source != "text":
            await update.message.reply_text(
                i18n.t(locale, "manual.invalid_response")
            )
            return
        await _accept_manual_document(
            update, context.bot, session, attachment, original_name
        )
        return

    if session.stage == MANUAL_UPLOAD_STAGE and source != "text":
        await update.message.reply_text(
            i18n.t(locale, "manual.text_only")
        )
        return

    suffix = Path(original_name).suffix[:10] or ".bin"
    path = config.UPLOAD_DIR / f"{user_id}_{uuid.uuid4().hex}{suffix}"

    await context.bot.send_chat_action(
        chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT
    )

    try:
        md5_hex, sha256_key = await _download_attachment(attachment, path)
    except Exception as exc:
        logger.exception("Could not download upload: %s", exc)
        try:
            os.remove(path)
        except OSError:
            pass
        await update.message.reply_text(
            i18n.t(locale, "upload.download_failed", error=exc)
        )
        return

    uploaded = UploadedFile(
        file_path=str(path),
        file_name=original_name,
        md5=md5_hex,
        sha256_key=sha256_key,
        source=_source_for_name(original_name),
    )

    result = _register_upload(session, uploaded)
    if result == "mixed":
        state.discard_file(uploaded)
        await update.message.reply_text(i18n.t(locale, "upload.mixed"))
        return
    if result == "manual_text":
        state.discard_file(uploaded)
        await update.message.reply_text(
            i18n.t(locale, "manual.text_only")
        )
        return
    if result == "manual_single":
        state.discard_file(uploaded)
        await update.message.reply_text(
            i18n.t(locale, "manual.single_file")
        )
        return
    if result == "added":
        await update.message.reply_text(
            i18n.t(
                locale,
                "upload.added",
                file_name=original_name,
                count=len(session.batch_files),
            )
        )
        return

    await update.message.reply_text(
        i18n.t(locale, "upload.received", file_name=original_name),
        reply_markup=ui.language_keyboard(locale),
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = _user_id(update)
    chat_id = _chat_id(update)
    locale = _locale(update)

    if not state.auth.is_authenticated(user_id):
        await query.message.reply_text(
            i18n.t(locale, "auth.callback_required")
        )
        return

    session = state.get_session(user_id, chat_id)
    session.locale = locale
    action, value, job_id = ui.parse_callback(query.data)

    if action in ("raw", "ref"):
        if value not in ("use", "new"):
            await _edit(query, i18n.t(locale, "cache.unknown"))
            return
        if not state.jobs.begin_review(user_id, job_id):
            await _edit(query, i18n.t(locale, "cache.inactive"))
            return

        try:
            if action == "raw":
                message = await state.jobs.resolve_raw_cache(
                    job_id, value == "use"
                )
            else:
                message = await state.jobs.resolve_refinement_cache(
                    job_id, value == "use"
                )
            await _edit(query, message)
        except Exception as exc:
            logger.exception("Could not resolve cache choice: %s", exc)
            await _edit(
                query,
                i18n.t(locale, "cache.resolve_failed", error=exc),
            )
        finally:
            await state.jobs.finish_review(user_id)
        return

    if action == "refine" and value == "retry":
        try:
            message = await state.jobs.retry_refinement(job_id)
        except Exception as exc:
            logger.exception("Could not retry refinement: %s", exc)
            message = i18n.t(locale, "cache.retry_failed", error=exc)
        await _edit(query, message)
        return

    if not session.batch_files:
        await _edit(query, i18n.t(locale, "cache.no_files"))
        return

    if action == "confirm":
        if session.stage != "await_confirm":
            await _edit(query, i18n.t(locale, "confirm.inactive"))
            return

        await _clear_markup(query)

        if value == "yes":
            await _finalize(context.bot, chat_id, session)
            return

        if value == "edit":
            session.stage = "await_language"
            session.lang = None
            session.fmt = None
            session.zh_format = None
            session.en_prompt = ""
            session.refine = False
            session.c2w = False
            session.char_limit = 30
            state.save_runtime()
            await _edit(
                query,
                i18n.t(locale, "confirm.edit_language"),
                ui.language_keyboard(locale),
            )
            return

        if value == "cancel":
            state.discard_wizard(session)
            state.save_runtime()
            await _edit(query, i18n.t(locale, "confirm.cancelled"))
            return

        await _edit(query, i18n.t(locale, "confirm.unknown"))
        return

    if action == "lang":
        languages = {
            "cantonese": "Cantonese",
            "mandarin": "Mandarin",
            "english": "English",
        }
        session.lang = languages.get(value)
        if not session.lang:
            await _edit(query, i18n.t(locale, "language.unknown"))
            return
        if _is_text_batch(session):
            session.refine = True
            session.fmt = None
            session.zh_format = None
            session.en_prompt = ""
            state.save_runtime()
            if session.lang == "Cantonese":
                session.stage = "await_c2w"
                state.save_runtime()
                await _edit(
                    query,
                    i18n.t(locale, "c2w.prompt"),
                    ui.c2w_keyboard(locale),
                )
            elif _text_batch_needs_char_limit(session):
                session.stage = "await_char_limit"
                state.save_runtime()
                await _edit(
                    query, i18n.t(locale, "char_limit.prompt")
                )
            else:
                await _request_confirm(query, session)
            return
        session.stage = "await_format"
        state.save_runtime()
        await _edit(
            query,
            i18n.t(
                locale,
                "format.language_chosen",
                lang=i18n.display_language(locale, session.lang),
            ),
            ui.format_keyboard(locale),
        )
        return

    if action == "fmt":
        if not session.lang:
            await _edit(
                query, i18n.t(locale, "format.choose_language_first")
            )
            return
        formats = {"text": "Text", "srt": "SRT"}
        session.fmt = formats.get(value)
        if not session.fmt:
            await _edit(query, i18n.t(locale, "format.unknown"))
            return

        if session.lang == "English":
            session.stage = "await_prompt"
            state.save_runtime()
            await _edit(
                query,
                i18n.t(locale, "prompt.whisper"),
                ui.prompt_keyboard(locale),
            )
        else:
            session.stage = "await_zh"
            state.save_runtime()
            await _edit(
                query,
                i18n.t(locale, "zh.prompt"),
                ui.zh_keyboard(locale),
            )
        return

    if action == "zh":
        zh_formats = {
            "trad": "Traditional Chinese",
            "simp": "Simplified Chinese",
        }
        session.zh_format = zh_formats.get(value)
        if not session.zh_format:
            await _edit(query, i18n.t(locale, "zh.unknown"))
            return
        session.stage = "await_refine"
        state.save_runtime()
        await _edit(
            query,
            i18n.t(locale, "wizard.use_refinement"),
            ui.refine_keyboard(locale),
        )
        return

    if action == "prompt" and value == "skip":
        session.en_prompt = ""
        session.stage = "await_refine"
        state.save_runtime()
        await _edit(
            query,
            i18n.t(locale, "wizard.use_refinement"),
            ui.refine_keyboard(locale),
        )
        return

    if action == "refine":
        if _is_text_batch(session):
            session.refine = True
            state.save_runtime()
            await _request_confirm(query, session)
            return

        session.refine = value == "yes"
        state.save_runtime()
        if not session.refine:
            await _request_confirm(query, session)
            return

        if session.lang == "Cantonese":
            session.stage = "await_c2w"
            state.save_runtime()
            await _edit(
                query,
                i18n.t(locale, "c2w.prompt"),
                ui.c2w_keyboard(locale),
            )
            return

        if session.fmt == "SRT":
            session.stage = "await_char_limit"
            state.save_runtime()
            await _edit(query, i18n.t(locale, "char_limit.prompt"))
            return

        await _request_confirm(query, session)
        return

    if action == "c2w":
        session.c2w = value == "yes"
        state.save_runtime()
        if _is_text_batch(session):
            if _text_batch_needs_char_limit(session):
                session.stage = "await_char_limit"
                state.save_runtime()
                await _edit(
                    query, i18n.t(locale, "char_limit.prompt")
                )
            else:
                await _request_confirm(query, session)
            return
        if session.fmt == "SRT":
            session.stage = "await_char_limit"
            state.save_runtime()
            await _edit(query, i18n.t(locale, "char_limit.prompt"))
            return
        await _request_confirm(query, session)
        return

    await _edit(query, i18n.t(locale, "selection.unknown"))


def _remove_path(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _manual_prompt_filename(name: str, index: int, total: int) -> str:
    stem = Path(name).stem or "refinement"
    return f"{stem} (manual prompt {index} of {total}).txt"


async def _send_manual_prompt(bot, session) -> None:
    index = session.manual_step
    if index < 0 or index >= len(session.manual_steps):
        return
    total = len(session.manual_steps)
    prompt = manual.render_prompt(
        session.manual_steps[index],
        index + 1,
        total,
        session.manual_lang or "English",
        session.manual_fmt,
        session.manual_c2w,
        session.manual_char_limit,
    )
    header_key = (
        "manual.plan_header"
        if manual.is_plan_step(session.manual_steps[index])
        else "manual.prompt_header"
    )
    await bot.send_message(
        chat_id=session.chat_id,
        text=i18n.t(
            session.locale,
            header_key,
            step=index + 1,
            total=total,
        ),
    )
    code_block = f"<pre>{html.escape(prompt)}</pre>"
    if len(code_block) <= _TELEGRAM_MESSAGE_LIMIT:
        await bot.send_message(
            chat_id=session.chat_id,
            text=code_block,
            parse_mode=ParseMode.HTML,
        )
        return
    document = InputFile(
        prompt.encode("utf-8"),
        filename=_manual_prompt_filename(
            session.manual_name, index + 1, total
        ),
    )
    await bot.send_document(chat_id=session.chat_id, document=document)


async def _finish_manual_refinement(update, bot, session) -> None:
    locale = session.locale
    try:
        final_text = manual.merge_outputs(
            session.manual_fmt, session.manual_outputs
        )
    except Exception as exc:
        logger.exception("Could not merge manual refinement outputs")
        await update.message.reply_text(
            i18n.t(locale, "manual.merge_failed", error=exc)
        )
        state.clear_wizard(session)
        state.save_runtime()
        return

    try:
        await delivery.send_text_document(
            bot,
            session.chat_id,
            final_text,
            session.manual_fmt,
            session.manual_name or "refinement",
            "refined",
            locale,
        )
    except Exception as exc:
        logger.warning("Could not send manual refinement: %s", exc)
        await update.message.reply_text(
            i18n.t(locale, "manual.send_failed", error=exc)
        )
        state.clear_wizard(session)
        state.save_runtime()
        return

    try:
        await update.message.reply_text(
            i18n.t(locale, "manual.complete")
        )
    except Exception:
        logger.debug("Could not send manual completion message", exc_info=True)
    state.clear_wizard(session)
    state.save_runtime()


async def _accept_manual_plan(update, bot, session, step, text: str) -> None:
    locale = session.locale
    try:
        payload = manual.parse_json_response(text)
        steps = await asyncio.to_thread(
            manual.build_planned_steps, step, payload
        )
        key = "manual.plan_applied"
    except Exception as exc:
        logger.warning("Could not apply manual structure plan: %s", exc)
        try:
            steps = await asyncio.to_thread(
                manual.build_fallback_steps,
                step.get("text", ""),
                session.manual_fmt,
            )
            key = "manual.plan_fallback"
        except Exception as fallback_exc:
            logger.exception("Could not build manual refinement prompts")
            await update.message.reply_text(
                i18n.t(
                    locale, "manual.start_failed", error=fallback_exc
                )
            )
            state.clear_wizard(session)
            state.save_runtime()
            return

    session.manual_steps[session.manual_step : session.manual_step + 1] = (
        steps
    )
    state.save_runtime()
    await update.message.reply_text(
        i18n.t(locale, key, count=len(steps))
    )
    await _send_manual_prompt(bot, session)


async def _accept_manual_response(update, bot, session, text: str) -> None:
    locale = session.locale
    if (
        not session.manual_steps
        or session.manual_step >= len(session.manual_steps)
    ):
        await update.message.reply_text(
            i18n.t(locale, "manual.exceed_unavailable")
        )
        return

    step = session.manual_steps[session.manual_step]
    if manual.is_plan_step(step):
        await _accept_manual_plan(update, bot, session, step, text)
        return

    try:
        response = manual.normalize_response(
            text, step.get("kind", manual.TEXT_KIND)
        )
    except ValueError:
        key = (
            "manual.invalid_srt"
            if step.get("kind") == manual.SRT_KIND
            else "manual.empty_response"
        )
        await update.message.reply_text(i18n.t(locale, key))
        return

    session.manual_outputs.append(response)
    session.manual_step += 1
    state.save_runtime()

    if session.manual_step < len(session.manual_steps):
        await _send_manual_prompt(bot, session)
        return
    await _finish_manual_refinement(update, bot, session)


async def _accept_manual_document(
    update, bot, session, attachment, original_name: str
) -> None:
    locale = session.locale
    suffix = Path(original_name).suffix[:10] or ".txt"
    path = config.UPLOAD_DIR / (
        f"{session.user_id}_{uuid.uuid4().hex}{suffix}"
    )
    await bot.send_chat_action(
        chat_id=session.chat_id, action=ChatAction.UPLOAD_DOCUMENT
    )
    try:
        await _download_attachment(attachment, path)
        text = await asyncio.to_thread(manual.read_text, str(path))
    except Exception as exc:
        logger.exception("Could not read manual refinement response")
        _remove_path(path)
        await update.message.reply_text(
            i18n.t(locale, "manual.read_failed", error=exc)
        )
        return
    _remove_path(path)
    await _accept_manual_response(update, bot, session, text)


async def _start_manual_refinement(bot, session) -> None:
    locale = session.locale
    uploaded = session.batch_files[0]
    fmt = _upload_format(uploaded)
    name = uploaded.file_name
    lang = session.lang
    c2w = session.c2w
    char_limit = session.char_limit

    try:
        raw_text = await asyncio.to_thread(
            manual.read_text, uploaded.file_path
        )
        if not raw_text.strip():
            raise ValueError("The uploaded text is empty.")
        if fmt == manual.SRT_KIND and not chunking.parse_srt(raw_text):
            raise ValueError("The uploaded SRT is not valid.")
        steps = await asyncio.to_thread(manual.build_steps, raw_text, fmt)
    except Exception as exc:
        logger.exception("Could not start manual refinement")
        await bot.send_message(
            chat_id=session.chat_id,
            text=i18n.t(locale, "manual.start_failed", error=exc),
        )
        state.discard_wizard(session)
        state.save_runtime()
        return

    state.discard_wizard(session)
    session.manual_refine = True
    session.manual_name = name
    session.manual_lang = lang
    session.manual_fmt = fmt
    session.manual_c2w = c2w
    session.manual_char_limit = char_limit
    session.manual_steps = steps
    session.manual_step = 0
    session.manual_outputs = []
    session.stage = MANUAL_RESPONSE_STAGE
    state.save_runtime()

    try:
        await _send_manual_prompt(bot, session)
    except Exception as exc:
        logger.exception("Could not send the first manual prompt")
        await bot.send_message(
            chat_id=session.chat_id,
            text=i18n.t(locale, "manual.send_failed", error=exc),
        )


async def resume_manual_sessions(bot) -> None:
    for session in list(state.sessions.values()):
        if (
            session.stage != MANUAL_RESPONSE_STAGE
            or not session.manual_steps
        ):
            continue
        try:
            await bot.send_message(
                chat_id=session.chat_id,
                text=i18n.t(session.locale, "manual.restart_resume"),
            )
            await _send_manual_prompt(bot, session)
        except Exception:
            logger.exception(
                "Could not resume manual refinement for user %s",
                session.user_id,
            )


def _summary_text(session) -> str:
    """Builds a review of the settings that will apply to the batch."""
    locale = session.locale
    is_text = _is_text_batch(session)
    lines = [
        i18n.t(locale, "summary.review"),
        "",
        i18n.t(locale, "summary.files", count=len(session.batch_files)),
        i18n.t(
            locale,
            "summary.language",
            lang=i18n.display_language(locale, session.lang),
        ),
    ]
    if is_text:
        formats = sorted(
            {
                _upload_format(uploaded)
                for uploaded in session.batch_files
                if uploaded.source == "text"
            }
        )
        lines.append(
            i18n.t(
                locale,
                "summary.output_from_extension",
                formats=", ".join(
                    i18n.display_format(locale, item) for item in formats
                ),
            )
        )
    else:
        lines.append(
            i18n.t(
                locale,
                "summary.output",
                fmt=i18n.display_format(locale, session.fmt),
            )
        )
        if session.zh_format:
            lines.append(
                i18n.t(
                    locale,
                    "summary.zh_format",
                    zh=i18n.display_zh_format(locale, session.zh_format),
                )
            )
        if session.lang == "English":
            prompt = session.en_prompt or i18n.t(locale, "summary.none")
            lines.append(
                i18n.t(locale, "summary.initial_prompt", prompt=prompt)
            )
    if is_text:
        key = (
            "summary.manual_refine"
            if session.manual_refine
            else "summary.refine_required"
        )
        lines.append(i18n.t(locale, key))
    else:
        lines.append(
            i18n.t(
                locale,
                "summary.refine",
                value=i18n.t(
                    locale,
                    "summary.yes" if session.refine else "summary.no",
                ),
            )
        )
    if session.refine and session.lang == "Cantonese":
        lines.append(
            i18n.t(
                locale,
                "summary.c2w",
                value=i18n.t(
                    locale,
                    "summary.yes" if session.c2w else "summary.no",
                ),
            )
        )
    if session.refine and (
        session.fmt == "SRT" or (is_text and _text_batch_needs_char_limit(session))
    ):
        lines.append(
            i18n.t(
                locale,
                "summary.char_limit",
                count=session.char_limit,
            )
        )
    lines.append("")
    lines.append(i18n.t(locale, "summary.footer"))
    return "\n".join(lines)


async def _request_confirm(query, session) -> None:
    """Moves the wizard to a review step before any jobs are submitted."""
    session.stage = "await_confirm"
    state.save_runtime()
    await _edit(
        query,
        _summary_text(session),
        ui.confirm_keyboard(session.locale),
    )


async def _request_confirm_text(message, session) -> None:
    """Review step for wizard flows that finish on a text reply."""
    session.stage = "await_confirm"
    state.save_runtime()
    await message.reply_text(
        _summary_text(session),
        reply_markup=ui.confirm_keyboard(session.locale),
    )


async def _finalize(bot, chat_id: int, session) -> None:
    is_text = _is_text_batch(session)
    if (
        not session.batch_files
        or not session.lang
        or (not is_text and not session.fmt)
    ):
        await bot.send_message(
            chat_id=chat_id,
            text=i18n.t(session.locale, "finalize.missing"),
        )
        return

    if session.manual_refine:
        await _start_manual_refinement(bot, session)
        return

    files = list(session.batch_files)
    priorities = state.assign_batch_priorities(session, len(files))
    records = []
    for uploaded, priority in zip(files, priorities):
        is_text_upload = uploaded.source == "text"
        job = TranscriptionJob(
            job_id=uuid.uuid4().hex[:16],
            user_id=session.user_id,
            chat_id=session.chat_id,
            file_path=uploaded.file_path,
            file_name=uploaded.file_name,
            md5=uploaded.md5,
            sha256_key=uploaded.sha256_key,
            priority=priority,
            completion_seq=state.new_completion_seq(),
            lang=session.lang,
            fmt=_upload_format(uploaded) if is_text_upload else session.fmt,
            zh_format=None if is_text_upload else session.zh_format,
            en_prompt="" if is_text_upload else session.en_prompt,
            refine=True if is_text_upload else session.refine,
            c2w=session.c2w,
            char_limit=session.char_limit,
            source=uploaded.source,
            locale=session.locale,
        )
        record = JobRecord(job=job, status="pending")
        state.jobs_by_id[job.job_id] = record
        records.append(record)

    state.clear_wizard(session)
    state.save_runtime()

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=i18n.t(
                session.locale, "finalize.submitted", count=len(files)
            ),
        )
    except Exception as exc:
        logger.warning("Could not send batch acknowledgement: %s", exc)

    for record in records:
        job = record.job
        if job.source == "text":
            record.status = "queued"
            await state.jobs.enqueue_transcription(job)
            continue

        cached = state.cache_store.load_raw(
            job.md5, job.lang, job.fmt, job.sha256_key
        )
        if cached:
            record.raw_text_pre = cached.get("text", "")
            record.status = "raw_cache"
            await state.jobs.enqueue_review(job.user_id, job.job_id, "raw")
        else:
            record.status = "queued"
            await state.jobs.enqueue_transcription(job)
    state.save_runtime()
