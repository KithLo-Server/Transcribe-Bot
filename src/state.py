"""Shared runtime state: authentication, cache store, jobs and user sessions."""

import base64
import logging
import os
import time
from typing import Dict, List, Optional, Set

import config
from auth import AuthManager, load_questions
from cache import CacheStore
from models import (
    JobRecord,
    TranscriptionJob,
    UploadedFile,
    UserSession,
)
from storage import atomic_write_json, read_json

logger = logging.getLogger(__name__)

auth: Optional[AuthManager] = None
cache_store: Optional[CacheStore] = None
jobs = None
sessions: Dict[int, UserSession] = {}
jobs_by_id: Dict[str, JobRecord] = {}
_next_completion_seq = 1


def _encode_key(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii") if value else ""


def _decode_key(value: Optional[str]) -> bytes:
    if not value:
        return b""
    return base64.b64decode(value.encode("ascii"), validate=True)


def _uploaded_to_dict(uploaded: UploadedFile) -> Dict:
    return {
        "file_path": uploaded.file_path,
        "file_name": uploaded.file_name,
        "md5": uploaded.md5,
        "sha256_key": _encode_key(uploaded.sha256_key),
        "source": uploaded.source,
    }


def _uploaded_from_dict(data: Dict) -> UploadedFile:
    return UploadedFile(
        file_path=str(data["file_path"]),
        file_name=str(data["file_name"]),
        md5=str(data["md5"]),
        sha256_key=_decode_key(data.get("sha256_key")),
        source=str(data.get("source", "media")),
    )


def _manual_step_to_dict(step: Dict) -> Dict:
    data = {
        "kind": str(step.get("kind", "Text")),
        "text": str(step.get("text", "")),
    }
    if "heading_context" in step:
        data["heading_context"] = str(step.get("heading_context", ""))
    if "starts_with_heading" in step:
        data["starts_with_heading"] = bool(
            step.get("starts_with_heading")
        )
    return data


def _manual_step_from_dict(data: Dict) -> Dict:
    step = {
        "kind": str(data.get("kind", "Text")),
        "text": str(data.get("text", "")),
    }
    if "heading_context" in data:
        step["heading_context"] = str(data.get("heading_context", ""))
    if "starts_with_heading" in data:
        step["starts_with_heading"] = bool(
            data.get("starts_with_heading")
        )
    return step


def _session_to_dict(session: UserSession) -> Dict:
    return {
        "user_id": session.user_id,
        "chat_id": session.chat_id,
        "stage": session.stage,
        "batch_files": [
            _uploaded_to_dict(uploaded) for uploaded in session.batch_files
        ],
        "lang": session.lang,
        "fmt": session.fmt,
        "zh_format": session.zh_format,
        "en_prompt": session.en_prompt,
        "refine": session.refine,
        "c2w": session.c2w,
        "char_limit": session.char_limit,
        "next_priority": session.next_priority,
        "active_count": session.active_count,
        "locale": session.locale,
        "manual_refine": session.manual_refine,
        "manual_name": session.manual_name,
        "manual_lang": session.manual_lang,
        "manual_fmt": session.manual_fmt,
        "manual_c2w": session.manual_c2w,
        "manual_char_limit": session.manual_char_limit,
        "manual_steps": [
            _manual_step_to_dict(step)
            for step in session.manual_steps
            if isinstance(step, dict)
        ],
        "manual_step": session.manual_step,
        "manual_outputs": [str(output) for output in session.manual_outputs],
    }


def _session_from_dict(data: Dict) -> UserSession:
    return UserSession(
        user_id=int(data["user_id"]),
        chat_id=int(data["chat_id"]),
        stage=str(data.get("stage", "await_upload")),
        batch_files=[
            _uploaded_from_dict(item)
            for item in data.get("batch_files", [])
            if isinstance(item, dict)
        ],
        lang=data.get("lang"),
        fmt=data.get("fmt"),
        zh_format=data.get("zh_format"),
        en_prompt=str(data.get("en_prompt", "")),
        refine=bool(data.get("refine", False)),
        c2w=bool(data.get("c2w", False)),
        char_limit=int(data.get("char_limit", 30)),
        next_priority=int(data.get("next_priority", 1)),
        active_count=int(data.get("active_count", 0)),
        locale=str(data.get("locale", "en")),
        manual_refine=bool(data.get("manual_refine", False)),
        manual_name=str(data.get("manual_name", "")),
        manual_lang=data.get("manual_lang"),
        manual_fmt=str(data.get("manual_fmt", "Text")),
        manual_c2w=bool(data.get("manual_c2w", False)),
        manual_char_limit=int(data.get("manual_char_limit", 30)),
        manual_steps=[
            _manual_step_from_dict(step)
            for step in data.get("manual_steps", [])
            if isinstance(step, dict)
        ],
        manual_step=int(data.get("manual_step", 0)),
        manual_outputs=[
            str(output) for output in data.get("manual_outputs", [])
        ],
    )


def _record_to_dict(record: JobRecord) -> Dict:
    job = record.job
    return {
        "job": {
            "job_id": job.job_id,
            "user_id": job.user_id,
            "chat_id": job.chat_id,
            "file_path": job.file_path,
            "file_name": job.file_name,
            "md5": job.md5,
            "sha256_key": _encode_key(job.sha256_key),
            "priority": job.priority,
            "completion_seq": job.completion_seq,
            "lang": job.lang,
            "fmt": job.fmt,
            "zh_format": job.zh_format,
            "en_prompt": job.en_prompt,
            "refine": job.refine,
            "c2w": job.c2w,
            "char_limit": job.char_limit,
            "source": job.source,
            "locale": job.locale,
        },
        "status": record.status,
        "counted_active": record.counted_active,
        "retry_message_id": record.retry_message_id,
    }


def _record_from_dict(data: Dict) -> JobRecord:
    raw = data["job"]
    job = TranscriptionJob(
        job_id=str(raw["job_id"]),
        user_id=int(raw["user_id"]),
        chat_id=int(raw["chat_id"]),
        file_path=str(raw["file_path"]),
        file_name=str(raw["file_name"]),
        md5=str(raw["md5"]),
        sha256_key=_decode_key(raw.get("sha256_key")),
        priority=int(raw["priority"]),
        completion_seq=int(raw["completion_seq"]),
        lang=str(raw["lang"]),
        fmt=str(raw["fmt"]),
        zh_format=raw.get("zh_format"),
        en_prompt=str(raw.get("en_prompt", "")),
        refine=bool(raw.get("refine", False)),
        c2w=bool(raw.get("c2w", False)),
        char_limit=int(raw.get("char_limit", 30)),
        source=str(raw.get("source", "media")),
        locale=str(raw.get("locale", "en")),
    )
    retry_message_id = data.get("retry_message_id")
    if retry_message_id is not None:
        retry_message_id = int(retry_message_id)
    return JobRecord(
        job=job,
        status=str(data.get("status", "queued")),
        counted_active=bool(data.get("counted_active", True)),
        retry_message_id=retry_message_id,
    )


def _load_runtime() -> None:
    global _next_completion_seq
    sessions.clear()
    jobs_by_id.clear()

    data = read_json(config.RUNTIME_STATE_FILE)
    if not isinstance(data, dict):
        _next_completion_seq = 1
        return

    for raw in data.get("sessions", []):
        try:
            session = _session_from_dict(raw)
        except Exception as exc:
            logger.warning("Ignoring invalid saved session: %s", exc)
            continue
        session.batch_files = [
            uploaded
            for uploaded in session.batch_files
            if os.path.isfile(uploaded.file_path)
        ]
        if session.stage == "manual_await_response":
            if (
                not session.manual_steps
                or session.manual_step < 0
                or session.manual_step >= len(session.manual_steps)
                or len(session.manual_outputs) != session.manual_step
            ):
                clear_wizard(session)
        elif (
            session.stage != "await_upload"
            and session.stage != "await_manual_upload"
            and not session.batch_files
        ):
            clear_wizard(session)
        sessions[session.user_id] = session

    max_sequence = 0
    for raw in data.get("jobs", []):
        try:
            record = _record_from_dict(raw)
        except Exception as exc:
            logger.warning("Ignoring invalid saved job: %s", exc)
            continue
        jobs_by_id[record.job.job_id] = record
        max_sequence = max(max_sequence, record.job.completion_seq)

    persisted_sequence = data.get("next_completion_seq", 1)
    try:
        persisted_sequence = int(persisted_sequence)
    except (TypeError, ValueError):
        persisted_sequence = 1
    _next_completion_seq = max(persisted_sequence, max_sequence + 1)
    _reconcile_sessions()


def _reconcile_sessions() -> None:
    active_counts: Dict[int, int] = {}
    max_priority: Dict[int, int] = {}
    for record in jobs_by_id.values():
        if not record.counted_active:
            continue
        user_id = record.job.user_id
        active_counts[user_id] = active_counts.get(user_id, 0) + 1
        max_priority[user_id] = max(
            max_priority.get(user_id, 0), record.job.priority
        )
        if user_id not in sessions:
            sessions[user_id] = UserSession(
                user_id=user_id, chat_id=record.job.chat_id
            )

    for user_id, session in sessions.items():
        session.active_count = active_counts.get(user_id, 0)
        if session.active_count == 0:
            session.next_priority = 1
        else:
            session.next_priority = max(
                session.next_priority, max_priority.get(user_id, 0) + 1
            )


def _referenced_paths() -> Set[str]:
    """Returns uploads still owned by a job; idle wizard files keep the 24h TTL."""
    return {
        record.job.file_path
        for record in jobs_by_id.values()
        if record.job.file_path
    }


def _cleanup_stale_uploads(protected: Optional[Set[str]] = None) -> None:
    cutoff = time.time() - config.STALE_UPLOAD_SECONDS
    protected_paths = {
        os.path.abspath(path) for path in (protected or set()) if path
    }
    try:
        for entry in config.UPLOAD_DIR.iterdir():
            if os.path.abspath(str(entry)) in protected_paths:
                continue
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def save_runtime() -> None:
    atomic_write_json(
        config.RUNTIME_STATE_FILE,
        {
            "version": 1,
            "next_completion_seq": _next_completion_seq,
            "sessions": [
                _session_to_dict(session)
                for session in sorted(sessions.values(), key=lambda item: item.user_id)
            ],
            "jobs": [
                _record_to_dict(record)
                for record in sorted(
                    jobs_by_id.values(),
                    key=lambda item: (
                        item.job.completion_seq,
                        item.job.job_id,
                    ),
                )
            ],
        },
    )


def init_state() -> None:
    global auth, cache_store
    config.ensure_dirs()
    _load_runtime()
    _cleanup_stale_uploads(_referenced_paths())
    questions = load_questions(str(config.SECURITY_QUESTIONS_FILE))
    auth = AuthManager(
        questions,
        str(config.AUTH_STATE_FILE),
        max_failures=config.MAX_AUTH_FAILURES,
        lockout_seconds=config.LOCKOUT_SECONDS,
    )
    cache_store = CacheStore(config.RAW_CACHE_DIR, config.REFINE_CACHE_DIR)
    save_runtime()


def get_session(user_id: int, chat_id: int) -> UserSession:
    session = sessions.get(user_id)
    if session is None or session.chat_id != chat_id:
        session = UserSession(user_id=user_id, chat_id=chat_id)
        sessions[user_id] = session
    return session


def discard_file(uploaded: UploadedFile) -> None:
    try:
        os.remove(uploaded.file_path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not remove upload %s: %s", uploaded.file_path, exc)


def discard_wizard(session: UserSession) -> None:
    for uploaded in session.batch_files:
        discard_file(uploaded)
    clear_wizard(session)


def clear_wizard(session: UserSession) -> None:
    session.batch_files = []
    session.lang = None
    session.fmt = None
    session.zh_format = None
    session.en_prompt = ""
    session.refine = False
    session.c2w = False
    session.char_limit = 30
    session.stage = "await_upload"
    session.manual_refine = False
    session.manual_name = ""
    session.manual_lang = None
    session.manual_fmt = "Text"
    session.manual_c2w = False
    session.manual_char_limit = 30
    session.manual_steps = []
    session.manual_step = 0
    session.manual_outputs = []


def assign_batch_priorities(session: UserSession, count: int) -> List[int]:
    """Assigns consecutive priorities to a completed settings batch."""
    if session.active_count == 0:
        session.next_priority = 1
    priorities = []
    for _ in range(count):
        priorities.append(session.next_priority)
        session.next_priority += 1
        session.active_count += 1
    return priorities


def new_completion_seq() -> int:
    global _next_completion_seq
    value = _next_completion_seq
    _next_completion_seq += 1
    return value


def complete_active(record: JobRecord) -> None:
    """Marks a job's transcription workflow as finished for priority purposes."""
    if not record.counted_active:
        return
    record.counted_active = False
    session = sessions.get(record.job.user_id)
    if session and session.active_count > 0:
        session.active_count -= 1
        if session.active_count == 0:
            session.next_priority = 1


def remove_job(job_id: str) -> Optional[JobRecord]:
    return jobs_by_id.pop(job_id, None)
