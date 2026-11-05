"""Plain data objects shared by the bot, handlers and job workers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UploadedFile:
    file_path: str
    file_name: str
    md5: str
    sha256_key: bytes
    source: str = "media"


@dataclass
class TranscriptionJob:
    job_id: str
    user_id: int
    chat_id: int
    file_path: str
    file_name: str
    md5: str
    sha256_key: bytes
    priority: int
    completion_seq: int
    lang: str
    fmt: str
    zh_format: Optional[str]
    en_prompt: str
    refine: bool
    c2w: bool
    char_limit: int
    source: str = "media"
    locale: str = "en"


@dataclass
class RefinementJob:
    job_id: str
    user_id: int
    chat_id: int
    file_name: str
    raw_text_pre: str
    raw_text_display: str
    lang: str
    fmt: str
    zh_format: Optional[str]
    c2w: bool
    char_limit: int
    locale: str = "en"


@dataclass
class JobRecord:
    job: TranscriptionJob
    status: str
    raw_text_pre: Optional[str] = None
    raw_text_display: Optional[str] = None
    refinement: Optional[RefinementJob] = None
    counted_active: bool = True
    retry_message_id: Optional[int] = None


@dataclass
class UserSession:
    user_id: int
    chat_id: int
    stage: str = "await_upload"
    batch_files: List[UploadedFile] = field(default_factory=list)
    lang: Optional[str] = None
    fmt: Optional[str] = None
    zh_format: Optional[str] = None
    en_prompt: str = ""
    refine: bool = False
    c2w: bool = False
    char_limit: int = 30
    next_priority: int = 1
    active_count: int = 0
    locale: str = "en"
    manual_refine: bool = False
    manual_name: str = ""
    manual_lang: Optional[str] = None
    manual_fmt: str = "Text"
    manual_c2w: bool = False
    manual_char_limit: int = 30
    manual_steps: List[Dict[str, str]] = field(default_factory=list)
    manual_step: int = 0
    manual_outputs: List[str] = field(default_factory=list)
