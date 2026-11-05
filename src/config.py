"""Environment configuration and filesystem locations for the transcribe bot."""

import logging
import os
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("TranscribeBot")

# Third-party libraries are noisy at INFO level: httpx logs every Telegram HTTP
# request, while FunASR/modelscope log model loading and inference details.
# Raise their named loggers, and filter records they emit through the root logger
# directly (as FunASR does), so our own INFO logs remain unchanged.
_QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "telegram.request",
    "funasr",
    "modelscope",
    "faster_whisper",
    "transformers",
    "google_genai",
)
for _quiet_logger in _QUIET_LOGGERS:
    logging.getLogger(_quiet_logger).setLevel(logging.WARNING)

_QUIET_LOG_PATHS = (
    "/funasr/",
    "/modelscope/",
    "/faster_whisper/",
    "/transformers/",
    "/google/genai/",
)


class _ThirdPartyInfoFilter(logging.Filter):
    """Drops INFO/DEBUG records emitted from quiet third-party packages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        path = record.pathname.replace("\\", "/")
        return not any(marker in path for marker in _QUIET_LOG_PATHS)


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_ThirdPartyInfoFilter())

TELEGRAM_BOT_TOKEN = os.getenv("TRANSCRIBE_BOT_TOKEN") or os.getenv(
    "TELEGRAM_BOT_TOKEN", ""
)
TELEGRAM_API_BASE_URL = os.getenv(
    "TELEGRAM_API_BASE_URL", "https://api.telegram.org/bot"
)
TELEGRAM_API_FILE_BASE_URL = os.getenv(
    "TELEGRAM_API_FILE_BASE_URL", "https://api.telegram.org/file/bot"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def parse_api_keys(value: str) -> list[str]:
    """Splits a comma-separated API key list, ignoring blank entries."""
    return [key.strip() for key in value.split(",") if key.strip()]


def parse_optional_user_id(value: str) -> Optional[int]:
    """Parses an optional Telegram user ID, ignoring blank or invalid values."""
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid TRANSCRIBE_ADMIN_USER_ID: %r", value)
        return None


TRANSCRIBE_ADMIN_USER_ID = parse_optional_user_id(
    os.getenv("TRANSCRIBE_ADMIN_USER_ID", "")
)
GEMINI_API_KEYS = parse_api_keys(GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_CHUNK_MODEL = os.getenv("GEMINI_CHUNK_MODEL", "gemini-3.1-flash-lite")
GEMINI_C2W_MODEL = os.getenv("GEMINI_C2W_MODEL", "gemini-3.8-flash")
GEMINI_MODEL_COOLDOWN_SECONDS = int(
    os.getenv("GEMINI_MODEL_COOLDOWN_SECONDS", "60")
)
GEMINI_QUOTA_COOLDOWN_SECONDS = int(
    os.getenv("GEMINI_QUOTA_COOLDOWN_SECONDS", "60")
)
GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS = int(
    os.getenv("GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS", "3600")
)

CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
REFINE_CONCURRENCY = int(os.getenv("REFINE_CONCURRENCY", "10"))
REFINE_CHUNK_TOKENS = int(os.getenv("REFINE_CHUNK_TOKENS", "4000"))
REFINE_CHUNK_HARD_TOKENS = int(
    os.getenv("REFINE_CHUNK_HARD_TOKENS", "6000")
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data/bot"))
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
RAW_CACHE_DIR = CACHE_DIR / "raw"
REFINE_CACHE_DIR = CACHE_DIR / "refine"
STATE_DIR = DATA_DIR / "state"
AUTH_STATE_FILE = STATE_DIR / "auth.json"
RUNTIME_STATE_FILE = STATE_DIR / "runtime.json"

SECURITY_QUESTIONS_FILE = Path(
    os.getenv("SECURITY_QUESTIONS_FILE", str(DATA_DIR / "security_questions.json"))
)

MAX_AUTH_FAILURES = 5
LOCKOUT_SECONDS = 3600
STALE_UPLOAD_SECONDS = 24 * 60 * 60
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))

TELEGRAM_READ_TIMEOUT = float(os.getenv("TELEGRAM_READ_TIMEOUT", "120"))
TELEGRAM_WRITE_TIMEOUT = float(os.getenv("TELEGRAM_WRITE_TIMEOUT", "600"))
TELEGRAM_CONNECT_TIMEOUT = float(os.getenv("TELEGRAM_CONNECT_TIMEOUT", "30"))
TELEGRAM_POOL_TIMEOUT = float(os.getenv("TELEGRAM_POOL_TIMEOUT", "30"))
TELEGRAM_MEDIA_WRITE_TIMEOUT = float(
    os.getenv("TELEGRAM_MEDIA_WRITE_TIMEOUT", "600")
)


def ensure_dirs() -> None:
    """Creates the runtime directories used for uploads, cache and state."""
    for directory in (
        DATA_DIR,
        UPLOAD_DIR,
        RAW_CACHE_DIR,
        REFINE_CACHE_DIR,
        STATE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
