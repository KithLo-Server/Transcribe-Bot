"""Transcription and AI refinement, adapted from the Gradio transcribe app.

Transcription returns text *before* OpenCC conversion so the cache stores the raw
transcription exactly as produced by Whisper/FunASR.
"""

import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import opencc
import torch
from faster_whisper import WhisperModel
from funasr import AutoModel
from google import genai
from google.genai import types

import cache
import chunking
import config
import prompts

logger = logging.getLogger(__name__)


def _log_job_id(job_id: Optional[str]) -> str:
    return job_id or "unknown"


def _refine_kwargs(
    job_id: Optional[str], cache_store
) -> dict:
    """Keeps optional cache arguments out of legacy call signatures."""
    kwargs = {"job_id": job_id}
    if cache_store is not None:
        kwargs["cache_store"] = cache_store
    return kwargs


def _load_intermediate(
    cache_store, settings, *, job_id: Optional[str] = None
) -> Optional[str]:
    if cache_store is None:
        return None
    try:
        payload = cache_store.load_refinement_part(settings)
    except Exception as exc:
        logger.warning(
            "Could not read intermediate refinement cache: job_id=%s: %s",
            _log_job_id(job_id),
            exc,
        )
        return None
    if not payload:
        return None
    logger.info(
        "Using cached intermediate refinement response: job_id=%s kind=%s",
        _log_job_id(job_id),
        settings.get("kind"),
    )
    return payload.get("text")


def _save_intermediate(
    cache_store, settings, text: str, *, job_id: Optional[str] = None
) -> None:
    if cache_store is None:
        return
    try:
        cache_store.save_refinement_part(settings, text)
    except Exception as exc:
        logger.warning(
            "Could not save intermediate refinement cache: job_id=%s: %s",
            _log_job_id(job_id),
            exc,
        )


torch.set_num_threads(config.CPU_THREADS)

_whisper_model: Optional[WhisperModel] = None
_funasr_model: Optional[AutoModel] = None
_model_lock = threading.Lock()

# Converters load dictionaries on construction, so keep one per worker thread
# instead of rebuilding them for every transcription/refinement.
_opencc_local = threading.local()


def _monkeypatch_av() -> None:
    try:
        import av

        if not hasattr(av, "_patched_for_whisper"):
            original_open = av.open

            def safe_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return original_open(*args, **kwargs)

            av.open = safe_open
            av._patched_for_whisper = True
    except ImportError:
        pass


_monkeypatch_av()


def _load_models() -> None:
    global _whisper_model, _funasr_model
    with _model_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(
                "medium.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=config.CPU_THREADS,
            )
        if _funasr_model is None:
            _funasr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                device="cpu",
                ncpu=config.CPU_THREADS,
                trust_remote_code=True,
                disable_pbar=True,
                vad_kwargs={"disable_pbar": True},
                punc_kwargs={"disable_pbar": True},
            )


def warm_up() -> None:
    _load_models()


def _models():
    if _whisper_model is None or _funasr_model is None:
        _load_models()
    return _whisper_model, _funasr_model


def format_srt_time(ms: float) -> str:
    total_ms = max(0, int(round(ms)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def clean_sensevoice_text(text: str) -> str:
    return re.sub(r"<\|.*?\|>", "", text).strip()


def build_funasr_srt(item: dict) -> str:
    srt_blocks = []
    sentence_info = item.get("sentence_info")

    if sentence_info:
        for seg in sentence_info:
            text = clean_sensevoice_text(seg.get("text", ""))
            if not text:
                continue

            if "start" in seg and "end" in seg:
                start_ms, end_ms = seg["start"], seg["end"]
            elif "timestamp" in seg and seg["timestamp"]:
                ts = seg["timestamp"]
                if isinstance(ts[0], (list, tuple)):
                    start_ms, end_ms = ts[0][0], ts[-1][1]
                else:
                    start_ms, end_ms = ts[0], ts[1]
            else:
                start_ms, end_ms = 0, 0

            idx = len(srt_blocks) + 1
            srt_blocks.append(
                f"{idx}\n{format_srt_time(start_ms)} --> "
                f"{format_srt_time(end_ms)}\n{text}"
            )

        if srt_blocks:
            return "\n\n".join(srt_blocks)

    text = clean_sensevoice_text(item.get("text", ""))
    timestamps = item.get("timestamp") or item.get("timestamps") or []

    if timestamps and text:
        start_ms = timestamps[0][0] if isinstance(timestamps[0], (list, tuple)) else 0
        end_ms = timestamps[-1][1] if isinstance(timestamps[-1], (list, tuple)) else 0
        return f"1\n{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n{text}"

    return text


def transcribe(
    audio_path: str,
    lang: str,
    output_format: str,
    en_prompt: str = "",
) -> str:
    whisper_model, funasr_model = _models()

    if lang == "English":
        segments, _ = whisper_model.transcribe(
            audio_path,
            initial_prompt=en_prompt if en_prompt else None,
            beam_size=1,
            vad_filter=True,
        )
        if output_format == "SRT":
            srt_blocks = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                idx = len(srt_blocks) + 1
                start_str = format_srt_time(segment.start * 1000.0)
                end_str = format_srt_time(segment.end * 1000.0)
                srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}")
            return "\n\n".join(srt_blocks)
        return "".join([segment.text for segment in segments])

    lang_code = "yue" if lang == "Cantonese" else "zh"

    if output_format == "SRT":
        try:
            res = funasr_model.generate(
                input=audio_path,
                language=lang_code,
                merge_vad=False,
                sentence_timestamp=True,
                output_timestamp=True,
                return_time_stamps=True,
            )
        except Exception:
            original_punc = getattr(funasr_model, "punc_model", None)
            try:
                funasr_model.punc_model = None
                res = funasr_model.generate(
                    input=audio_path,
                    language=lang_code,
                    merge_vad=False,
                    use_itn=True,
                    sentence_timestamp=True,
                    output_timestamp=True,
                )
            finally:
                funasr_model.punc_model = original_punc

        item = res[0] if isinstance(res, list) and res else {}
        return build_funasr_srt(item)

    res = funasr_model.generate(input=audio_path, language=lang_code, merge_vad=False)
    raw_text = res[0]["text"] if isinstance(res, list) and res else str(res)
    return clean_sensevoice_text(raw_text)


def _opencc_converter(name: str):
    cache = getattr(_opencc_local, "converters", None)
    if cache is None:
        cache = {}
        _opencc_local.converters = cache
    converter = cache.get(name)
    if converter is None:
        converter = opencc.OpenCC(name)
        cache[name] = converter
    return converter


def apply_opencc(text: str, lang: str, zh_format: Optional[str]) -> str:
    if lang not in ("Cantonese", "Mandarin") or not zh_format:
        return text
    name = "s2t" if zh_format == "Traditional Chinese" else "t2s"
    return _opencc_converter(name).convert(text)


_REFINE_MIN_LENGTH_RATIO = 0.2
_REFINE_MIN_SPLIT_TOKENS = 200
_CANTONESE_MARKERS = (
    "嘅", "咗", "喺", "唔", "佢", "哋", "乜", "冇", "啲", "咁", "噉", "嗰", "喎", "囉", "㗎",
)
_BAD_FINISH_REASONS = {
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "OTHER",
    "CONTINUATION",
}

# A long retry delay means the quota will not recover during a normal retry
# window, so it is treated like a daily quota exhaustion instead.
_QUOTA_LONG_RETRY_SECONDS = 300
_DAILY_QUOTA_MARKERS = (
    "per day",
    "per-day",
    "per_day",
    "perday",
    "daily",
)


class _QuotaExhausted(RuntimeError):
    """Raised when Gemini quota is exhausted for every configured key."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class _ModelUnavailable(RuntimeError):
    """Raised while a Gemini model is cooling down after a 503."""

    def __init__(
        self,
        model: str,
        *,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            f"Gemini model {model} is temporarily unavailable; "
            "retry after the cooldown."
        )
        self.model = model
        self.retry_after = retry_after


def _exception_text(exc: BaseException) -> str:
    """Collects the parts of a google-genai error that describe its quota."""
    parts = [str(exc), str(getattr(exc, "message", "") or "")]
    for attribute in ("details", "response_json"):
        value = getattr(exc, attribute, None)
        if value is None:
            continue
        try:
            parts.append(json.dumps(value, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            parts.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(
                json.dumps(
                    response.json(), ensure_ascii=False, default=str
                )
            )
        except Exception:
            parts.append(str(getattr(response, "text", "") or ""))
    return " ".join(part for part in parts if part)


def _retry_after_seconds(text: str) -> Optional[float]:
    """Reads a RetryInfo delay from a Gemini error payload."""
    patterns = (
        r"retry[_ ]?delay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)\s*s",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _quota_retry_after(exc: BaseException) -> Optional[float]:
    """Returns how long a 429 key should be cooled down.

    Every 429 is treated as a quota/rate-limit signal so a key is never retried
    in a tight loop. ``None`` is returned for non-429 errors so other handlers
    (e.g. the model-wide 503 cooldown) can process them.

    A long RetryInfo delay (or explicit daily-quota text) uses the daily
    cooldown; short/transient 429s use the general quota cooldown.
    """
    if getattr(exc, "code", None) != 429:
        return None
    text = _exception_text(exc)
    lowered = text.lower()
    retry_after = _retry_after_seconds(text)
    is_daily = any(marker in lowered for marker in _DAILY_QUOTA_MARKERS)
    if (
        not is_daily
        and retry_after is not None
        and retry_after >= _QUOTA_LONG_RETRY_SECONDS
    ):
        is_daily = True
    if retry_after is not None and retry_after > 0:
        return retry_after
    if is_daily:
        return float(max(1, config.GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS))
    return float(max(1, config.GEMINI_QUOTA_COOLDOWN_SECONDS))


_CORRECTION_INSTRUCTION = (
    "CORRECTION: Your previous response was rejected because it shortened or "
    "omitted content, or because Cantonese wording remained. Convert the entire "
    "input again. Preserve every sentence and detail; do not summarize, merge "
    "away, or omit anything. Replace all Cantonese colloquial forms with "
    "Standard Written Chinese."
)

_CANTONESE_GUIDELINES = prompts.CANTONESE_GUIDELINES
_c2w_system_instruction = prompts.c2w_system_instruction


def _thinking_config(model: str):
    # Register conversion requires syntactic reordering before token generation.
    # Give the model enough reasoning room to plan sentence structures.
    if "3.5" in model or "3.1" in model:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=1024)
    return None


class _OutputTooLong(RuntimeError):
    """Raised when Gemini hits its per-request output token ceiling."""


def _visible_text(text: str, output_format: str) -> str:
    if output_format != "SRT":
        without_headings = re.sub(r"(?m)^#{1,6}\s+.*$", "", text)
        return re.sub(r"\s+", "", without_headings)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(stripped)
    return re.sub(r"\s+", "", "".join(lines))


def _conversion_issues(
    raw_text: str, result_text: str, output_format: str
) -> list[str]:
    issues = []
    source = _visible_text(raw_text, output_format)
    result = _visible_text(result_text, output_format)
    if source and len(result) < len(source) * _REFINE_MIN_LENGTH_RATIO:
        issues.append(
            "the converted text is too short "
            f"({len(result)} vs {len(source)} characters)"
        )
    markers = sum(result.count(marker) for marker in _CANTONESE_MARKERS)
    if markers >= 3:
        issues.append(
            f"the converted text still contains {markers} Cantonese markers"
        )
    return issues


def _response_text(
    response, output_format: str, *, job_id: Optional[str] = None
) -> str:
    usage = response.usage_metadata
    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
    logger.info(
        "Gemini refinement finished: job_id=%s reason=%s prompt_tokens=%s "
        "output_tokens=%s thoughts_tokens=%s",
        _log_job_id(job_id),
        finish_reason,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "thoughts_token_count", None),
    )
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise RuntimeError(f"Gemini blocked the request ({feedback.block_reason}).")
    finish_value = getattr(finish_reason, "value", finish_reason)
    if finish_value == "MAX_TOKENS":
        raise _OutputTooLong(
            "Gemini stopped before completing the refinement (MAX_TOKENS)."
        )
    if finish_value in _BAD_FINISH_REASONS:
        raise RuntimeError(
            f"Gemini stopped before completing the refinement ({finish_value})."
        )
    text = (response.text or "").strip()
    if output_format == "SRT":
        text = re.sub(
            r"^```(?:srt|plaintext|text)?\s*\n",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\n```\s*$", "", text)
    if not text:
        raise RuntimeError("Gemini returned an empty refinement.")
    return text


def _parse_json_payload(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except (TypeError, ValueError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except ValueError:
                continue
    raise ValueError("Gemini did not return valid JSON.")


class _RoundRobinModels:
    """Routes model calls through a round-robin client pool."""

    def __init__(self, pool: "_RoundRobinClient"):
        self._pool = pool

    def generate_content(self, **kwargs):
        return self._pool.request("generate_content", **kwargs)

    def count_tokens(self, **kwargs):
        return self._pool.request("count_tokens", **kwargs)


class _RoundRobinClient:
    """Rotates requests across clients created from multiple API keys."""

    def __init__(self, clients):
        clients = list(clients)
        if not clients:
            raise ValueError("At least one Gemini client is required.")
        self._clients = clients
        self._index = 0
        self._cooldowns = [0.0] * len(clients)
        self._model_cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()
        self.models = _RoundRobinModels(self)

    def _take_locked(self):
        """Returns the next healthy client, advancing the round-robin cursor."""
        now = time.monotonic()
        for offset in range(len(self._clients)):
            index = (self._index + offset) % len(self._clients)
            if self._cooldowns[index] <= now:
                self._index = (index + 1) % len(self._clients)
                return index, self._clients[index]
        return None, None

    def _earliest_wait_locked(self) -> int:
        now = time.monotonic()
        return max(1, int(math.ceil(min(self._cooldowns) - now)))

    def _quota_error_locked(self) -> _QuotaExhausted:
        return _QuotaExhausted(
            "Gemini quota is exhausted for all configured API keys.",
            retry_after=self._earliest_wait_locked(),
        )

    def next(self):
        with self._lock:
            _index, client = self._take_locked()
            if client is None:
                raise self._quota_error_locked()
        return client

    def _mark_exhausted(self, index: int, retry_after: float) -> None:
        cooldown = time.monotonic() + max(1.0, float(retry_after))
        with self._lock:
            self._cooldowns[index] = max(self._cooldowns[index], cooldown)

    def _raise_if_model_cooling_locked(self, model: str) -> None:
        until = self._model_cooldowns.get(model, 0.0)
        wait = until - time.monotonic()
        if wait > 0:
            raise _ModelUnavailable(
                model, retry_after=max(1, int(math.ceil(wait)))
            )

    def _mark_model_unavailable(self, model: str, retry_after: float) -> None:
        cooldown = time.monotonic() + max(1.0, float(retry_after))
        with self._lock:
            self._model_cooldowns[model] = max(
                self._model_cooldowns.get(model, 0.0), cooldown
            )

    def request(self, method_name: str, **kwargs):
        """Calls a model method, cooling the model on 503 and keys on 429."""
        model = kwargs.get("model")
        if model is not None:
            with self._lock:
                self._raise_if_model_cooling_locked(model)
        total = len(self._clients)
        for _attempt in range(total):
            with self._lock:
                if model is not None:
                    self._raise_if_model_cooling_locked(model)
                index, client = self._take_locked()
                if client is None:
                    raise self._quota_error_locked()
            try:
                return getattr(client.models, method_name)(**kwargs)
            except Exception as exc:
                if model is not None and getattr(exc, "code", None) == 503:
                    retry_after = _retry_after_seconds(_exception_text(exc))
                    if retry_after is None or retry_after <= 0:
                        retry_after = float(
                            config.GEMINI_MODEL_COOLDOWN_SECONDS
                        )
                    logger.warning(
                        "Gemini model unavailable; halting requests: "
                        "model=%s retry_after=%.0fs",
                        model,
                        retry_after,
                    )
                    self._mark_model_unavailable(model, retry_after)
                    raise _ModelUnavailable(
                        model, retry_after=retry_after
                    ) from exc
                retry_after = _quota_retry_after(exc)
                if retry_after is None:
                    raise
                logger.warning(
                    "Gemini key quota exhausted; trying another key: "
                    "key_index=%s retry_after=%.0fs",
                    index,
                    retry_after,
                )
                self._mark_exhausted(index, retry_after)
        with self._lock:
            raise self._quota_error_locked()


_client_pool: Optional[_RoundRobinClient] = None
_client_pool_lock = threading.Lock()


def _gemini_client() -> _RoundRobinClient:
    """Returns the shared client pool, creating it on first use."""
    global _client_pool
    if _client_pool is None:
        with _client_pool_lock:
            if _client_pool is None:
                _client_pool = _RoundRobinClient(
                    genai.Client(api_key=key)
                    for key in config.GEMINI_API_KEYS
                )
    return _client_pool


_SOURCE_RESTRICTION = prompts.SOURCE_RESTRICTION


def _generate(
    client,
    model: str,
    system_instruction: Optional[str],
    contents: str,
    *,
    temperature: float,
    automatic_continuation: bool,
    job_id: Optional[str] = None,
):
    instruction = _SOURCE_RESTRICTION
    if system_instruction:
        instruction = f"{_SOURCE_RESTRICTION}\n\n{system_instruction}"
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=instruction,
        thinking_config=_thinking_config(model),
        automatic_continuation=automatic_continuation,
        tools=[],
    )

    def _call():
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        )

    started = time.monotonic()
    try:
        response = _call()
    except Exception as exc:  # noqa: BLE001 - translate quota errors
        quota_retry_after = _quota_retry_after(exc)
        if quota_retry_after is None:
            raise
        logger.warning(
            "Gemini quota/rate limit exhausted: job_id=%s; retry after "
            "%.0fs instead of retrying automatically.",
            _log_job_id(job_id),
            quota_retry_after,
        )
        raise _QuotaExhausted(
            "Gemini quota is exhausted; retry after the quota resets.",
            retry_after=quota_retry_after,
        ) from exc
    logger.info(
        "Gemini API call succeeded: job_id=%s model=%s elapsed=%.2fs",
        _log_job_id(job_id),
        model,
        time.monotonic() - started,
    )
    return response


class _TokenCounter:
    """Counts tokens with Gemini, falling back to a conservative estimate."""

    def __init__(
        self,
        client,
        model: str,
        sample_text: str,
        *,
        job_id: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.job_id = job_id
        self._cache = {}
        self._ratio = None
        self._remote_available = True
        self.prime(sample_text)

    def _remote_count(self, text: str) -> int:
        if not self._remote_available:
            raise RuntimeError("Gemini token counting is unavailable.")
        try:
            result = self.client.models.count_tokens(
                model=self.model, contents=text
            )
            return int(result.total_tokens)
        except Exception as exc:
            self._remote_available = False
            logger.warning(
                "Gemini token counting unavailable: job_id=%s; using "
                "estimates: %s",
                _log_job_id(self.job_id),
                exc,
            )
            raise

    def count(self, text: str) -> int:
        if not text:
            return 0
        if text in self._cache:
            return self._cache[text]
        try:
            tokens = self._remote_count(text)
        except Exception:
            tokens = chunking.estimate_tokens(text)
        self._cache[text] = tokens
        return tokens

    def prime(self, text: str) -> None:
        tokens = self.count(text)
        if text and tokens:
            self._ratio = len(text) / tokens

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        if self._ratio:
            return max(1, int(math.ceil(len(text) / self._ratio)))
        return chunking.estimate_tokens(text)


def _request_refinement(
    client,
    model: str,
    system_instruction: Optional[str],
    contents: str,
    raw_text: str,
    output_format: str,
    cantonese_to_written: bool,
    *,
    automatic_continuation: bool = True,
    job_id: Optional[str] = None,
    cache_store=None,
    validate_result: Optional[Callable[[str], bool]] = None,
) -> str:
    cache_kind = "srt" if output_format == "SRT" else "text"
    settings = None
    if cache_store is not None:
        settings = cache.refinement_part_settings(
            cache_kind,
            model,
            system_instruction,
            contents,
            output_format,
            automatic_continuation,
        )
        cached = _load_intermediate(
            cache_store, settings, job_id=job_id
        )
        if cached is not None:
            if validate_result is None or validate_result(cached):
                return cached
            logger.warning(
                "Ignoring invalid cached intermediate response: "
                "job_id=%s kind=%s",
                _log_job_id(job_id),
                cache_kind,
            )

    issues: list[str] = []
    for attempt in range(2 if cantonese_to_written else 1):
        instruction = system_instruction
        if attempt:
            instruction = f"{instruction}\n\n{_CORRECTION_INSTRUCTION}"
        response = _generate(
            client,
            model,
            instruction,
            contents,
            temperature=0.1,
            automatic_continuation=automatic_continuation,
            job_id=job_id,
        )
        result_text = _response_text(response, output_format, job_id=job_id)
        valid = validate_result is None or validate_result(result_text)
        if not cantonese_to_written:
            if valid:
                _save_intermediate(
                    cache_store, settings, result_text, job_id=job_id
                )
            return result_text
        issues = _conversion_issues(raw_text, result_text, output_format)
        if not issues:
            if valid:
                _save_intermediate(
                    cache_store, settings, result_text, job_id=job_id
                )
            return result_text
        logger.warning(
            "Cantonese conversion attempt %s was incomplete: job_id=%s: %s",
            attempt + 1,
            _log_job_id(job_id),
            "; ".join(issues),
        )
    return result_text


def _request_json(
    client,
    model: str,
    system_instruction: str,
    contents: str,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
    cache_kind: str = "json",
):
    settings = None
    if cache_store is not None:
        settings = cache.refinement_part_settings(
            cache_kind,
            model,
            system_instruction,
            contents,
            "JSON",
            False,
        )
        cached = _load_intermediate(
            cache_store, settings, job_id=job_id
        )
        if cached is not None:
            try:
                return _parse_json_payload(cached)
            except ValueError as exc:
                logger.warning(
                    "Ignoring invalid cached intermediate JSON: "
                    "job_id=%s kind=%s: %s",
                    _log_job_id(job_id),
                    cache_kind,
                    exc,
                )

    response = _generate(
        client,
        model,
        system_instruction,
        contents,
        temperature=0.0,
        automatic_continuation=False,
        job_id=job_id,
    )
    text = _response_text(response, "Text", job_id=job_id)
    payload = _parse_json_payload(text)
    _save_intermediate(cache_store, settings, text, job_id=job_id)
    return payload


def _request_text_plan(
    client,
    units,
    lang: str,
    cantonese_to_written: bool,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
):
    numbered_units = [re.sub(r"\s+", " ", unit).strip() for unit in units]
    numbered = "\n".join(
        f"[{index}] {unit}"
        for index, unit in enumerate(numbered_units, start=1)
        if unit
    )
    system_instruction = prompts.text_plan_system_instruction(
        lang, cantonese_to_written
    )
    contents = prompts.text_plan_prompt(numbered)
    payload = _request_json(
        client,
        config.GEMINI_MODEL,
        system_instruction,
        contents,
        **_refine_kwargs(job_id, cache_store),
        cache_kind="text_plan",
    )
    return chunking.parse_text_sections(payload, len(units))


def _request_srt_plan(
    client,
    lang: str,
    candidates,
    target_tokens: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
):
    if not candidates:
        return []
    lines = []
    for candidate in candidates:
        lines.append(
            f"- split after block {candidate.block_index} "
            f"(~{candidate.cumulative_tokens} tokens): "
            f"...{candidate.before} | next: {candidate.after}..."
        )
    system_instruction = prompts.srt_plan_system_instruction()
    contents = prompts.srt_plan_prompt(
        lang, target_tokens, "\n".join(lines)
    )
    payload = _request_json(
        client,
        config.GEMINI_MODEL,
        system_instruction,
        contents,
        **_refine_kwargs(job_id, cache_store),
        cache_kind="srt_plan",
    )
    return chunking.parse_srt_splits(payload, candidates)


def _single_text_refinement(
    client,
    model: str,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    if cantonese_to_written:
        system_instruction = _c2w_system_instruction("Text", char_limit)
        contents = raw_text
    else:
        system_instruction = None
        contents = prompts.text_single_prompt(raw_text, lang)
    return _request_refinement(
        client,
        model,
        system_instruction,
        contents,
        raw_text,
        "Text",
        cantonese_to_written,
        **_refine_kwargs(job_id, cache_store),
    )


def _text_chunk_refinement(
    client,
    model: str,
    chunk,
    part: int,
    total: int,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    if cantonese_to_written:
        system_instruction = (
            _c2w_system_instruction("Text", char_limit)
            + "\n\n"
            + prompts.text_chunk_system_instruction(part, total)
        )
        contents = chunk.prompt_text
    else:
        system_instruction = None
        contents = prompts.text_chunk_prompt(
            chunk.prompt_text,
            lang,
            part,
            total,
            heading_context=chunk.heading_context,
            starts_with_heading=chunk.starts_with_heading,
        )
    return _request_refinement(
        client,
        model,
        system_instruction,
        contents,
        chunk.source_text,
        "Text",
        cantonese_to_written,
        automatic_continuation=False,
        **_refine_kwargs(job_id, cache_store),
    )


def _join_text_outputs(outputs) -> str:
    text = "\n\n".join(part.strip() for part in outputs if part.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _refine_chunks_in_parallel(
    fn: Callable, chunks: Sequence
) -> list:
    """Runs ``fn`` over ``chunks`` concurrently, routing each call through the
    round-robin client pool so the API keys are used in parallel.

    Results are returned in the same order as ``chunks``, so joined output stays
    deterministic. With a single key or a single chunk this degrades to plain
    sequential execution (no thread overhead).
    """
    workers = max(1, min(len(chunks), len(config.GEMINI_API_KEYS)))
    if workers <= 1:
        return [fn(chunk) for chunk in chunks]

    # ``executor.map`` preserves input order and re-raises the first exception,
    # matching the previous sequential generator's error behaviour.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, chunks))


def _refine_text_chunk(
    client,
    model: str,
    chunk,
    part: int,
    total: int,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    counter: _TokenCounter,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    try:
        return _text_chunk_refinement(
            client,
            model,
            chunk,
            part,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )
    except _OutputTooLong:
        pieces = chunking.split_text_chunk_for_retry(
            chunk,
            floor_tokens=_REFINE_MIN_SPLIT_TOKENS,
            estimate=counter.estimate,
            count=counter.estimate,
        )
        if len(pieces) <= 1:
            raise
        logger.warning(
            "Text refinement chunk %s/%s exceeded the output cap: job_id=%s; "
            "splitting into %s parts.",
            part,
            total,
            _log_job_id(job_id),
            len(pieces),
        )
        return _join_text_outputs(
            _refine_text_chunk(
                client,
                model,
                piece,
                part,
                total,
                lang,
                cantonese_to_written,
                char_limit,
                counter,
                **_refine_kwargs(job_id, cache_store),
            )
            for piece in pieces
        )


def _refine_text_with_split(
    client,
    model: str,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    counter: _TokenCounter,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    """Refines a short text, splitting it once the model reports MAX_TOKENS."""
    try:
        return _single_text_refinement(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )
    except _OutputTooLong:
        tokens = counter.estimate(raw_text)
        if tokens <= _REFINE_MIN_SPLIT_TOKENS:
            raise
        target = max(1, tokens // 2)
        chunks = chunking.group_text_sections(
            chunking.split_text_units(raw_text),
            [chunking.Section(1)],
            target,
            target,
            counter.estimate,
            counter.estimate,
        )
        if len(chunks) <= 1:
            raise
        model = _refine_model(cantonese_to_written, chunked=True)
        total = len(chunks)
        logger.warning(
            "Text refinement exceeded the output cap: job_id=%s; splitting "
            "into %s parts.",
            _log_job_id(job_id),
            total,
        )

        def _refine_one(chunk_index_and_chunk):
            index, chunk = chunk_index_and_chunk
            return _refine_text_chunk(
                client,
                model,
                chunk,
                index,
                total,
                lang,
                cantonese_to_written,
                char_limit,
                counter,
                **_refine_kwargs(job_id, cache_store),
            )

        return _join_text_outputs(
            _refine_chunks_in_parallel(
                _refine_one, list(enumerate(chunks, start=1))
            )
        )


def _single_srt_refinement(
    client,
    model: str,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    if cantonese_to_written:
        system_instruction = _c2w_system_instruction("SRT", char_limit)
        contents = raw_text
    else:
        system_instruction = None
        contents = _srt_prompt(raw_text, lang, char_limit)
    return _request_refinement(
        client,
        model,
        system_instruction,
        contents,
        raw_text,
        "SRT",
        cantonese_to_written,
        **_refine_kwargs(job_id, cache_store),
    )


_srt_prompt = prompts.srt_prompt


def _srt_chunk_refinement(
    client,
    model: str,
    chunk,
    part: int,
    total: int,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    if cantonese_to_written:
        system_instruction = (
            _c2w_system_instruction("SRT", char_limit)
            + "\n\n"
            + prompts.srt_chunk_system_instruction(part, total)
        )
        contents = chunk.text
    else:
        system_instruction = None
        contents = _srt_prompt(chunk.text, lang, char_limit, (part, total))
    result = _request_refinement(
        client,
        model,
        system_instruction,
        contents,
        chunk.text,
        "SRT",
        cantonese_to_written,
        automatic_continuation=False,
        validate_result=lambda text: bool(chunking.parse_srt(text)),
        **_refine_kwargs(job_id, cache_store),
    )
    if not chunking.parse_srt(result):
        raise RuntimeError("Gemini returned an invalid SRT refinement chunk.")
    return result


def _refine_srt_chunk(
    client,
    model: str,
    chunk,
    part: int,
    total: int,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    try:
        return _srt_chunk_refinement(
            client,
            model,
            chunk,
            part,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )
    except _OutputTooLong:
        pieces = chunking.split_srt_chunk_for_retry(
            chunk, chunking.estimate_tokens
        )
        if len(pieces) <= 1:
            raise
        logger.warning(
            "SRT refinement chunk %s/%s exceeded the output cap: job_id=%s; "
            "splitting into %s parts.",
            part,
            total,
            _log_job_id(job_id),
            len(pieces),
        )
        return chunking.merge_srt_texts(
            _refine_srt_chunk(
                client,
                model,
                piece,
                part,
                total,
                lang,
                cantonese_to_written,
                char_limit,
                **_refine_kwargs(job_id, cache_store),
            )
            for piece in pieces
        )


def _refine_srt_with_split(
    client,
    model: str,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    counter: _TokenCounter,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    """Refines a short SRT, splitting it once the model reports MAX_TOKENS."""
    try:
        return _single_srt_refinement(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )
    except _OutputTooLong:
        tokens = counter.estimate(raw_text)
        if tokens <= _REFINE_MIN_SPLIT_TOKENS:
            raise
        blocks = chunking.parse_srt(raw_text)
        if len(blocks) < 2:
            raise
        target = max(1, tokens // 2)
        chunks = chunking.build_srt_chunks(
            blocks,
            [],
            target,
            target,
            counter.estimate,
            counter.estimate,
        )
        if len(chunks) <= 1:
            chunks = chunking.split_srt_chunk_for_retry(
                chunking.SrtChunk(
                    blocks=blocks, text=chunking.render_srt(blocks)
                ),
                counter.estimate,
            )
        if len(chunks) <= 1:
            raise
        model = _refine_model(cantonese_to_written, chunked=True)
        total = len(chunks)
        logger.warning(
            "SRT refinement exceeded the output cap: job_id=%s; splitting "
            "into %s parts.",
            _log_job_id(job_id),
            total,
        )

        def _refine_one(chunk_index_and_chunk):
            index, chunk = chunk_index_and_chunk
            return _refine_srt_chunk(
                client,
                model,
                chunk,
                index,
                total,
                lang,
                cantonese_to_written,
                char_limit,
                **_refine_kwargs(job_id, cache_store),
            )

        return chunking.merge_srt_texts(
            _refine_chunks_in_parallel(
                _refine_one, list(enumerate(chunks, start=1))
            )
        )


def _refine_model(cantonese_to_written: bool, *, chunked: bool = False) -> str:
    """Selects the Gemini model used for an AI refinement call."""
    if cantonese_to_written:
        return config.GEMINI_C2W_MODEL
    return config.GEMINI_CHUNK_MODEL if chunked else config.GEMINI_MODEL


def _refine_text(
    client,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    counter: _TokenCounter,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    model = _refine_model(cantonese_to_written)
    total_tokens = counter.count(raw_text)
    if total_tokens < config.REFINE_CHUNK_TOKENS:
        return _refine_text_with_split(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )

    units = chunking.split_text_units(raw_text)
    try:
        sections = _request_text_plan(
            client,
            units,
            lang,
            cantonese_to_written,
            **_refine_kwargs(job_id, cache_store),
        )
    except Exception as exc:
        logger.warning(
            "Text structure planning failed: job_id=%s; using sentence-based "
            "fallback: %s",
            _log_job_id(job_id),
            exc,
        )
        sections = [chunking.Section(1)]
    chunks = chunking.group_text_sections(
        units,
        sections,
        config.REFINE_CHUNK_TOKENS,
        config.REFINE_CHUNK_HARD_TOKENS,
        counter.estimate,
        counter.estimate,
    )
    if not chunks:
        return _refine_text_with_split(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )
    model = _refine_model(cantonese_to_written, chunked=True)
    total = len(chunks)
    logger.info(
        "Text refinement using %s: job_id=%s, %s tokens, %s chunks, model=%s",
        "Cantonese-to-written" if cantonese_to_written else lang,
        _log_job_id(job_id),
        total_tokens,
        total,
        model,
    )
    def _refine_one(chunk_index_and_chunk):
        index, chunk = chunk_index_and_chunk
        return _refine_text_chunk(
            client,
            model,
            chunk,
            index,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )

    return _join_text_outputs(
        _refine_chunks_in_parallel(
            _refine_one, list(enumerate(chunks, start=1))
        )
    )


def _refine_srt(
    client,
    raw_text: str,
    lang: str,
    cantonese_to_written: bool,
    char_limit: int,
    counter: _TokenCounter,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    model = _refine_model(cantonese_to_written)
    total_tokens = counter.count(raw_text)
    if total_tokens < config.REFINE_CHUNK_TOKENS:
        return _refine_srt_with_split(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )

    blocks = chunking.parse_srt(raw_text)
    if len(blocks) < 2:
        return _single_srt_refinement(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )

    candidates = chunking.build_srt_candidates(blocks, counter.estimate)
    try:
        split_after = _request_srt_plan(
            client,
            lang,
            candidates,
            config.REFINE_CHUNK_TOKENS,
            **_refine_kwargs(job_id, cache_store),
        )
    except Exception as exc:
        logger.warning(
            "SRT split planning failed: job_id=%s; using sentence-safe "
            "fallback: %s",
            _log_job_id(job_id),
            exc,
        )
        split_after = []
    chunks = chunking.build_srt_chunks(
        blocks,
        split_after,
        config.REFINE_CHUNK_TOKENS,
        config.REFINE_CHUNK_HARD_TOKENS,
        counter.estimate,
        counter.estimate,
    )
    if not chunks:
        return _refine_srt_with_split(
            client,
            model,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )
    model = _refine_model(cantonese_to_written, chunked=True)
    total = len(chunks)
    logger.info(
        "SRT refinement using %s: job_id=%s, %s tokens, %s chunks, model=%s",
        "Cantonese-to-written" if cantonese_to_written else lang,
        _log_job_id(job_id),
        total_tokens,
        total,
        model,
    )
    def _refine_one(chunk_index_and_chunk):
        index, chunk = chunk_index_and_chunk
        return _refine_srt_chunk(
            client,
            model,
            chunk,
            index,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            **_refine_kwargs(job_id, cache_store),
        )

    return chunking.merge_srt_texts(
        _refine_chunks_in_parallel(
            _refine_one, list(enumerate(chunks, start=1))
        )
    )


def refine(
    raw_text: str,
    lang: str,
    output_format: str,
    cantonese_to_written: bool,
    srt_char_limit: int = 30,
    *,
    job_id: Optional[str] = None,
    cache_store=None,
) -> str:
    if not raw_text or not raw_text.strip():
        raise ValueError("No transcription found.")
    if not config.GEMINI_API_KEYS:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = _gemini_client()
    char_limit = int(srt_char_limit) if srt_char_limit else 30
    counter = _TokenCounter(
        client, config.GEMINI_MODEL, raw_text, job_id=job_id
    )

    if output_format == "SRT":
        return _refine_srt(
            client,
            raw_text,
            lang,
            cantonese_to_written,
            char_limit,
            counter,
            **_refine_kwargs(job_id, cache_store),
        )
    return _refine_text(
        client,
        raw_text,
        lang,
        cantonese_to_written,
        char_limit,
        counter,
        **_refine_kwargs(job_id, cache_store),
    )
