"""Encrypted transcription and refinement cache.

Cache files use AES-GCM. The first 12 bytes are the nonce, the remainder is the
ciphertext with its authentication tag. Raw transcription caches are keyed by the
SHA-256 of the uploaded media; refinement caches are keyed by the SHA-256 of the
OpenCC-converted transcription text.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

NONCE_BYTES = 12
REFINE_CACHE_VERSION = 3
REFINE_PART_CACHE_VERSION = 1


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def hash_file(path: str) -> Tuple[str, bytes]:
    """Returns (md5 hex, sha256 digest) while reading the file once in chunks."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.digest()


def copy_and_hash(src: str, dst: str) -> Tuple[str, bytes]:
    """Copies ``src`` to ``dst`` and hashes it in the same pass."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(src, "rb") as source, open(dst, "wb") as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(chunk)
            sha.update(chunk)
            destination.write(chunk)
    return md5.hexdigest(), sha.digest()


def _slug(value) -> str:
    """Makes an arbitrary setting safe for use inside a cache filename."""
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "none"


def raw_settings(lang: str, fmt: str) -> Dict[str, str]:
    return {"language": lang, "format": fmt}


def refine_settings(
    lang: str,
    fmt: str,
    zh_format: Optional[str],
    cantonese_to_written: bool,
    srt_char_limit: Optional[int],
) -> Dict[str, object]:
    return {
        "language": lang,
        "format": fmt,
        "zh_format": zh_format,
        "cantonese_to_written": bool(cantonese_to_written),
        "srt_char_limit": srt_char_limit,
        "refinement_version": REFINE_CACHE_VERSION,
    }


def refinement_part_settings(
    kind: str,
    model: str,
    system_instruction: Optional[str],
    contents: str,
    output_format: str,
    automatic_continuation: bool,
) -> Dict[str, object]:
    """Builds the stable key for one intermediate Gemini response."""
    return {
        "kind": kind,
        "model": model,
        "instruction_sha256": hashlib.sha256(
            (system_instruction or "").encode("utf-8")
        ).hexdigest(),
        "contents_sha256": hashlib.sha256(
            contents.encode("utf-8")
        ).hexdigest(),
        "output_format": output_format,
        "automatic_continuation": bool(automatic_continuation),
        "refinement_part_version": REFINE_PART_CACHE_VERSION,
    }


def _part_key(settings: Dict) -> Tuple[str, bytes]:
    data = json.dumps(
        settings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(data).digest()
    return digest.hex(), digest


def _fmt_token(fmt: str) -> str:
    return "srt" if fmt == "SRT" else "txt"


class CacheStore:
    def __init__(
        self,
        raw_dir: Path,
        refine_dir: Path,
        part_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.refine_dir = Path(refine_dir)
        self.part_dir = (
            Path(part_dir)
            if part_dir is not None
            else self.refine_dir / "parts"
        )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.refine_dir.mkdir(parents=True, exist_ok=True)
        self.part_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Paths -----------------
    def raw_path(self, md5_hex: str, lang: str, fmt: str) -> Path:
        return self.raw_dir / f"{md5_hex}_{_slug(lang)}_{_fmt_token(fmt)}.enc"

    def refine_path(
        self,
        display_md5: str,
        lang: str,
        fmt: str,
        cantonese_to_written: bool,
        srt_char_limit: Optional[int],
    ) -> Path:
        char_token = str(srt_char_limit) if srt_char_limit is not None else "none"
        c2w_token = "1" if cantonese_to_written else "0"
        name = (
            f"{display_md5}_{_slug(lang)}_{_fmt_token(fmt)}_"
            f"{c2w_token}_{char_token}.enc"
        )
        return self.refine_dir / name

    # ----------------- Encryption -----------------
    @staticmethod
    def _encrypt(key: bytes, payload: Dict) -> bytes:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        nonce = os.urandom(NONCE_BYTES)
        return nonce + AESGCM(key).encrypt(nonce, data, None)

    @staticmethod
    def _decrypt(key: bytes, blob: bytes) -> Dict:
        nonce, ciphertext = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
        data = AESGCM(key).decrypt(nonce, ciphertext, None)
        return json.loads(data.decode("utf-8"))

    @staticmethod
    def _write(path: Path, blob: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def _load(self, path: Path, key: bytes) -> Optional[Dict]:
        try:
            with open(path, "rb") as handle:
                return self._decrypt(key, handle.read())
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Ignoring unreadable cache %s: %s", path.name, exc)
            return None

    # ----------------- Raw transcription -----------------
    def save_raw(
        self,
        md5_hex: str,
        lang: str,
        fmt: str,
        key: bytes,
        text: str,
    ) -> Path:
        path = self.raw_path(md5_hex, lang, fmt)
        payload = {"settings": raw_settings(lang, fmt), "text": text}
        self._write(path, self._encrypt(key, payload))
        return path

    def load_raw(
        self, md5_hex: str, lang: str, fmt: str, key: bytes
    ) -> Optional[Dict]:
        path = self.raw_path(md5_hex, lang, fmt)
        payload = self._load(path, key)
        if payload and payload.get("settings") == raw_settings(lang, fmt):
            return payload
        return None

    # ----------------- AI refinement -----------------
    def save_refinement(
        self,
        display_text: str,
        lang: str,
        fmt: str,
        zh_format: Optional[str],
        cantonese_to_written: bool,
        srt_char_limit: Optional[int],
        refined_text: str,
    ) -> Path:
        display_md5 = md5_text(display_text)
        path = self.refine_path(
            display_md5, lang, fmt, cantonese_to_written, srt_char_limit
        )
        payload = {
            "settings": refine_settings(
                lang, fmt, zh_format, cantonese_to_written, srt_char_limit
            ),
            "text": refined_text,
        }
        self._write(path, self._encrypt(sha256_text(display_text), payload))
        return path

    def load_refinement(
        self,
        display_text: str,
        lang: str,
        fmt: str,
        zh_format: Optional[str],
        cantonese_to_written: bool,
        srt_char_limit: Optional[int],
    ) -> Optional[Dict]:
        display_md5 = md5_text(display_text)
        path = self.refine_path(
            display_md5, lang, fmt, cantonese_to_written, srt_char_limit
        )
        payload = self._load(path, sha256_text(display_text))
        settings = refine_settings(
            lang, fmt, zh_format, cantonese_to_written, srt_char_limit
        )
        if payload and payload.get("settings") == settings:
            return payload
        return None

    # ----------------- Intermediate AI refinement -----------------
    def refinement_part_path(self, settings: Dict) -> Path:
        token, _key = _part_key(settings)
        return self.part_dir / f"{_slug(settings.get('kind', 'part'))}_{token}.enc"

    def save_refinement_part(self, settings: Dict, text: str) -> Path:
        path = self.refinement_part_path(settings)
        _token, key = _part_key(settings)
        payload = {"settings": settings, "text": text}
        self._write(path, self._encrypt(key, payload))
        return path

    def load_refinement_part(self, settings: Dict) -> Optional[Dict]:
        path = self.refinement_part_path(settings)
        _token, key = _part_key(settings)
        payload = self._load(path, key)
        if payload and payload.get("settings") == settings:
            return payload
        return None
