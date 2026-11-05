"""Unit tests for the encrypted transcription cache."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from cache import (
    CacheStore,
    copy_and_hash,
    hash_file,
    md5_text,
    refinement_part_settings,
    sha256_text,
)


class CacheStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.store = CacheStore(root / "raw", root / "refine")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_raw_cache_roundtrip_and_filename(self):
        key = hashlib.sha256(b"media bytes").digest()
        md5_hex = hashlib.md5(b"media bytes").hexdigest()

        path = self.store.save_raw(
            md5_hex, "Cantonese", "SRT", key, "未轉換的文字"
        )
        self.assertEqual(path.name, f"{md5_hex}_cantonese_srt.enc")
        self.assertNotIn("未轉換的文字".encode("utf-8"), path.read_bytes())

        payload = self.store.load_raw(md5_hex, "Cantonese", "SRT", key)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["text"], "未轉換的文字")
        self.assertEqual(
            payload["settings"], {"language": "Cantonese", "format": "SRT"}
        )

    def test_raw_cache_rejects_wrong_key_and_language(self):
        key = hashlib.sha256(b"media").digest()
        md5_hex = hashlib.md5(b"media").hexdigest()
        self.store.save_raw(md5_hex, "Mandarin", "Text", key, "text")

        wrong_key = hashlib.sha256(b"other").digest()
        self.assertIsNone(
            self.store.load_raw(md5_hex, "Mandarin", "Text", wrong_key)
        )
        self.assertIsNone(
            self.store.load_raw(md5_hex, "Cantonese", "Text", key)
        )

    def test_refinement_cache_depends_on_all_settings(self):
        raw_text = "raw transcript"
        base = dict(
            lang="Cantonese",
            fmt="SRT",
            zh_format="Traditional Chinese",
            cantonese_to_written=True,
            srt_char_limit=30,
        )
        path = self.store.save_refinement(raw_text, refined_text="refined", **base)
        self.assertIn(md5_text(raw_text), path.name)
        self.assertNotIn("trad", path.name)
        self.assertNotIn("simp", path.name)
        self.assertIn("_1_30.enc", path.name)

        payload = self.store.load_refinement(raw_text, **base)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["text"], "refined")

        changed = dict(base, cantonese_to_written=False)
        self.assertIsNone(self.store.load_refinement(raw_text, **changed))

        changed = dict(base, srt_char_limit=40)
        self.assertIsNone(self.store.load_refinement(raw_text, **changed))

        changed = dict(base, zh_format="Simplified Chinese")
        self.assertIsNone(self.store.load_refinement(raw_text, **changed))

    def test_refinement_cache_is_keyed_by_converted_text(self):
        base = dict(
            lang="Mandarin",
            fmt="Text",
            zh_format="Traditional Chinese",
            cantonese_to_written=False,
            srt_char_limit=None,
        )
        converted = "轉換後文字"
        unconverted = "转换后文字"

        self.store.save_refinement(converted, refined_text="refined", **base)
        self.assertIsNotNone(self.store.load_refinement(converted, **base))
        self.assertIsNone(self.store.load_refinement(unconverted, **base))

    def test_refinement_cache_version_invalidates_old_payloads(self):
        import cache

        raw_text = "raw transcript"
        base = dict(
            lang="Cantonese",
            fmt="SRT",
            zh_format="Traditional Chinese",
            cantonese_to_written=True,
            srt_char_limit=30,
        )
        original = cache.REFINE_CACHE_VERSION
        try:
            cache.REFINE_CACHE_VERSION = original + 1
            self.store.save_refinement(
                raw_text, refined_text="old", **base
            )
            cache.REFINE_CACHE_VERSION = original
            self.assertIsNone(self.store.load_refinement(raw_text, **base))
        finally:
            cache.REFINE_CACHE_VERSION = original

    def test_corrupt_cache_is_treated_as_miss(self):
        key = hashlib.sha256(b"media").digest()
        md5_hex = hashlib.md5(b"media").hexdigest()
        path = self.store.raw_path(md5_hex, "English", "Text")
        self.assertEqual(path.name, f"{md5_hex}_english_txt.enc")
        path.write_bytes(b"not a valid encrypted payload")
        self.assertIsNone(
            self.store.load_raw(md5_hex, "English", "Text", key)
        )

    def test_hash_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"chunked hashing")
            path = handle.name
        try:
            md5_hex, digest = hash_file(path)
            self.assertEqual(md5_hex, hashlib.md5(b"chunked hashing").hexdigest())
            self.assertEqual(digest, hashlib.sha256(b"chunked hashing").digest())
        finally:
            os.remove(path)

    def test_copy_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            destination = Path(directory) / "destination.bin"
            payload = b"streamed copy and hash"
            source.write_bytes(payload)

            md5_hex, digest = copy_and_hash(str(source), str(destination))

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(md5_hex, hashlib.md5(payload).hexdigest())
            self.assertEqual(digest, hashlib.sha256(payload).digest())

    def test_refinement_key_uses_raw_text(self):
        self.assertEqual(
            sha256_text("raw"), hashlib.sha256(b"raw").digest()
        )
        self.assertEqual(md5_text("raw"), hashlib.md5(b"raw").hexdigest())

    def test_refinement_part_cache_roundtrip_and_isolation(self):
        settings = refinement_part_settings(
            "text",
            "model-a",
            "instruction",
            "chunk text",
            "Text",
            False,
        )
        path = self.store.save_refinement_part(settings, "refined chunk")

        self.assertEqual(path.parent, self.store.part_dir)
        self.assertTrue(path.name.startswith("text_"))
        self.assertNotIn("chunk text".encode("utf-8"), path.read_bytes())
        payload = self.store.load_refinement_part(settings)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["text"], "refined chunk")

        variants = [
            refinement_part_settings(
                "srt", "model-a", "instruction", "chunk text", "Text", False
            ),
            refinement_part_settings(
                "text", "model-b", "instruction", "chunk text", "Text", False
            ),
            refinement_part_settings(
                "text", "model-a", "other", "chunk text", "Text", False
            ),
            refinement_part_settings(
                "text", "model-a", "instruction", "other text", "Text", False
            ),
            refinement_part_settings(
                "text", "model-a", "instruction", "chunk text", "SRT", False
            ),
            refinement_part_settings(
                "text", "model-a", "instruction", "chunk text", "Text", True
            ),
        ]
        for variant in variants:
            self.assertIsNone(self.store.load_refinement_part(variant))

    def test_refinement_part_cache_version_invalidates_old_payloads(self):
        import cache

        settings = refinement_part_settings(
            "text",
            "model-a",
            "instruction",
            "chunk text",
            "Text",
            False,
        )
        original = cache.REFINE_PART_CACHE_VERSION
        try:
            self.store.save_refinement_part(settings, "old")
            cache.REFINE_PART_CACHE_VERSION = original + 1
            updated = refinement_part_settings(
                "text",
                "model-a",
                "instruction",
                "chunk text",
                "Text",
                False,
            )
            self.assertIsNone(self.store.load_refinement_part(updated))
        finally:
            cache.REFINE_PART_CACHE_VERSION = original


if __name__ == "__main__":
    unittest.main()
