"""Tests for the exponential-backoff retry logic in core.py.

The real core module imports torch/funasr/google-genai, which are heavy. We
stub those dependencies into ``sys.modules`` with lightweight fakes so the
retry path can be exercised directly and quickly.
"""

import logging
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401


def _install_stubs():
    """Install fake heavy dependencies ahead of importing ``core``."""

    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None

    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = object

    funasr = types.ModuleType("funasr")
    funasr.AutoModel = object

    opencc = types.ModuleType("opencc")
    opencc.OpenCC = object

    genai = types.ModuleType("google.genai")
    google = types.ModuleType("google")
    google.genai = genai

    types_mod = types.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = lambda **kwargs: kwargs
    types_mod.ThinkingConfig = lambda **kwargs: kwargs
    types_mod.ThinkingLevel = types.SimpleNamespace(LOW="LOW")

    sys.modules.update(
        {
            "torch": torch,
            "faster_whisper": faster_whisper,
            "funasr": funasr,
            "opencc": opencc,
            "google": google,
            "google.genai": genai,
            "google.genai.types": types_mod,
        }
    )
    return genai, types_mod


_install_stubs()

# Ensure google.genai.types is registered as a submodule reference.
import google.genai  # noqa: E402
import google.genai.types  # noqa: E402

import core  # noqa: E402
from cache import CacheStore  # noqa: E402


def _response(text, finish_reason="STOP"):
    return types.SimpleNamespace(
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=0,
        ),
        candidates=[types.SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=None,
        text=text,
    )


class ModelLoadingTests(unittest.TestCase):
    def test_funasr_progress_bars_are_disabled_everywhere(self):
        with (
            mock.patch.object(core, "WhisperModel", return_value=mock.Mock()),
            mock.patch.object(core, "AutoModel", return_value=mock.Mock()) as auto_model,
            mock.patch.object(core, "_whisper_model", None),
            mock.patch.object(core, "_funasr_model", None),
        ):
            core._load_models()

        kwargs = auto_model.call_args.kwargs
        self.assertTrue(kwargs["disable_pbar"])
        self.assertTrue(kwargs["vad_kwargs"]["disable_pbar"])
        self.assertTrue(kwargs["punc_kwargs"]["disable_pbar"])


class QuietLoggingTests(unittest.TestCase):
    def test_funasr_info_is_dropped_but_warnings_and_app_logs_survive(self):
        log_filter = core.config._ThirdPartyInfoFilter()

        def record(level, pathname):
            return logging.LogRecord(
                "root", level, pathname, 1, "message", None, None
            )

        self.assertFalse(
            log_filter.filter(
                record(logging.INFO, "/site-packages/funasr/auto/auto_model.py")
            )
        )
        self.assertFalse(
            log_filter.filter(
                record(logging.INFO, "/site-packages/google/genai/models.py")
            )
        )
        self.assertTrue(
            log_filter.filter(
                record(logging.WARNING, "/site-packages/funasr/auto/auto_model.py")
            )
        )
        self.assertTrue(
            log_filter.filter(
                record(logging.INFO, "/workspace/transcribe-bot/core.py")
            )
        )


class _FakeAPIError(Exception):
    def __init__(self, code, *, payload=None, message=None):
        super().__init__(message or f"{code} status")
        self.code = code
        if payload is not None:
            self.response_json = payload


def _daily_quota_error(retry_delay=None):
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaId": (
                        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                    )
                }
            ],
        }
    ]
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return _FakeAPIError(
        429,
        payload={
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": (
                    "Quota exceeded for quota metric "
                    "'Generate requests per day per project per model'."
                ),
                "details": details,
            }
        },
    )


def _unavailable_error(retry_delay=None):
    payload = {
        "error": {
            "code": 503,
            "status": "UNAVAILABLE",
            "message": (
                "This model is currently experiencing high demand. "
                "Spikes in demand are usually temporary."
            ),
        }
    }
    if retry_delay is not None:
        payload["error"]["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        ]
    return _FakeAPIError(503, payload=payload)


class GenerateTests(unittest.TestCase):
    def _client(self, side_effect):
        client = mock.Mock()
        client.models.generate_content.side_effect = side_effect
        return client

    def test_generate_logs_include_job_id(self):
        client = self._client([mock.Mock(text="ok")])
        with self.assertLogs("core", level="INFO") as logs:
            core._generate(
                client,
                "gemini",
                None,
                "contents",
                temperature=0.1,
                automatic_continuation=True,
                job_id="job-123",
            )

        output = "\n".join(logs.output)
        self.assertIn("Gemini API call succeeded: job_id=job-123", output)

    def test_generate_does_not_retry_503(self):
        client = self._client([_FakeAPIError(503), mock.Mock(text="ok")])
        with mock.patch.object(core.time, "sleep") as sleep:
            with self.assertRaises(_FakeAPIError):
                core._generate(
                    client,
                    "gemini",
                    None,
                    "contents",
                    temperature=0.1,
                    automatic_continuation=False,
                )
        self.assertEqual(client.models.generate_content.call_count, 1)
        sleep.assert_not_called()

    def test_reraises_non_retryable_without_sleeping(self):
        client = self._client([RuntimeError("boom")])
        with mock.patch.object(core.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                core._generate(
                    client,
                    "gemini",
                    None,
                    "contents",
                    temperature=0.1,
                    automatic_continuation=False,
                )
        sleep.assert_not_called()


class QuotaHandlingTests(unittest.TestCase):
    def test_daily_quota_is_classified_from_payload(self):
        self.assertEqual(
            core._quota_retry_after(_daily_quota_error("120s")), 120.0
        )

    def test_daily_quota_without_retry_info_uses_configured_cooldown(self):
        with mock.patch.object(
            core.config, "GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS", 900
        ):
            self.assertEqual(
                core._quota_retry_after(_daily_quota_error()), 900.0
            )

    def test_any_429_is_treated_as_quota_cooldown(self):
        error = _FakeAPIError(
            429,
            message=(
                "429 RESOURCE_EXHAUSTED: rate limit exceeded; "
                "retryDelay: 5s"
            ),
        )
        self.assertEqual(core._quota_retry_after(error), 5.0)

    def test_plain_429_without_retry_info_uses_general_cooldown(self):
        with mock.patch.object(core.config, "GEMINI_QUOTA_COOLDOWN_SECONDS", 60):
            self.assertEqual(core._quota_retry_after(_FakeAPIError(429)), 60.0)

    def test_generate_raises_quota_error_for_plain_429_without_sleeping(self):
        client = mock.Mock()
        client.models.generate_content.side_effect = _FakeAPIError(429)
        with (
            mock.patch.object(core.time, "sleep") as sleep,
            mock.patch.object(core.config, "GEMINI_QUOTA_COOLDOWN_SECONDS", 60),
        ):
            with self.assertRaises(core._QuotaExhausted) as raised:
                core._generate(
                    client,
                    "gemini",
                    None,
                    "contents",
                    temperature=0.1,
                    automatic_continuation=False,
                    job_id="quota-job",
                )
        self.assertEqual(raised.exception.retry_after, 60.0)
        sleep.assert_not_called()

    def test_generate_raises_quota_error_without_sleeping(self):
        client = mock.Mock()
        client.models.generate_content.side_effect = _daily_quota_error("60s")
        with mock.patch.object(core.time, "sleep") as sleep:
            with self.assertRaises(core._QuotaExhausted) as raised:
                core._generate(
                    client,
                    "gemini",
                    None,
                    "contents",
                    temperature=0.1,
                    automatic_continuation=False,
                    job_id="quota-job",
                )
        self.assertEqual(raised.exception.retry_after, 60.0)
        sleep.assert_not_called()

    def test_pool_fails_over_to_next_key(self):
        exhausted = mock.Mock()
        exhausted.models.generate_content.side_effect = _daily_quota_error("60s")
        healthy = mock.Mock()
        healthy.models.generate_content.return_value = mock.Mock(text="ok")
        pool = core._RoundRobinClient([exhausted, healthy])

        response = pool.models.generate_content(model="gemini", contents="x")

        self.assertEqual(response.text, "ok")
        self.assertEqual(
            exhausted.models.generate_content.call_count, 1
        )
        self.assertEqual(healthy.models.generate_content.call_count, 1)

        # The exhausted key remains on cooldown for subsequent requests.
        response = pool.models.generate_content(model="gemini", contents="y")
        self.assertEqual(response.text, "ok")
        self.assertEqual(
            exhausted.models.generate_content.call_count, 1
        )
        self.assertEqual(healthy.models.generate_content.call_count, 2)

    def test_pool_reports_quota_error_when_all_keys_are_exhausted(self):
        clients = [mock.Mock(), mock.Mock()]
        for client in clients:
            client.models.generate_content.side_effect = _daily_quota_error(
                "60s"
            )
        pool = core._RoundRobinClient(clients)

        with self.assertRaises(core._QuotaExhausted) as raised:
            pool.models.generate_content(model="gemini", contents="x")

        self.assertGreater(raised.exception.retry_after or 0, 0)
        self.assertEqual(
            clients[0].models.generate_content.call_count, 1
        )
        self.assertEqual(
            clients[1].models.generate_content.call_count, 1
        )


class ModelCooldownTests(unittest.TestCase):
    def _client(self, *, side_effect=None, result=None):
        client = mock.Mock()
        if side_effect is not None:
            client.models.generate_content.side_effect = side_effect
        if result is not None:
            client.models.generate_content.return_value = result
        return client

    def test_503_marks_model_and_halts_following_requests(self):
        failing = self._client(side_effect=_FakeAPIError(503))
        healthy = self._client(result=mock.Mock(text="ok"))
        pool = core._RoundRobinClient([failing, healthy])

        with mock.patch.object(
            core.config, "GEMINI_MODEL_COOLDOWN_SECONDS", 60
        ):
            with self.assertRaises(core._ModelUnavailable) as raised:
                pool.models.generate_content(model="gemini-3.8-flash", contents="x")
            self.assertEqual(raised.exception.retry_after, 60.0)

            # A 503 is model-wide, so the other key must not be tried.
            self.assertEqual(failing.models.generate_content.call_count, 1)
            self.assertEqual(healthy.models.generate_content.call_count, 0)

            # Later requests for the same model are halted before dispatch.
            with self.assertRaises(core._ModelUnavailable):
                pool.models.generate_content(
                    model="gemini-3.8-flash", contents="y"
                )

        self.assertEqual(failing.models.generate_content.call_count, 1)
        self.assertEqual(healthy.models.generate_content.call_count, 0)

    def test_503_uses_retry_info(self):
        client = self._client(side_effect=_unavailable_error("15s"))
        pool = core._RoundRobinClient([client])

        with self.assertRaises(core._ModelUnavailable) as raised:
            pool.models.generate_content(model="gemini", contents="x")

        self.assertEqual(raised.exception.retry_after, 15.0)

    def test_503_without_retry_info_uses_configured_cooldown(self):
        client = self._client(side_effect=_FakeAPIError(503))
        pool = core._RoundRobinClient([client])

        with mock.patch.object(
            core.config, "GEMINI_MODEL_COOLDOWN_SECONDS", 45
        ):
            with self.assertRaises(core._ModelUnavailable) as raised:
                pool.models.generate_content(model="gemini", contents="x")

        self.assertEqual(raised.exception.retry_after, 45.0)

    def test_model_cooldown_is_scoped_to_the_model(self):
        failing = self._client(side_effect=_FakeAPIError(503))
        healthy = self._client(result=mock.Mock(text="ok"))
        pool = core._RoundRobinClient([failing, healthy])

        with self.assertRaises(core._ModelUnavailable):
            pool.models.generate_content(model="bad-model", contents="x")

        response = pool.models.generate_content(model="good-model", contents="y")
        self.assertEqual(response.text, "ok")

    def test_model_cooldown_expires(self):
        client = self._client(result=mock.Mock(text="ok"))
        pool = core._RoundRobinClient([client])
        pool._model_cooldowns["gemini"] = core.time.monotonic() - 1

        response = pool.models.generate_content(model="gemini", contents="x")
        self.assertEqual(response.text, "ok")

    def test_count_tokens_honours_model_cooldown(self):
        client = self._client(side_effect=_FakeAPIError(503))
        client.models.count_tokens.return_value = mock.Mock(total_tokens=7)
        pool = core._RoundRobinClient([client])

        with self.assertRaises(core._ModelUnavailable):
            pool.models.generate_content(model="gemini", contents="x")
        with self.assertRaises(core._ModelUnavailable):
            pool.models.count_tokens(model="gemini", contents="x")

        self.assertEqual(client.models.count_tokens.call_count, 0)


class SourceRestrictionTests(unittest.TestCase):
    def _generate(self, system_instruction):
        client = mock.Mock()
        client.models.generate_content.return_value = mock.Mock(text="ok")
        core._generate(
            client,
            "gemini",
            system_instruction,
            "contents",
            temperature=0.1,
            automatic_continuation=False,
        )
        return client.models.generate_content.call_args.kwargs["config"]

    def test_generate_disables_tools_and_restricts_sources(self):
        config = self._generate("You are an editor.")

        self.assertEqual(config["tools"], [])
        self.assertIn(core._SOURCE_RESTRICTION, config["system_instruction"])
        self.assertIn("You are an editor.", config["system_instruction"])

    def test_generate_adds_source_restriction_without_system_instruction(self):
        config = self._generate(None)

        self.assertEqual(config["tools"], [])
        self.assertEqual(
            config["system_instruction"], core._SOURCE_RESTRICTION
        )


class ModelSelectionTests(unittest.TestCase):
    def test_c2w_uses_dedicated_model(self):
        self.assertEqual(
            core._refine_model(True), core.config.GEMINI_C2W_MODEL
        )
        self.assertEqual(
            core._refine_model(True, chunked=True),
            core.config.GEMINI_C2W_MODEL,
        )

    def test_non_c2w_uses_standard_models(self):
        self.assertEqual(
            core._refine_model(False), core.config.GEMINI_MODEL
        )
        self.assertEqual(
            core._refine_model(False, chunked=True),
            core.config.GEMINI_CHUNK_MODEL,
        )


class ParallelChunkTests(unittest.TestCase):
    """The round-robin client is shared across chunk workers, so parallel
    refinement must preserve input order while actually overlapping calls."""

    def test_preserves_input_order(self):
        chunks = ["zero", "one", "two", "three", "four"]
        with mock.patch.object(core.config, "GEMINI_API_KEYS", ["a", "b", "c"]):
            result = core._refine_chunks_in_parallel(lambda c: c.upper(), chunks)
        self.assertEqual(result, ["ZERO", "ONE", "TWO", "THREE", "FOUR"])

    def test_single_key_degrades_to_sequential(self):
        chunks = ["a", "b", "c"]
        calls = []
        with mock.patch.object(core.config, "GEMINI_API_KEYS", ["only"]):
            result = core._refine_chunks_in_parallel(
                lambda c: calls.append(c) or c, chunks
            )
        self.assertEqual(result, ["a", "b", "c"])
        self.assertEqual(calls, ["a", "b", "c"])

    def test_runs_concurrently_when_multiple_keys(self):
        # A barrier ensures every worker has entered before any returns, which
        # can only happen if the chunks are actually dispatched in parallel.
        chunks = list(range(4))
        barrier = threading.Barrier(4)
        active = threading.Lock()
        peak = {"value": 0, "concurrent": 0}

        def fn(item):
            with active:
                peak["concurrent"] += 1
                peak["value"] = max(peak["value"], peak["concurrent"])
            barrier.wait()
            with active:
                peak["concurrent"] -= 1
            return item

        with mock.patch.object(core.config, "GEMINI_API_KEYS", ["k1", "k2", "k3", "k4"]):
            result = core._refine_chunks_in_parallel(fn, chunks)
        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(peak["concurrent"], 0)
        # Four chunks spread across four keys should overlap fully.
        self.assertEqual(peak["value"], 4)

class MaxTokensSplitTests(unittest.TestCase):
    """MAX_TOKENS must shrink the refinement input instead of failing the job."""

    class _Counter:
        def __init__(self, counted=None):
            self.counted = counted

        def count(self, text):
            return self.counted if self.counted is not None else len(text)

        def estimate(self, text):
            return len(text)

    def _srt(self, blocks=8):
        lines = []
        for index in range(1, blocks + 1):
            lines.append(
                f"{index}\n"
                f"00:00:0{index},000 --> 00:00:0{index + 1},000\n"
                f"This is sentence number {index}.\n"
            )
        return "\n".join(lines) + "\n"

    def test_response_text_maps_max_tokens_to_output_too_long(self):
        response = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
                thoughts_token_count=0,
            ),
            candidates=[types.SimpleNamespace(finish_reason="MAX_TOKENS")],
            prompt_feedback=None,
            text="partial",
        )

        with self.assertRaises(core._OutputTooLong):
            core._response_text(response, "Text", job_id="job")

    def test_single_text_refinement_splits_on_max_tokens(self):
        raw_text = "Sentence one. " * 50
        calls = []

        def fake_chunk_refinement(
            client,
            model,
            chunk,
            part,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            *,
            job_id=None,
        ):
            calls.append(chunk.source_text)
            return chunk.source_text

        with (
            mock.patch.object(
                core,
                "_single_text_refinement",
                side_effect=core._OutputTooLong("MAX_TOKENS"),
            ),
            mock.patch.object(
                core, "_text_chunk_refinement", side_effect=fake_chunk_refinement
            ),
        ):
            result = core._refine_text(
                object(),
                raw_text,
                "English",
                False,
                30,
                self._Counter(counted=1000),
                job_id="job",
            )

        self.assertGreater(len(calls), 1)
        self.assertEqual(
            "".join(result.split()), "".join(raw_text.split())
        )

    def test_text_chunk_below_hard_cap_still_splits(self):
        text = "Sentence. " * 400
        chunk = core.chunking.TextChunk(
            blocks=[core.chunking.TextBlock(units=[text])],
            source_text=text,
            prompt_text=text,
            heading_context="",
            starts_with_heading=False,
        )
        calls = []

        def fake_chunk_refinement(
            client,
            model,
            chunk,
            part,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            *,
            job_id=None,
        ):
            calls.append(chunk.source_text)
            if len(calls) == 1:
                raise core._OutputTooLong("MAX_TOKENS")
            return chunk.source_text

        with mock.patch.object(
            core, "_text_chunk_refinement", side_effect=fake_chunk_refinement
        ):
            result = core._refine_text_chunk(
                object(),
                "model",
                chunk,
                1,
                1,
                "English",
                False,
                30,
                self._Counter(),
                job_id="job",
            )

        self.assertGreater(len(calls), 1)
        self.assertEqual("".join(result.split()), "".join(text.split()))

    def test_text_chunk_below_split_floor_reraises(self):
        text = "Short sentence."
        chunk = core.chunking.TextChunk(
            blocks=[core.chunking.TextBlock(units=[text])],
            source_text=text,
            prompt_text=text,
            heading_context="",
            starts_with_heading=False,
        )

        with mock.patch.object(
            core,
            "_text_chunk_refinement",
            side_effect=core._OutputTooLong("MAX_TOKENS"),
        ) as refinement:
            with self.assertRaises(core._OutputTooLong):
                core._refine_text_chunk(
                    object(),
                    "model",
                    chunk,
                    1,
                    1,
                    "English",
                    False,
                    30,
                    self._Counter(),
                    job_id="job",
                )

        self.assertEqual(refinement.call_count, 1)

    def test_single_srt_refinement_splits_and_merges(self):
        raw_text = self._srt(8)
        calls = []

        def fake_chunk_refinement(
            client,
            model,
            chunk,
            part,
            total,
            lang,
            cantonese_to_written,
            char_limit,
            *,
            job_id=None,
        ):
            calls.append(chunk.text)
            return chunk.text

        with (
            mock.patch.object(
                core,
                "_single_srt_refinement",
                side_effect=core._OutputTooLong("MAX_TOKENS"),
            ),
            mock.patch.object(
                core, "_srt_chunk_refinement", side_effect=fake_chunk_refinement
            ),
        ):
            result = core._refine_srt(
                object(),
                raw_text,
                "English",
                False,
                30,
                self._Counter(counted=1000),
                job_id="job",
            )

        self.assertGreater(len(calls), 1)
        blocks = core.chunking.parse_srt(result)
        self.assertEqual(
            [block.text for block in blocks],
            [f"This is sentence number {index}." for index in range(1, 9)],
        )

    def test_single_block_srt_refinement_reraises(self):
        raw_text = (
            "1\n00:00:01,000 --> 00:00:02,000\n"
            + "This is a long sentence. " * 20
            + "\n"
        )

        with mock.patch.object(
            core,
            "_single_srt_refinement",
            side_effect=core._OutputTooLong("MAX_TOKENS"),
        ):
            with self.assertRaises(core._OutputTooLong):
                core._refine_srt(
                    object(),
                    raw_text,
                    "English",
                    False,
                    30,
                    self._Counter(counted=1000),
                    job_id="job",
                )

    def test_single_block_srt_chunk_reraises_original_error(self):
        block = core.chunking.SrtBlock(
            index=1,
            timestamp="00:00:01,000 --> 00:00:02,000",
            text="Only one block.",
        )
        chunk = core.chunking.SrtChunk(
            blocks=[block], text=core.chunking.render_srt([block])
        )

        with mock.patch.object(
            core,
            "_srt_chunk_refinement",
            side_effect=core._OutputTooLong("MAX_TOKENS"),
        ):
            with self.assertRaises(core._OutputTooLong):
                core._refine_srt_chunk(
                    object(),
                    "model",
                    chunk,
                    1,
                    1,
                    "English",
                    False,
                    30,
                    job_id="job",
                )


class OpenCCTests(unittest.TestCase):
    def setUp(self):
        core._opencc_local = threading.local()

    def test_apply_opencc_reuses_converters_per_thread(self):
        calls = []

        class FakeConverter:
            def __init__(self, name):
                calls.append(name)
                self.name = name

            def convert(self, text):
                return f"{self.name}:{text}"

        with mock.patch.object(core.opencc, "OpenCC", FakeConverter):
            first = core.apply_opencc("甲", "Cantonese", "Traditional Chinese")
            second = core.apply_opencc("乙", "Cantonese", "Traditional Chinese")
            simplified = core.apply_opencc(
                "丙", "Mandarin", "Simplified Chinese"
            )

        self.assertEqual(calls, ["s2t", "t2s"])
        self.assertEqual(first, "s2t:甲")
        self.assertEqual(second, "s2t:乙")
        self.assertEqual(simplified, "t2s:丙")


class IntermediateCacheTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.store = CacheStore(root / "raw", root / "refine")

    def tearDown(self):
        self.tempdir.cleanup()

    def _client(self, text):
        client = mock.Mock()
        client.models.generate_content.return_value = _response(text)
        return client

    def test_refinement_response_is_reused_for_same_request(self):
        client = self._client("refined chunk")

        first = core._request_refinement(
            client,
            "model",
            "instruction",
            "chunk contents",
            "chunk contents",
            "Text",
            False,
            job_id="job",
            cache_store=self.store,
        )
        second = core._request_refinement(
            client,
            "model",
            "instruction",
            "chunk contents",
            "chunk contents",
            "Text",
            False,
            job_id="job",
            cache_store=self.store,
        )

        self.assertEqual(first, "refined chunk")
        self.assertEqual(second, "refined chunk")
        self.assertEqual(client.models.generate_content.call_count, 1)

        core._request_refinement(
            client,
            "model",
            "instruction",
            "different contents",
            "different contents",
            "Text",
            False,
            job_id="job",
            cache_store=self.store,
        )
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_json_plan_response_is_reused(self):
        client = self._client(
            '{"sections": [{"start_unit": 1, "heading": null}]}'
        )
        kwargs = dict(cache_store=self.store, cache_kind="text_plan")

        first = core._request_json(
            client, "model", "instruction", "contents", **kwargs
        )
        second = core._request_json(
            client, "model", "instruction", "contents", **kwargs
        )

        self.assertEqual(first, second)
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_successful_chunks_are_reused_after_partial_failure(self):
        chunks = [
            core.chunking.TextChunk(
                blocks=[
                    core.chunking.TextBlock(units=["First. "])
                ],
                source_text="First. ",
                prompt_text="First. ",
                heading_context="",
                starts_with_heading=False,
            ),
            core.chunking.TextChunk(
                blocks=[
                    core.chunking.TextBlock(units=["Second. "])
                ],
                source_text="Second. ",
                prompt_text="Second. ",
                heading_context="",
                starts_with_heading=False,
            ),
        ]
        calls = []
        failed = []

        def generate_content(**kwargs):
            contents = kwargs["contents"]
            if "First." in contents:
                calls.append("First.")
                return _response("refined First.")
            calls.append("Second.")
            if not failed:
                failed.append(True)
                raise RuntimeError("boom")
            return _response("refined Second.")

        client = mock.Mock()
        client.models.generate_content.side_effect = generate_content
        counter = MaxTokensSplitTests._Counter(counted=1000)

        with (
            mock.patch.object(core.config, "REFINE_CHUNK_TOKENS", 100),
            mock.patch.object(core.config, "GEMINI_API_KEYS", ["one"]),
            mock.patch.object(
                core,
                "_request_text_plan",
                return_value=[core.chunking.Section(1)],
            ),
            mock.patch.object(
                core.chunking, "group_text_sections", return_value=chunks
            ),
        ):
            with self.assertRaises(RuntimeError):
                core._refine_text(
                    client,
                    "First. Second. ",
                    "English",
                    False,
                    30,
                    counter,
                    job_id="job",
                    cache_store=self.store,
                )
            result = core._refine_text(
                client,
                "First. Second. ",
                "English",
                False,
                30,
                counter,
                job_id="job",
                cache_store=self.store,
            )

        self.assertIn("refined First.", result)
        self.assertIn("refined Second.", result)
        self.assertEqual(calls.count("First."), 1)
        self.assertEqual(calls.count("Second."), 2)

    def test_invalid_srt_response_is_not_cached(self):
        client = self._client("not valid srt")
        kwargs = dict(
            raw_text="not valid srt",
            output_format="SRT",
            cantonese_to_written=False,
            validate_result=lambda text: bool(core.chunking.parse_srt(text)),
            job_id="job",
            cache_store=self.store,
        )

        first = core._request_refinement(
            client, "model", "instruction", "contents", **kwargs
        )
        second = core._request_refinement(
            client, "model", "instruction", "contents", **kwargs
        )

        self.assertEqual(first, "not valid srt")
        self.assertEqual(second, "not valid srt")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_incomplete_cantonese_response_is_not_cached(self):
        client = self._client("x")
        kwargs = dict(
            raw_text="This is a long source text.",
            output_format="Text",
            cantonese_to_written=True,
            job_id="job",
            cache_store=self.store,
        )

        first = core._request_refinement(
            client, "model", "instruction", "contents", **kwargs
        )
        second = core._request_refinement(
            client, "model", "instruction", "contents", **kwargs
        )

        self.assertEqual(first, "x")
        self.assertEqual(second, "x")
        self.assertEqual(client.models.generate_content.call_count, 4)


class TokenCountingTests(unittest.TestCase):
    def test_chunking_uses_local_estimates(self):
        class FakeCounter:
            def __init__(self):
                self.counted = []

            def count(self, text):
                self.counted.append(text)
                return 5000

            def estimate(self, text):
                return 1000

        counter = FakeCounter()
        raw_text = "Sentence. " * 100
        captured = {}

        def fake_group(units, sections, target, hard, estimate, count):
            captured["estimate"] = estimate
            captured["count"] = count
            return []

        with (
            mock.patch.object(
                core,
                "_request_text_plan",
                return_value=[core.chunking.Section(1)],
            ),
            mock.patch.object(
                core.chunking, "group_text_sections", side_effect=fake_group
            ),
            mock.patch.object(
                core, "_single_text_refinement", return_value="ok"
            ),
        ):
            result = core._refine_text(
                object(),
                raw_text,
                "English",
                False,
                30,
                counter,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(counter.counted, [raw_text])
        self.assertEqual(captured["estimate"], counter.estimate)
        self.assertEqual(captured["count"], counter.estimate)


class TranscribeOptionsTests(unittest.TestCase):
    def test_english_transcription_uses_fast_greedy_vad_settings(self):
        class FakeWhisper:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, path, **kwargs):
                self.kwargs = kwargs
                return iter([]), None

        whisper = FakeWhisper()
        with mock.patch.object(
            core, "_models", return_value=(whisper, object())
        ):
            core.transcribe("audio.mp3", "English", "Text", "prompt")

        self.assertEqual(whisper.kwargs["beam_size"], 1)
        self.assertTrue(whisper.kwargs["vad_filter"])
        self.assertEqual(whisper.kwargs["initial_prompt"], "prompt")


if __name__ == "__main__":
    unittest.main()
