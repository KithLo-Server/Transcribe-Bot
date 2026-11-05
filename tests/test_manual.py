"""Tests for deterministic manual-refinement prompt handling."""

import unittest

import _bootstrap  # noqa: F401
import chunking
import manual


def _srt(blocks=8):
    lines = []
    for index in range(1, blocks + 1):
        lines.append(
            f"{index}\n"
            f"00:00:0{index},000 --> 00:00:0{index + 1},000\n"
            f"This is sentence number {index}.\n"
        )
    return "\n".join(lines) + "\n"


class ResponseTests(unittest.TestCase):
    def test_strips_outer_code_fences(self):
        self.assertEqual(
            manual.strip_outer_code_fence("```text\nhello\n```"),
            "hello",
        )
        self.assertEqual(
            manual.strip_outer_code_fence("plain"),
            "plain",
        )

    def test_normalize_rejects_empty_and_invalid_srt(self):
        with self.assertRaises(ValueError):
            manual.normalize_response("   ", manual.TEXT_KIND)
        with self.assertRaises(ValueError):
            manual.normalize_response("not srt", manual.SRT_KIND)

    def test_normalize_accepts_srt_in_a_code_fence(self):
        value = manual.normalize_response(
            f"```srt\n{_srt(1)}```", manual.SRT_KIND
        )
        self.assertEqual(len(chunking.parse_srt(value)), 1)


class StepTests(unittest.TestCase):
    def test_short_text_is_a_single_step(self):
        steps = manual.build_steps("Short text.", "Text", chunk_tokens=1000)
        self.assertEqual(steps, [{"kind": "Text", "text": "Short text."}])

    def test_long_text_starts_with_a_plan_step(self):
        text = "This is a sentence. " * 400
        steps = manual.build_steps(
            text, "Text", chunk_tokens=100, hard_tokens=100
        )
        self.assertEqual(
            steps, [{"kind": manual.TEXT_PLAN_KIND, "text": text}]
        )

    def test_long_text_fallback_is_split(self):
        text = "This is a sentence. " * 400
        steps = manual.build_fallback_steps(
            text, "Text", chunk_tokens=100, hard_tokens=100
        )
        self.assertGreater(len(steps), 1)
        self.assertEqual(
            "".join(
                "".join(step["text"].split()) for step in steps
            ),
            "".join(text.split()),
        )

    def test_long_srt_starts_with_a_plan_step(self):
        text = _srt(8)
        steps = manual.build_steps(
            text, "SRT", chunk_tokens=10, hard_tokens=10
        )
        self.assertEqual(
            steps, [{"kind": manual.SRT_PLAN_KIND, "text": text}]
        )

    def test_long_srt_fallback_is_split(self):
        text = _srt(8)
        steps = manual.build_fallback_steps(
            text, "SRT", chunk_tokens=10, hard_tokens=10
        )
        self.assertGreater(len(steps), 1)
        merged = chunking.merge_srt_texts([step["text"] for step in steps])
        self.assertEqual(
            [block.text for block in chunking.parse_srt(merged)],
            [
                f"This is sentence number {index}."
                for index in range(1, 9)
            ],
        )

    def test_render_prompt_contains_source_restriction_and_text(self):
        prompt = manual.render_prompt(
            {"kind": "Text", "text": "hello"},
            1,
            1,
            "English",
            "Text",
            False,
            30,
        )
        self.assertIn("Work only from the supplied text.", prompt)
        self.assertIn("hello", prompt)

    def test_render_prompt_adds_part_numbering(self):
        prompt = manual.render_prompt(
            {"kind": "SRT", "text": _srt(1)},
            2,
            3,
            "English",
            "SRT",
            False,
            30,
        )
        self.assertIn("PART 2 OF 3", prompt)

    def test_render_text_plan_prompt_requests_json(self):
        prompt = manual.render_prompt(
            {"kind": manual.TEXT_PLAN_KIND, "text": "First. Second."},
            1,
            1,
            "English",
            "Text",
            False,
            30,
        )
        self.assertIn("document structure planner", prompt)
        self.assertIn("[1] First.", prompt)
        self.assertIn('"start_unit"', prompt)

    def test_render_srt_plan_prompt_lists_candidates(self):
        prompt = manual.render_prompt(
            {"kind": manual.SRT_PLAN_KIND, "text": _srt(3)},
            1,
            1,
            "English",
            "SRT",
            False,
            30,
        )
        self.assertIn("split_after_blocks", prompt)
        self.assertIn("split after block", prompt)

    def test_parse_json_response_strips_code_fence(self):
        self.assertEqual(
            manual.parse_json_response('```json\n{"sections": []}\n```'),
            {"sections": []},
        )
        with self.assertRaises(ValueError):
            manual.parse_json_response("not json")

    def test_build_planned_steps_keeps_headings(self):
        text = "First sentence. Second sentence. Third sentence."
        payload = {
            "sections": [
                {"start_unit": 1, "heading": "## Intro"},
                {"start_unit": 2, "heading": None},
            ]
        }
        steps = manual.build_planned_steps(
            {"kind": manual.TEXT_PLAN_KIND, "text": text},
            payload,
            chunk_tokens=1000,
            hard_tokens=1000,
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["kind"], manual.TEXT_KIND)
        self.assertIn("## Intro", steps[0]["text"])
        self.assertTrue(steps[0]["starts_with_heading"])
        self.assertIn("First sentence.", steps[0]["text"])

    def test_build_planned_srt_steps_uses_split_points(self):
        text = _srt(4)
        blocks = chunking.parse_srt(text)
        candidates = chunking.build_srt_candidates(blocks)
        steps = manual.build_planned_steps(
            {"kind": manual.SRT_PLAN_KIND, "text": text},
            {"split_after_blocks": [candidates[0].block_index]},
            chunk_tokens=10,
            hard_tokens=10,
        )
        self.assertGreaterEqual(len(steps), 2)

    def test_split_step_leaves_plan_unchanged(self):
        step = {"kind": manual.TEXT_PLAN_KIND, "text": "This is a sentence. " * 200}
        self.assertEqual(manual.split_step(step), [step])

    def test_split_step_halves_text(self):
        text = "This is a sentence. " * 200
        step = {"kind": "Text", "text": text}
        pieces = manual.split_step(step)
        self.assertGreater(len(pieces), 1)
        self.assertEqual(
            "".join(
                "".join(piece["text"].split()) for piece in pieces
            ),
            "".join(text.split()),
        )

    def test_split_step_splits_srt_at_block_boundary(self):
        step = {"kind": "SRT", "text": _srt(4)}
        pieces = manual.split_step(step)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(len(chunking.parse_srt(pieces[0]["text"])), 2)
        self.assertEqual(len(chunking.parse_srt(pieces[1]["text"])), 2)

    def test_split_step_leaves_single_srt_block(self):
        step = {"kind": "SRT", "text": _srt(1)}
        self.assertEqual(manual.split_step(step), [step])


class MergeTests(unittest.TestCase):
    def test_merge_text_outputs(self):
        self.assertEqual(
            manual.merge_outputs("Text", [" first ", "second"]),
            "first\n\nsecond",
        )

    def test_merge_srt_outputs_reindexes(self):
        first = _srt(2)
        second = _srt(2).replace("1\n", "1\n", 1)
        merged = manual.merge_outputs("SRT", [first, second])
        blocks = chunking.parse_srt(merged)
        self.assertEqual([block.index for block in blocks], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
