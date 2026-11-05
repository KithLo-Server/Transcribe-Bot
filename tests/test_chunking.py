"""Tests for deterministic refinement chunking."""

import unittest

import _bootstrap  # noqa: F401
import chunking


def char_count(text):
    return len(text)


class TextChunkingTests(unittest.TestCase):
    def test_split_text_units_reconstructs_exact_text(self):
        text = (
            "Hello world. This is a test! 這是第一句。這是第二句。\n\n"
            "New paragraph starts here? Yes."
        )
        units = chunking.split_text_units(text)
        self.assertEqual("".join(units), text)
        self.assertGreaterEqual(len(units), 5)

    def test_split_text_units_handles_text_without_punctuation(self):
        text = "no punctuation at all"
        self.assertEqual(chunking.split_text_units(text), [text])

    def test_parse_text_sections_validates_and_normalises(self):
        payload = {
            "sections": [
                {"start_unit": 3, "heading": "Title"},
                {"start_unit": 3, "heading": "## Later"},
                {"start_unit": 0, "heading": "bad"},
                {"start_unit": 99, "heading": "bad"},
                {"start_unit": 4, "heading": "### Detail"},
            ]
        }
        sections = chunking.parse_text_sections(payload, unit_count=5)
        self.assertEqual(
            sections,
            [
                chunking.Section(1, None),
                chunking.Section(3, "## Title"),
                chunking.Section(4, "### Detail"),
            ],
        )

    def test_group_text_sections_covers_input_and_keeps_headings(self):
        text = "One. Two. Three. Four. Five. Six."
        units = chunking.split_text_units(text)
        sections = [
            chunking.Section(1, "## First"),
            chunking.Section(3, "### Second"),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=20,
            hard_tokens=40,
            estimate=char_count,
            count=char_count,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), text)
        self.assertTrue(chunks[0].starts_with_heading)
        self.assertIn("## First", chunks[0].prompt_text)
        self.assertIn("### Second", "".join(c.prompt_text for c in chunks))
        for chunk in chunks:
            self.assertTrue(chunk.prompt_text)

    def test_oversized_paragraph_is_split_without_losing_text(self):
        text = "word " * 200
        units = chunking.split_text_units(text)
        sections = [chunking.Section(1, "## Heading")]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=100,
            hard_tokens=120,
            estimate=char_count,
            count=char_count,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.source_text for chunk in chunks), text)
        self.assertTrue(chunks[0].starts_with_heading)
        self.assertFalse(any(chunk.starts_with_heading for chunk in chunks[1:]))

    def test_heading_starts_a_new_section(self):
        units = ["A" * 10, "B" * 10]
        sections = [
            chunking.Section(1, "## A"),
            chunking.Section(2, "## B"),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=20,
            hard_tokens=40,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].starts_with_heading)
        self.assertTrue(chunks[1].starts_with_heading)
        self.assertIn("## A", chunks[0].prompt_text)
        self.assertIn("## B", chunks[1].prompt_text)
        self.assertEqual(
            "".join(chunk.source_text for chunk in chunks), "".join(units)
        )

    def test_small_sections_join_up_to_the_soft_limit(self):
        units = ["A" * 10, "B" * 10, "C" * 3]
        sections = [
            chunking.Section(1, "## A"),
            chunking.Section(2, "## B"),
            chunking.Section(3, "## C"),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=35,
            hard_tokens=40,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 2)
        self.assertIn("## A", chunks[0].prompt_text)
        self.assertIn("## B", chunks[0].prompt_text)
        self.assertIn("## C", chunks[1].prompt_text)
        self.assertEqual(
            "".join(chunk.source_text for chunk in chunks), "".join(units)
        )

    def test_branch_a_carries_the_last_chunk_into_the_next_section(self):
        units = ["A" * 10, "B" * 10, "C" * 10, "D" * 9]
        sections = [
            chunking.Section(1, "## A"),
            chunking.Section(2, None),
            chunking.Section(3, None),
            chunking.Section(4, "## B"),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=30,
            hard_tokens=100,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_text, "A" * 10 + "B" * 10)
        self.assertEqual(chunks[1].source_text, "C" * 10 + "D" * 9)
        self.assertTrue(chunks[0].starts_with_heading)
        self.assertFalse(chunks[1].starts_with_heading)
        self.assertIn("## B", chunks[1].prompt_text)

    def test_leftover_is_emitted_when_next_section_alone_fits(self):
        units = ["A" * 10, "B" * 10, "C" * 10, "D" * 16]
        sections = [
            chunking.Section(1, "## A"),
            chunking.Section(2, None),
            chunking.Section(3, None),
            chunking.Section(4, "## B"),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=30,
            hard_tokens=100,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[1].source_text, "C" * 10)
        self.assertTrue(chunks[2].starts_with_heading)
        self.assertIn("## B", chunks[2].prompt_text)
        self.assertEqual(
            "".join(chunk.source_text for chunk in chunks), "".join(units)
        )

    def test_leftover_and_oversized_section_reenter_branch_a(self):
        units = ["A" * 10, "B" * 10, "C" * 10, "D" * 9, "E" * 20]
        sections = [
            chunking.Section(1, "## A"),
            chunking.Section(2, None),
            chunking.Section(3, None),
            chunking.Section(4, "## B"),
            chunking.Section(5, None),
        ]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=30,
            hard_tokens=100,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [chunk.source_text for chunk in chunks],
            [
                "A" * 10 + "B" * 10,
                "C" * 10 + "D" * 9,
                "E" * 20,
            ],
        )
        self.assertEqual(
            "".join(chunk.source_text for chunk in chunks), "".join(units)
        )

    def test_oversized_paragraph_splits_before_heading_grouping(self):
        units = ["word "] * 40
        sections = [chunking.Section(1, "## A")]
        chunks = chunking.group_text_sections(
            units,
            sections,
            target=40,
            hard_tokens=60,
            estimate=char_count,
            count=char_count,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].starts_with_heading)
        self.assertFalse(any(chunk.starts_with_heading for chunk in chunks[1:]))
        self.assertEqual(
            "".join(chunk.source_text for chunk in chunks), "".join(units)
        )
        for chunk in chunks:
            self.assertLessEqual(char_count(chunk.prompt_text), 60)

    def test_split_text_chunk_reduces_an_oversized_chunk(self):
        text = "A. B. C. D. E. F. G. H."
        units = chunking.split_text_units(text)
        chunks = chunking.group_text_sections(
            units,
            [chunking.Section(1, None)],
            target=100,
            hard_tokens=120,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 1)
        pieces = chunking.split_text_chunk(
            chunks[0],
            target=10,
            hard_tokens=12,
            estimate=char_count,
            count=char_count,
        )
        self.assertGreater(len(pieces), 1)
        self.assertEqual(
            "".join(piece.source_text for piece in pieces), chunks[0].source_text
        )

    def test_split_text_chunk_for_retry_forces_a_split(self):
        text = "A. B. C. D. E. F. G. H."
        units = chunking.split_text_units(text)
        chunks = chunking.group_text_sections(
            units,
            [chunking.Section(1, None)],
            target=100,
            hard_tokens=120,
            estimate=char_count,
            count=char_count,
        )
        self.assertEqual(len(chunks), 1)

        pieces = chunking.split_text_chunk_for_retry(
            chunks[0], estimate=char_count, count=char_count
        )
        self.assertGreater(len(pieces), 1)
        self.assertEqual(
            "".join(piece.source_text for piece in pieces),
            chunks[0].source_text,
        )

        # Below the floor the failing chunk is returned unchanged, so callers can
        # surface the original error instead of splitting forever.
        self.assertEqual(
            chunking.split_text_chunk_for_retry(
                chunks[0],
                floor_tokens=len(text) + 1,
                estimate=char_count,
                count=char_count,
            ),
            [chunks[0]],
        )


class SrtChunkingTests(unittest.TestCase):
    SRT = """1
00:00:01,000 --> 00:00:02,000
Hello world.
2
00:00:02,000 --> 00:00:03,000
This is a test.
3
00:00:03,000 --> 00:00:04,000
Still going
4
00:00:04,000 --> 00:00:05,000
and ending.
"""

    def test_parse_srt_accepts_missing_blank_lines(self):
        blocks = chunking.parse_srt(self.SRT)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[0].text, "Hello world.")
        self.assertEqual(blocks[3].text, "and ending.")

    def test_parse_srt_accepts_blank_lines_and_renders(self):
        spaced = self.SRT.replace("\n2\n", "\n\n2\n").replace(
            "\n3\n", "\n\n3\n"
        ).replace("\n4\n", "\n\n4\n")
        blocks = chunking.parse_srt(spaced)
        self.assertEqual(len(blocks), 4)
        self.assertIn("Hello world.", chunking.render_srt(blocks))

    def test_candidates_only_include_sentence_endings(self):
        blocks = chunking.parse_srt(self.SRT)
        candidates = chunking.build_srt_candidates(blocks, char_count)
        self.assertEqual([c.block_index for c in candidates], [1, 2])

    def test_parse_srt_splits_maps_unsafe_points_to_previous_candidate(self):
        blocks = chunking.parse_srt(self.SRT)
        candidates = chunking.build_srt_candidates(blocks, char_count)
        splits = chunking.parse_srt_splits(
            {"split_after_blocks": [3, 2, 0, 99]}, candidates
        )
        self.assertEqual(splits, [2])

    def test_build_srt_chunks_respects_splits_and_merge_reindexes(self):
        blocks = chunking.parse_srt(self.SRT)
        chunks = chunking.build_srt_chunks(
            blocks, [2], target=100, hard_tokens=120, estimate=char_count
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].blocks[0].text, "Hello world.")
        self.assertEqual(chunks[1].blocks[-1].text, "and ending.")

        merged = chunking.merge_srt_texts([chunk.text for chunk in chunks])
        merged_blocks = chunking.parse_srt(merged)
        self.assertEqual([b.index for b in merged_blocks], [1, 2, 3, 4])
        self.assertEqual(merged_blocks[1].text, "This is a test.")

    def test_build_srt_chunks_repairs_oversized_chunks_at_safe_boundaries(self):
        blocks = chunking.parse_srt(self.SRT)
        chunks = chunking.build_srt_chunks(
            blocks, [], target=80, hard_tokens=100, estimate=char_count
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks[:-1]:
            self.assertTrue(
                chunking.ends_sentence(chunk.blocks[-1].text),
                "chunks should end on a sentence boundary when possible",
            )

    def test_split_srt_chunk_uses_sentence_boundary(self):
        blocks = chunking.parse_srt(self.SRT)
        chunk = chunking.SrtChunk(
            blocks=blocks,
            text=chunking.render_srt(blocks),
        )
        pieces = chunking.split_srt_chunk(chunk, char_count)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(chunking.ends_sentence(pieces[0].blocks[-1].text))
        self.assertEqual(
            [b.text for p in pieces for b in p.blocks],
            [b.text for b in blocks],
        )

    def test_split_srt_chunk_for_retry_handles_single_block(self):
        blocks = chunking.parse_srt(self.SRT)
        chunk = chunking.SrtChunk(
            blocks=blocks,
            text=chunking.render_srt(blocks),
        )
        self.assertEqual(
            len(chunking.split_srt_chunk_for_retry(chunk, char_count)), 2
        )

        single = chunking.SrtChunk(
            blocks=blocks[:1],
            text=chunking.render_srt(blocks[:1]),
        )
        self.assertEqual(
            chunking.split_srt_chunk_for_retry(single, char_count), [single]
        )

    def test_estimate_tokens_is_positive(self):
        self.assertGreater(chunking.estimate_tokens("hello"), 0)
        self.assertGreater(chunking.estimate_tokens("你好世界"), 0)


if __name__ == "__main__":
    unittest.main()
