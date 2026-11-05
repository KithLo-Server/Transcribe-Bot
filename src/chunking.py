"""Deterministic helpers for splitting long transcripts before AI refinement.

The helpers in this module are intentionally free of Gemini/network imports so
they can be unit tested without loading the transcription models.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

EstimateTokens = Callable[[str], int]
CountTokens = Callable[[str], int]

_SENTENCE_ENDINGS = "。！？!?；;…"
_CLOSERS = "”’\"'）)]}】」』"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SRT_TIME_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)


def estimate_tokens(text: str) -> int:
    """Conservative offline token estimate used when the API is unavailable."""
    if not text:
        return 0
    cjk = 0
    for char in text:
        code = ord(char)
        if (
            0x3000 <= code <= 0x30FF
            or 0x3400 <= code <= 0x9FFF
            or 0xAC00 <= code <= 0xD7AF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF00 <= code <= 0xFFEF
        ):
            cjk += 1
    other = len(text) - cjk
    return max(1, int(math.ceil(cjk * 1.05 + other / 3.5)))


# ----------------------------- Plain text -----------------------------


@dataclass
class Section:
    start_unit: int
    heading: Optional[str] = None


@dataclass
class TextBlock:
    units: List[str]
    heading: Optional[str] = None
    context_before: str = ""

    @property
    def source_text(self) -> str:
        return "".join(self.units)

    @property
    def prompt_text(self) -> str:
        source = self.source_text
        if self.heading:
            return f"{self.heading}\n\n{source}"
        return source


@dataclass
class TextChunk:
    blocks: List[TextBlock]
    source_text: str
    prompt_text: str
    heading_context: str
    starts_with_heading: bool


def _consume_tail(text: str, index: int) -> int:
    while index < len(text) and text[index] in _CLOSERS:
        index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def split_text_units(text: str) -> List[str]:
    """Splits text into sentence-ish units without losing any characters."""
    units: List[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\n":
            end = index + 1
            while end < len(text) and text[end] == "\n":
                end += 1
            units.append(text[start:end])
            start = index = end
            continue
        if char in _SENTENCE_ENDINGS:
            end = _consume_tail(text, index + 1)
            units.append(text[start:end])
            start = index = end
            continue
        if char == ".":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if (
                not previous.isdigit()
                and (not following or following.isspace() or following in _CLOSERS)
            ):
                end = _consume_tail(text, index + 1)
                units.append(text[start:end])
                start = index = end
                continue
        index += 1
    if start < len(text):
        units.append(text[start:])
    return units


def _normalise_heading(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    match = _HEADING_RE.match(text)
    if match:
        level = min(max(len(match.group(1)), 2), 3)
        title = match.group(2).strip()
    else:
        level = 2
        title = text
    title = title.strip("# ").strip()
    if not title:
        return None
    return f"{'#' * level} {title[:200]}"


def parse_text_sections(payload, unit_count: int) -> List[Section]:
    """Validates the planner JSON and returns sorted, gap-tolerant sections."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("sections") or payload.get("paragraphs") or []
    else:
        items = []
    if not isinstance(items, list):
        items = []

    by_start = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start_unit"))
        except (TypeError, ValueError):
            continue
        if start < 1 or start > unit_count:
            continue
        heading = _normalise_heading(item.get("heading"))
        by_start.setdefault(start, heading)

    by_start.setdefault(1, None)
    return [
        Section(start_unit=start, heading=by_start[start])
        for start in sorted(by_start)
    ]


def _update_heading_path(path, heading: Optional[str]):
    if not heading:
        return path
    level = len(heading) - len(heading.lstrip("#"))
    new_path = list(path)
    while new_path and new_path[-1][0] >= level:
        new_path.pop()
    new_path.append((level, heading))
    return new_path


def _heading_context(path) -> str:
    return " > ".join(heading for _level, heading in path)


def _make_text_blocks(units: Sequence[str], sections: Sequence[Section]):
    blocks: List[TextBlock] = []
    path = []
    for position, section in enumerate(sections):
        end = (
            sections[position + 1].start_unit - 1
            if position + 1 < len(sections)
            else len(units)
        )
        start = section.start_unit - 1
        block_units = list(units[start:end])
        context = _heading_context(path)
        path = _update_heading_path(path, section.heading)
        if not block_units and not section.heading:
            continue
        blocks.append(
            TextBlock(
                units=block_units,
                heading=section.heading,
                context_before=context,
            )
        )
    return blocks


def _split_text_to_estimate(
    text: str, max_tokens: int, estimate: EstimateTokens
) -> List[str]:
    if not text or estimate(text) <= max_tokens:
        return [text] if text else []
    pieces: List[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if estimate(remaining) <= max_tokens:
            pieces.append(remaining)
            break
        ratio = max_tokens / max(1, estimate(remaining))
        size = max(1, min(len(remaining) - 1, int(len(remaining) * ratio)))
        end = start + size
        boundary = end
        while boundary > start + 1 and text[boundary - 1] not in " \t\n。！？!?；;，,.":
            boundary -= 1
        if boundary > start + 1:
            end = boundary
        while end > start + 1 and estimate(text[start:end]) > max_tokens:
            end = start + max(1, (end - start) // 2)
        pieces.append(text[start:end])
        start = end
    return pieces


def _pack_units(
    units: Sequence[str], target: int, estimate: EstimateTokens
) -> List[List[str]]:
    pieces: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for unit in units:
        for part in _split_text_to_estimate(unit, max(1, target), estimate):
            part_tokens = estimate(part)
            if current and current_tokens + part_tokens > target:
                pieces.append(current)
                current = []
                current_tokens = 0
            current.append(part)
            current_tokens += part_tokens
    if current or not pieces:
        pieces.append(current)
    return pieces


def _split_oversized_block(
    block: TextBlock,
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextBlock]:
    if count(block.prompt_text) <= hard_tokens:
        return [block]

    pieces = _pack_units(block.units, target, estimate)
    if len(pieces) <= 1:
        pieces = _pack_units(block.units, max(1, target // 2), estimate)
    blocks: List[TextBlock] = []
    heading = block.heading
    for piece_units in pieces:
        blocks.append(
            TextBlock(
                units=piece_units,
                heading=heading,
                context_before=block.context_before,
            )
        )
        heading = None
    return blocks


def _finalize_text_chunk(blocks: Sequence[TextBlock]) -> TextChunk:
    source_text = "".join(block.source_text for block in blocks)
    prompt_parts = [block.prompt_text.strip("\n") for block in blocks]
    prompt_text = "\n\n".join(part for part in prompt_parts if part)
    return TextChunk(
        blocks=list(blocks),
        source_text=source_text,
        prompt_text=prompt_text,
        heading_context=blocks[0].context_before if blocks else "",
        starts_with_heading=bool(blocks and blocks[0].heading),
    )


def _balanced_block_index(
    blocks: Sequence[TextBlock], estimate: EstimateTokens
) -> int:
    total = sum(estimate(block.prompt_text) for block in blocks)
    running = 0
    for index, block in enumerate(blocks[:-1], start=1):
        running += estimate(block.prompt_text)
        if running >= total / 2:
            return index
    return max(1, len(blocks) // 2)


def _verify_text_chunk(
    chunk: TextChunk,
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextChunk]:
    if count(chunk.prompt_text) <= hard_tokens:
        return [chunk]
    if len(chunk.blocks) > 1:
        middle = _balanced_block_index(chunk.blocks, estimate)
        left = _finalize_text_chunk(chunk.blocks[:middle])
        right = _finalize_text_chunk(chunk.blocks[middle:])
        return _verify_text_chunk(
            left, target, hard_tokens, estimate, count
        ) + _verify_text_chunk(right, target, hard_tokens, estimate, count)

    block = chunk.blocks[0]
    smaller = max(1, min(target // 2, hard_tokens // 2))
    pieces = _split_oversized_block(block, smaller, smaller, estimate, count)
    if len(pieces) <= 1:
        split_units: List[str] = []
        for unit in block.units:
            split_units.extend(_split_text_to_estimate(unit, smaller, estimate))
        if len(split_units) <= 1 and block.units:
            split_units = _split_text_to_estimate(
                block.source_text, smaller, estimate
            )
        if len(split_units) <= 1:
            return [chunk]
        pieces = [
            TextBlock(
                units=[piece],
                heading=block.heading if index == 0 else None,
                context_before=block.context_before,
            )
            for index, piece in enumerate(split_units)
        ]

    result: List[TextChunk] = []
    for piece in pieces:
        result.extend(
            _verify_text_chunk(
                _finalize_text_chunk([piece]),
                target,
                hard_tokens,
                estimate,
                count,
            )
        )
    return result


def _merge_small_text_chunks(
    chunks: Sequence[TextChunk],
    target: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextChunk]:
    merged: List[TextChunk] = []
    for chunk in chunks:
        if merged:
            candidate = _finalize_text_chunk(
                list(merged[-1].blocks) + list(chunk.blocks)
            )
            if (
                estimate(candidate.prompt_text) <= target
                and count(candidate.prompt_text) <= target
            ):
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return merged


def _split_oversized_paragraphs(
    blocks: Sequence[TextBlock],
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextBlock]:
    split: List[TextBlock] = []
    for block in blocks:
        split.extend(
            _split_oversized_block(
                block, target, hard_tokens, estimate, count
            )
        )
    return split


def _heading_groups(blocks: Sequence[TextBlock]) -> List[List[TextBlock]]:
    groups: List[List[TextBlock]] = []
    current: List[TextBlock] = []
    for block in blocks:
        if block.heading and current:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)
    return groups


def _block_group_text(blocks: Sequence[TextBlock]) -> str:
    return _finalize_text_chunk(blocks).prompt_text


def _pack_paragraphs(
    blocks: Sequence[TextBlock], target: int, estimate: EstimateTokens
) -> List[List[TextBlock]]:
    packed: List[List[TextBlock]] = []
    current: List[TextBlock] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = estimate(_block_group_text([block]))
        if current and current_tokens + block_tokens > target:
            packed.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        packed.append(current)
    return packed


def _branch_a_blocks(
    blocks: Sequence[TextBlock], target: int, estimate: EstimateTokens
) -> tuple[List[List[TextBlock]], List[TextBlock]]:
    packed = _pack_paragraphs(blocks, target, estimate)
    if not packed:
        return [], []
    last = packed[-1]
    if estimate(_block_group_text(last)) > target:
        return packed, []
    return packed[:-1], last


def _branch_b_blocks(
    seed: Sequence[TextBlock],
    group: Sequence[Sequence[TextBlock]],
    index: int,
    target: int,
    estimate: EstimateTokens,
) -> tuple[List[TextBlock], int]:
    merged = list(seed)
    while index < len(group):
        candidate = merged + list(group[index])
        if estimate(_block_group_text(candidate)) <= target:
            merged = candidate
            index += 1
        else:
            break
    return merged, index


def _group_heading_sections(
    blocks: Sequence[TextBlock],
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextChunk]:
    sections = _heading_groups(blocks)
    chunk_groups: List[List[TextBlock]] = []
    carry: List[TextBlock] = []
    index = 0

    while index < len(sections):
        section = list(sections[index])
        section_tokens = estimate(_block_group_text(section))

        if carry:
            combined = carry + section
            combined_tokens = estimate(_block_group_text(combined))
            if combined_tokens <= target:
                merged, index = _branch_b_blocks(
                    combined, sections, index + 1, target, estimate
                )
                chunk_groups.append(merged)
                carry = []
            elif section_tokens <= target:
                chunk_groups.append(carry)
                carry = []
                merged, index = _branch_b_blocks(
                    section, sections, index + 1, target, estimate
                )
                chunk_groups.append(merged)
            else:
                emitted, carry = _branch_a_blocks(
                    combined, target, estimate
                )
                chunk_groups.extend(emitted)
                index += 1
            continue

        if section_tokens <= target:
            merged, index = _branch_b_blocks(
                section, sections, index + 1, target, estimate
            )
            chunk_groups.append(merged)
        else:
            emitted, carry = _branch_a_blocks(section, target, estimate)
            chunk_groups.extend(emitted)
            index += 1

    if carry:
        chunk_groups.append(carry)

    chunks: List[TextChunk] = []
    for group in chunk_groups:
        if not group:
            continue
        chunks.extend(
            _verify_text_chunk(
                _finalize_text_chunk(group),
                target,
                hard_tokens,
                estimate,
                count,
            )
        )
    return chunks


def _group_text_blocks(
    blocks: Sequence[TextBlock],
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    current: List[TextBlock] = []
    current_tokens = 0
    for block in blocks:
        for piece in _split_oversized_block(
            block, target, hard_tokens, estimate, count
        ):
            piece_tokens = estimate(piece.prompt_text)
            if current and current_tokens + piece_tokens > target:
                chunks.append(_finalize_text_chunk(current))
                current = []
                current_tokens = 0
            current.append(piece)
            current_tokens += piece_tokens
    if current:
        chunks.append(_finalize_text_chunk(current))

    verified: List[TextChunk] = []
    for chunk in chunks:
        verified.extend(
            _verify_text_chunk(
                chunk, target, hard_tokens, estimate, count
            )
        )
    return _merge_small_text_chunks(verified, target, estimate, count)


def group_text_sections(
    units: Sequence[str],
    sections: Sequence[Section],
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens = estimate_tokens,
    count: Optional[CountTokens] = None,
) -> List[TextChunk]:
    count = count or estimate
    blocks = _make_text_blocks(units, sections)
    blocks = _split_oversized_paragraphs(
        blocks, target, hard_tokens, estimate, count
    )
    return _group_heading_sections(
        blocks, target, hard_tokens, estimate, count
    )


def split_text_chunk(
    chunk: TextChunk,
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens = estimate_tokens,
    count: Optional[CountTokens] = None,
) -> List[TextChunk]:
    count = count or estimate
    return _group_text_blocks(
        chunk.blocks, max(1, target // 2), hard_tokens, estimate, count
    )


def split_text_chunk_for_retry(
    chunk: TextChunk,
    *,
    floor_tokens: int = 1,
    estimate: EstimateTokens = estimate_tokens,
    count: Optional[CountTokens] = None,
) -> List[TextChunk]:
    """Forces a failing chunk to shrink instead of respecting the global hard cap.

    A chunk can be below the normal hard limit yet still make the model hit its
    output ceiling. Deriving the target from the chunk itself guarantees progress
    on every retry, while ``floor_tokens`` stops pathological recursion.
    """
    count = count or estimate
    tokens = max(estimate(chunk.prompt_text), count(chunk.prompt_text))
    if tokens <= max(1, floor_tokens):
        return [chunk]
    target = max(1, tokens // 2)
    return _group_text_blocks(chunk.blocks, target, target, estimate, count)


# ------------------------------- SRT -------------------------------


@dataclass
class SrtBlock:
    index: int
    timestamp: str
    text: str

    @property
    def rendered(self) -> str:
        return f"{self.index}\n{self.timestamp}\n{self.text}"


@dataclass
class SrtCandidate:
    block_index: int
    cumulative_tokens: int
    before: str
    after: str


@dataclass
class SrtChunk:
    blocks: List[SrtBlock]
    text: str


def parse_srt(text: str) -> List[SrtBlock]:
    lines = text.strip().splitlines()
    blocks: List[SrtBlock] = []
    position = 0
    while position < len(lines):
        line = lines[position].strip()
        if not line:
            position += 1
            continue

        timestamp = None
        if (
            line.isdigit()
            and position + 1 < len(lines)
            and "-->" in lines[position + 1]
            and _SRT_TIME_RE.match(lines[position + 1].strip())
        ):
            timestamp = lines[position + 1].strip()
            position += 2
        elif "-->" in line and _SRT_TIME_RE.match(line):
            timestamp = line
            position += 1
        else:
            position += 1
            continue

        subtitle_lines: List[str] = []
        while position < len(lines):
            candidate = lines[position].strip()
            if not candidate:
                position += 1
                break
            if (
                candidate.isdigit()
                and position + 1 < len(lines)
                and "-->" in lines[position + 1]
                and _SRT_TIME_RE.match(lines[position + 1].strip())
            ):
                break
            if "-->" in candidate and _SRT_TIME_RE.match(candidate):
                break
            subtitle_lines.append(lines[position].rstrip())
            position += 1

        subtitle = "\n".join(subtitle_lines).strip()
        if subtitle:
            blocks.append(
                SrtBlock(
                    index=len(blocks) + 1,
                    timestamp=timestamp,
                    text=subtitle,
                )
            )
    return blocks


def render_srt(blocks: Sequence[SrtBlock], start_index: int = 1) -> str:
    rendered = []
    for offset, block in enumerate(blocks):
        rendered.append(
            f"{start_index + offset}\n{block.timestamp}\n{block.text}"
        )
    return "\n\n".join(rendered)


def merge_srt_texts(texts: Sequence[str]) -> str:
    blocks: List[SrtBlock] = []
    for text in texts:
        parsed = parse_srt(text)
        if not parsed:
            raise ValueError("Gemini returned an invalid SRT chunk.")
        blocks.extend(parsed)
    return render_srt(blocks)


def ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1].rstrip()
    if not stripped:
        return False
    return stripped[-1] in _SENTENCE_ENDINGS or stripped.endswith(".")


def build_srt_candidates(
    blocks: Sequence[SrtBlock], estimate: EstimateTokens = estimate_tokens
) -> List[SrtCandidate]:
    candidates: List[SrtCandidate] = []
    cumulative = 0
    for position, block in enumerate(blocks[:-1]):
        cumulative += estimate(block.rendered)
        if not ends_sentence(block.text):
            continue
        following = blocks[position + 1]
        candidates.append(
            SrtCandidate(
                block_index=block.index,
                cumulative_tokens=cumulative,
                before=re.sub(r"\s+", " ", block.text)[-80:],
                after=re.sub(r"\s+", " ", following.text)[:80],
            )
        )
    return candidates


def parse_srt_splits(payload, candidates: Sequence[SrtCandidate]) -> List[int]:
    if isinstance(payload, dict):
        values = (
            payload.get("split_after_blocks")
            or payload.get("splits")
            or payload.get("chunks")
            or []
        )
    elif isinstance(payload, list):
        values = payload
    else:
        values = []
    if not isinstance(values, list):
        return []

    allowed = sorted(candidate.block_index for candidate in candidates)
    chosen = set()
    for value in values:
        if isinstance(value, dict):
            value = (
                value.get("block_index")
                or value.get("split_after")
                or value.get("end_block")
            )
        try:
            requested = int(value)
        except (TypeError, ValueError):
            continue
        position = bisect.bisect_right(allowed, requested) - 1
        if position >= 0:
            chosen.add(allowed[position])
    return sorted(chosen)


def _srt_chunk(blocks: Sequence[SrtBlock]) -> SrtChunk:
    return SrtChunk(blocks=list(blocks), text=render_srt(blocks))


def _best_srt_end(
    blocks: Sequence[SrtBlock],
    start: int,
    limit: int,
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
) -> int:
    cumulative = 0
    target_end = None
    hard_end = None
    for position in range(start, limit):
        cumulative += estimate(blocks[position].rendered)
        if cumulative > hard_tokens:
            break
        if position == limit - 1 or ends_sentence(blocks[position].text):
            hard_end = position + 1
            if cumulative <= target:
                target_end = position + 1
    if target_end:
        return target_end
    if hard_end:
        return hard_end
    return min(limit, start + 1)


def _verify_srt_chunk(
    chunk: SrtChunk,
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[SrtChunk]:
    if count(chunk.text) <= hard_tokens:
        return [chunk]
    if len(chunk.blocks) <= 1:
        return [chunk]
    pieces = split_srt_chunk(chunk, estimate)
    result: List[SrtChunk] = []
    for piece in pieces:
        result.extend(
            _verify_srt_chunk(
                piece, target, hard_tokens, estimate, count
            )
        )
    return result


def _merge_small_srt_chunks(
    chunks: Sequence[SrtChunk],
    target: int,
    estimate: EstimateTokens,
    count: CountTokens,
) -> List[SrtChunk]:
    merged: List[SrtChunk] = []
    for chunk in chunks:
        if merged:
            candidate = _srt_chunk(
                list(merged[-1].blocks) + list(chunk.blocks)
            )
            if (
                estimate(candidate.text) <= target
                and count(candidate.text) <= target
            ):
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return merged


def build_srt_chunks(
    blocks: Sequence[SrtBlock],
    split_after: Sequence[int],
    target: int,
    hard_tokens: int,
    estimate: EstimateTokens = estimate_tokens,
    count: Optional[CountTokens] = None,
) -> List[SrtChunk]:
    count = count or estimate
    index_by_number = {block.index: position for position, block in enumerate(blocks)}
    allowed = set(blocks[position].index for position in range(len(blocks) - 1))
    boundaries = sorted(
        {value for value in split_after if value in allowed}
    )
    starts = [0] + [index_by_number[value] + 1 for value in boundaries]
    ends = starts[1:] + [len(blocks)]
    chunks: List[SrtChunk] = []
    for start, limit in zip(starts, ends):
        position = start
        while position < limit:
            end = _best_srt_end(
                blocks, position, limit, target, hard_tokens, estimate
            )
            if end <= position:
                end = min(limit, position + 1)
            chunks.append(_srt_chunk(blocks[position:end]))
            position = end
    verified: List[SrtChunk] = []
    for chunk in chunks:
        verified.extend(
            _verify_srt_chunk(
                chunk, target, hard_tokens, estimate, count
            )
        )
    return _merge_small_srt_chunks(verified, target, estimate, count)


def split_srt_chunk(
    chunk: SrtChunk, estimate: EstimateTokens = estimate_tokens
) -> List[SrtChunk]:
    if len(chunk.blocks) <= 1:
        raise ValueError("Could not split an oversized SRT refinement chunk.")
    total = sum(estimate(block.rendered) for block in chunk.blocks)
    running = 0
    middle = None
    fallback = None
    for position, block in enumerate(chunk.blocks[:-1], start=1):
        running += estimate(block.rendered)
        if ends_sentence(block.text):
            fallback = position
            if running >= total / 2:
                middle = position
                break
    middle = middle or fallback or max(1, len(chunk.blocks) // 2)
    return [
        _srt_chunk(chunk.blocks[:middle]),
        _srt_chunk(chunk.blocks[middle:]),
    ]


def split_srt_chunk_for_retry(
    chunk: SrtChunk, estimate: EstimateTokens = estimate_tokens
) -> List[SrtChunk]:
    """Splits an SRT chunk at a block boundary, or returns it when impossible."""
    if len(chunk.blocks) <= 1:
        return [chunk]
    return split_srt_chunk(chunk, estimate)
