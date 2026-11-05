"""Deterministic helpers for manual AI refinement.

Manual refinement never calls Gemini. Prompts are built locally, sent to the
user, and the user returns the AI responses. Long inputs begin with an AI
structure-planning prompt that returns JSON, matching automatic refinement.
Prompts can be split again with ``/exceed`` when an external AI hits its output
limit.
"""

import json
import re
from typing import Dict, List, Optional

import chunking
import config
import prompts

TEXT_KIND = "Text"
SRT_KIND = "SRT"
TEXT_PLAN_KIND = "TextPlan"
SRT_PLAN_KIND = "SrtPlan"
PLAN_KINDS = {TEXT_PLAN_KIND, SRT_PLAN_KIND}
_SPLIT_FLOOR_TOKENS = 200


def is_plan_step(step: Dict[str, str]) -> bool:
    return step.get("kind") in PLAN_KINDS


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def strip_outer_code_fence(text: str) -> str:
    value = (text or "").strip()
    if not (value.startswith("```") and value.endswith("```")):
        return value
    lines = value.splitlines()
    if len(lines) < 2:
        return value
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_response(text: str, kind: str) -> str:
    value = strip_outer_code_fence(text)
    if not value:
        raise ValueError("The AI response is empty.")
    if kind == SRT_KIND and not chunking.parse_srt(value):
        raise ValueError("The AI response is not valid SRT.")
    return value


def parse_json_response(text: str):
    cleaned = strip_outer_code_fence(text)
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
    raise ValueError("The AI response is not valid JSON.")


def _single_step(kind: str, text: str) -> List[Dict[str, str]]:
    return [{"kind": kind, "text": text}]


def _text_steps_from_chunks(chunks) -> List[Dict[str, str]]:
    return [
        {
            "kind": TEXT_KIND,
            "text": chunk.prompt_text,
            "heading_context": chunk.heading_context,
            "starts_with_heading": chunk.starts_with_heading,
        }
        for chunk in chunks
    ]


def _srt_steps_from_chunks(chunks) -> List[Dict[str, str]]:
    return [{"kind": SRT_KIND, "text": chunk.text} for chunk in chunks]


def build_fallback_steps(
    raw_text: str,
    fmt: str,
    *,
    chunk_tokens: Optional[int] = None,
    hard_tokens: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Splits the source deterministically when no structure plan is used."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("The text is empty.")

    chunk_tokens = chunk_tokens or config.REFINE_CHUNK_TOKENS
    hard_tokens = hard_tokens or config.REFINE_CHUNK_HARD_TOKENS
    estimate = chunking.estimate_tokens

    if fmt == SRT_KIND:
        blocks = chunking.parse_srt(text)
        if not blocks:
            raise ValueError("The text is not valid SRT.")
        if estimate(text) < chunk_tokens or len(blocks) < 2:
            return _single_step(SRT_KIND, raw_text)
        chunks = chunking.build_srt_chunks(
            blocks,
            [],
            chunk_tokens,
            hard_tokens,
            estimate,
            estimate,
        )
        if len(chunks) <= 1:
            return _single_step(SRT_KIND, raw_text)
        return _srt_steps_from_chunks(chunks)

    if estimate(text) < chunk_tokens:
        return _single_step(TEXT_KIND, raw_text)
    units = chunking.split_text_units(text)
    chunks = chunking.group_text_sections(
        units,
        [chunking.Section(1)],
        chunk_tokens,
        hard_tokens,
        estimate,
        estimate,
    )
    if len(chunks) <= 1:
        return _single_step(TEXT_KIND, raw_text)
    return _text_steps_from_chunks(chunks)


def build_steps(
    raw_text: str,
    fmt: str,
    *,
    chunk_tokens: Optional[int] = None,
    hard_tokens: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Builds the first prompts shown, including AI structure planning."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("The text is empty.")

    chunk_tokens = chunk_tokens or config.REFINE_CHUNK_TOKENS
    hard_tokens = hard_tokens or config.REFINE_CHUNK_HARD_TOKENS
    estimate = chunking.estimate_tokens

    if fmt == SRT_KIND:
        blocks = chunking.parse_srt(text)
        if not blocks:
            raise ValueError("The text is not valid SRT.")
        if estimate(text) < chunk_tokens or len(blocks) < 2:
            return _single_step(SRT_KIND, raw_text)
        candidates = chunking.build_srt_candidates(blocks, estimate)
        if not candidates:
            return build_fallback_steps(
                raw_text,
                fmt,
                chunk_tokens=chunk_tokens,
                hard_tokens=hard_tokens,
            )
        return [{"kind": SRT_PLAN_KIND, "text": raw_text}]

    if estimate(text) < chunk_tokens:
        return _single_step(TEXT_KIND, raw_text)
    return [{"kind": TEXT_PLAN_KIND, "text": raw_text}]


def _number_text_units(units) -> str:
    numbered_units = [
        re.sub(r"\s+", " ", unit).strip() for unit in units
    ]
    return "\n".join(
        f"[{index}] {unit}"
        for index, unit in enumerate(numbered_units, start=1)
        if unit
    )


def build_planned_steps(
    step: Dict[str, str],
    payload,
    *,
    chunk_tokens: Optional[int] = None,
    hard_tokens: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Applies an AI structure plan to build the refinement prompts."""
    raw_text = step.get("text", "")
    text = raw_text.strip()
    if not text:
        raise ValueError("The text is empty.")

    chunk_tokens = chunk_tokens or config.REFINE_CHUNK_TOKENS
    hard_tokens = hard_tokens or config.REFINE_CHUNK_HARD_TOKENS
    estimate = chunking.estimate_tokens

    if step.get("kind") == SRT_PLAN_KIND:
        blocks = chunking.parse_srt(text)
        if not blocks:
            raise ValueError("The text is not valid SRT.")
        candidates = chunking.build_srt_candidates(blocks, estimate)
        split_after = chunking.parse_srt_splits(payload, candidates)
        chunks = chunking.build_srt_chunks(
            blocks,
            split_after,
            chunk_tokens,
            hard_tokens,
            estimate,
            estimate,
        )
        if not chunks:
            return build_fallback_steps(
                raw_text,
                SRT_KIND,
                chunk_tokens=chunk_tokens,
                hard_tokens=hard_tokens,
            )
        return _srt_steps_from_chunks(chunks)

    units = chunking.split_text_units(text)
    sections = chunking.parse_text_sections(payload, len(units))
    chunks = chunking.group_text_sections(
        units,
        sections,
        chunk_tokens,
        hard_tokens,
        estimate,
        estimate,
    )
    if not chunks:
        return build_fallback_steps(
            raw_text,
            TEXT_KIND,
            chunk_tokens=chunk_tokens,
            hard_tokens=hard_tokens,
        )
    return _text_steps_from_chunks(chunks)


def render_prompt(
    step: Dict[str, str],
    step_number: int,
    total: int,
    lang: str,
    fmt: str,
    cantonese_to_written: bool,
    char_limit: int,
) -> str:
    kind = step.get("kind", fmt)
    text = step.get("text", "")
    chunked = total > 1 or "heading_context" in step

    if kind == SRT_PLAN_KIND:
        blocks = chunking.parse_srt(text)
        candidates = chunking.build_srt_candidates(
            blocks, chunking.estimate_tokens
        )
        lines = [
            (
                f"- split after block {candidate.block_index} "
                f"(~{candidate.cumulative_tokens} tokens): "
                f"...{candidate.before} | next: {candidate.after}..."
            )
            for candidate in candidates
        ]
        return prompts.compose_prompt(
            prompts.srt_plan_system_instruction(),
            prompts.srt_plan_prompt(
                lang,
                config.REFINE_CHUNK_TOKENS,
                "\n".join(lines),
            ),
        )

    if kind == TEXT_PLAN_KIND:
        units = chunking.split_text_units(text)
        return prompts.compose_prompt(
            prompts.text_plan_system_instruction(lang, cantonese_to_written),
            prompts.text_plan_prompt(_number_text_units(units)),
        )

    if kind == SRT_KIND:
        if cantonese_to_written:
            system_instruction = prompts.c2w_system_instruction(
                "SRT", char_limit
            )
            if chunked:
                system_instruction += (
                    "\n\n"
                    + prompts.srt_chunk_system_instruction(step_number, total)
                )
            contents = text
        else:
            system_instruction = ""
            part = (step_number, total) if chunked else None
            contents = prompts.srt_prompt(text, lang, char_limit, part)
    else:
        if cantonese_to_written:
            system_instruction = prompts.c2w_system_instruction(
                "Text", char_limit
            )
            if chunked:
                system_instruction += (
                    "\n\n"
                    + prompts.text_chunk_system_instruction(
                        step_number, total
                    )
                )
            contents = text
        else:
            system_instruction = ""
            if chunked:
                contents = prompts.text_chunk_prompt(
                    text,
                    lang,
                    step_number,
                    total,
                    heading_context=step.get("heading_context", ""),
                    starts_with_heading=bool(
                        step.get("starts_with_heading")
                    ),
                )
            else:
                contents = prompts.text_single_prompt(text, lang)

    return prompts.compose_prompt(system_instruction, contents)


def split_step(
    step: Dict[str, str], *, floor_tokens: int = _SPLIT_FLOOR_TOKENS
) -> List[Dict[str, str]]:
    """Splits a pending step into smaller steps, when possible."""
    kind = step.get("kind", TEXT_KIND)
    text = step.get("text", "")
    if not text or kind in PLAN_KINDS:
        return [step]

    if kind == SRT_KIND:
        blocks = chunking.parse_srt(text)
        if len(blocks) <= 1:
            return [step]
        chunk = chunking.SrtChunk(blocks=list(blocks), text=text)
        pieces = chunking.split_srt_chunk_for_retry(
            chunk, chunking.estimate_tokens
        )
        if len(pieces) <= 1:
            return [step]
        return [{"kind": SRT_KIND, "text": piece.text} for piece in pieces]

    estimate = chunking.estimate_tokens
    tokens = estimate(text)
    if tokens <= max(1, floor_tokens):
        return [step]
    units = chunking.split_text_units(text)
    target = max(1, tokens // 2)
    chunks = chunking.group_text_sections(
        units,
        [chunking.Section(1)],
        target,
        target,
        estimate,
        estimate,
    )
    pieces = [chunk.source_text for chunk in chunks if chunk.source_text.strip()]
    if len(pieces) <= 1 and len(units) > 1:
        middle = len(units) // 2
        pieces = ["".join(units[:middle]), "".join(units[middle:])]
    pieces = [piece for piece in pieces if piece.strip()]
    if len(pieces) <= 1:
        return [step]
    return [{"kind": TEXT_KIND, "text": piece} for piece in pieces]


def merge_outputs(fmt: str, outputs: List[str]) -> str:
    if fmt == SRT_KIND:
        return chunking.merge_srt_texts(outputs)
    joined = "\n\n".join(
        output.strip() for output in outputs if output and output.strip()
    )
    return re.sub(r"\n{3,}", "\n\n", joined).strip()
