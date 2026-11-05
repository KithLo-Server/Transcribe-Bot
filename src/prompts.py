"""Prompt templates shared by automatic and manual AI refinement."""

CANTONESE_GUIDELINES = """### Grammatical & Syntactic Transformations (CRITICAL):
1. Word Order & Adverbs: Move post-verbal adverbs to pre-verbal position.
   - 「行先」 -> 「先走」
   - 「睇多陣」 -> 「再看一會 / 多看一會」
   - 「講少句」 -> 「少說一句」
2. Comparative Structures: Convert [A + Adjective + 過 + B] to [A + 比 + B + Adjective].
   - 「佢高過我」 -> 「他比我高」
   - 「貴過頭」 -> 「太貴了」
3. Dative / Giving Constructions: Convert [畀 + Object + Person] to [把/將 + Object + 給 + Person] or [給 + Person + Object].
   - 「畀支筆我」 -> 「給我一支筆」 / 「把筆給我」
4. Aspect & Disposal Markers:
   - 「做緊」 -> 「正在做」
   - 「食咗」 -> 「吃了」
   - 「將啲嘢收埋」 -> 「把東西收好 / 藏起來」
   - 「俾人話」 -> 「被人說 / 被人罵」
5. Spoken Idioms & Collocations: Do not translate literally; use natural written expressions.
   - 「返工 / 放工」 -> 「上班 / 下班」
   - 「傾偈」 -> 「聊天」
   - 「拍拖」 -> 「談戀愛」
   - 「執笠」 -> 「倒閉」
   - 「搞掂」 -> 「搞定 / 辦妥」
   - 「好彩 / 陰功」 -> 「幸好 / 可憐」
   - 「點算」 -> 「怎麼辦」
   - 「唔使」 -> 「不用」
   - 「啱啱」 -> 「剛剛」
6. Modal Particles & Conversational Fillers:
   - Omit meaningless filler clusters (e.g., 「即係呢」、「噉樣樣」、「喇喎」、「㗎嘛」) or replace with natural punctuation or standard modal particles (「了」、「呢」、「吧」). Do not translate them literally."""

SOURCE_RESTRICTION = (
    "Do not use web search, external sources, or outside knowledge. "
    "Work only from the supplied text."
)


def c2w_system_instruction(output_format: str, char_limit: int) -> str:
    base_instructions = f"""You are a professional linguistic editor specializing in Cantonese-to-Standard-Written-Chinese (粵語口語轉規範書面語).

Transform colloquial spoken Cantonese into natural, publication-grade Standard Written Chinese (現代標準漢語). Ensure the output sounds like authentic written prose, NOT Cantonese grammar masked with Mandarin vocabulary. Keep the exact same script (Traditional or Simplified) as the input text.

{CANTONESE_GUIDELINES}

FEW-SHOT EXAMPLES:
- Input: 你食先啦，我仲有少少嘢未搞掂，陣間過嚟搵你哋。
  Output: 你先吃吧，我還有點事情沒處理完，待會兒過去找你們。
- Input: 呢部電話貴過嗰部成千蚊，但係好似冇咁好用喎。
  Output: 這部手機比那部貴了一千多塊，但是好像沒那麼好用。
- Input: 佢啱啱畀咗份文件我，叫我即刻交畀老闆。
  Output: 他剛才給了我一份文件，讓我立刻交給老闆。
"""

    if output_format == "SRT":
        return base_instructions + f"""
SUBTITLE & SRT RULES:
1. PRESERVE MEANING: Retain all substantive information and nuances. Remove only meaningless verbal filler repetitions and conversational padding.
2. MERGE SHORT SUBTITLES: Merge adjacent choppy blocks that form a single clause or sentence, up to ~{char_limit} characters per block.
3. TIMESTAMP ACCURACY:
   - When merging Subtitle A and Subtitle B, the merged timestamp MUST be [Start of A] --> [End of B].
   - Never retain the premature end timestamp of Subtitle A.
4. VALID FORMAT: Output strictly sequential indexes starting at 1, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, and standard subtitle text separated by single blank lines.
5. NO MARKDOWN: Output raw SRT only. No code blocks (```srt), no explanations."""

    return base_instructions + """
TEXT RULES:
1. PRESERVE MEANING: Retain all factual content, thoughts, and context while converting the syntax completely into standard written syntax.
2. CLEAN UP ORAL DISFLUENCY: Smooth out stutters, false starts, and excessive filler sounds into coherent written sentences.
3. STRUCTURE: Keep logical paragraphs and punctuation intact. Do not add Markdown titles or commentary."""


def text_plan_system_instruction(
    lang: str, cantonese_to_written: bool
) -> str:
    if cantonese_to_written:
        language_rule = (
            "Write heading titles in Standard Written Chinese, not colloquial "
            "Cantonese."
        )
    else:
        language_rule = f"Write heading titles in {lang}."
    return f"""You are a document structure planner.

Decide where paragraphs begin and where high-level Markdown headings (`##`, `###`) should be placed.

RULES:
1. PARAGRAPHS: Plan frequent paragraph breaks (roughly every 2 to 4 sentence units).
2. HEADINGS: Use headings when there is a thematic shift (typically every 4 to 8 paragraphs). Use `"heading": null` if no headings is given to a paragraph.
3. The first section must start at unit 1. {language_rule}

Return ONLY JSON in this shape:
{{"sections": [
    {{"start_unit": 1, "heading": "## Introduction"}},
    {{"start_unit": 4, "heading": null}},
    {{"start_unit": 8, "heading": null}},
    {{"start_unit": 12, "heading": null}},
    {{"start_unit": 16, "heading": "## Next Topic"}},
    {{"start_unit": 20, "heading": null}}
]}}"""


def text_plan_prompt(numbered_units: str) -> str:
    return f"""Numbered transcript units:

{numbered_units}

Return the section plan as JSON."""


def srt_plan_system_instruction() -> str:
    return """You are planning how to split a long SRT transcript
for AI refinement. Choose split points ONLY from the supplied candidate blocks.
Each candidate already ends a complete sentence or thought, so never invent a
different split point. Aim for chunks of about the target token count, using as
few splits as possible. Return ONLY JSON in this shape:
{"split_after_blocks": [12, 27]}"""


def srt_plan_prompt(
    lang: str, target_tokens: int, candidate_lines: str
) -> str:
    return f"""Language: {lang}
Target chunk size: about {target_tokens} tokens.

Candidate split points:
{candidate_lines}

Choose the split points as JSON."""


def text_single_prompt(raw_text: str, lang: str) -> str:
    return f"""You are an expert transcriber and editor.
Your task is to tidy up the following raw transcription.

CRITICAL RULES:
1. ZERO INFORMATION LOSS: DO NOT summarize, omit, or lose ANY content from the original audio. Keep all details, terms, and sentences intact.
2. FORMATTING ONLY:
   - Add logical paragraph breaks.
   - Fix missing or incorrect punctuation.
   - Correct obvious transcription mistakes and typos.
   - Add appropriate Markdown headings (##, ###) to structure the document.

The language of the text is {lang}.
Output ONLY the refined transcription without any introductory or concluding comments.

Raw Transcription:
{raw_text}
"""


def text_chunk_system_instruction(part: int, total: int) -> str:
    return f"""CHUNK RULES:
- This is part {part} of {total} of one longer document.
- Preserve every Markdown heading line exactly as supplied.
- Do not add or remove headings; the document structure was already decided.
- Do not add an introduction or conclusion.
- Output only the converted text for this part."""


def text_chunk_prompt(
    text: str,
    lang: str,
    part: int,
    total: int,
    *,
    heading_context: str = "",
    starts_with_heading: bool = False,
) -> str:
    if starts_with_heading:
        position_rule = (
            "This part starts with a heading. Keep that heading exactly."
        )
    else:
        position_rule = (
            "This part starts in the middle of a section. Do not add a "
            "heading at the start."
        )
    context = (
        f"\nCurrent heading context: {heading_context}"
        if heading_context
        else ""
    )
    return f"""You are an expert transcriber and editor refining part {part} of {total} of one long transcription in {lang}.

CRITICAL RULES:
1. ZERO INFORMATION LOSS: Do not summarize, omit, or lose any content. Keep every sentence and detail intact.
2. FORMATTING ONLY: Fix missing or incorrect punctuation, correct obvious transcription mistakes and typos, and keep the supplied paragraph breaks.
3. STRUCTURE: Preserve every supplied Markdown heading exactly. Do not add new headings. {position_rule}{context}
4. Output ONLY the refined text for this part. Do not add an introduction or conclusion.

Part transcription:
{text}
"""


def srt_prompt(raw_text: str, lang: str, char_limit: int, part=None) -> str:
    part_rule = ""
    if part is not None:
        part_rule = (
            f"\n4. **PART {part[0]} OF {part[1]}:** Output only this part. "
            "Do not add an introduction, conclusion, or Markdown headings."
        )
    return f"""You are an expert subtitle editor.
Your task is to refine the following raw SRT transcription in {lang}.

### CRITICAL RULES:
1. **ZERO INFORMATION LOSS:** DO NOT summarize, omit, or lose ANY content from the original subtitles. Keep all details, terms, and meaning intact.
2. **TEXT REFINEMENT:**
   - Correct obvious transcription mistakes and typos.
   - Fix punctuation naturally for subtitle readability.
3. **SRT FORMAT & CHARACTER LENGTH RULES:**
   - **STRICT SRT FORMAT:** Keep strictly valid SRT format (sequential index number, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, subtitle text, and a blank line between blocks). Do NOT use Markdown code fences (` ``` `) or Markdown headings (`##`).
   - **COMBINE SHORT SUBTITLES:** Combine consecutive short, choppy entries that form a continuous sentence into one natural block, as long as the combined length stays within approximately **{char_limit}** characters.
   - **CRITICAL TIMESTAMP MERGING RULE (DO NOT COPY PREMATURE END TIMESTAMPS):**
     When combining multiple short subtitle entries into one:
     * START TIMESTAMP: Take the start timestamp from the FIRST subtitle in the group.
     * END TIMESTAMP: You MUST look ahead and take the end timestamp from the LAST subtitle in the group.
     * FATAL ERROR TO AVOID: NEVER keep the first subtitle's end timestamp! If Subtitle 1 is `00:00:01,000 --> 00:00:02,200` and Subtitle 2 is `00:00:02,250 --> 00:00:04,800`, the merged entry MUST be `00:00:01,000 --> 00:00:04,800`. Outputting `00:00:01,000 --> 00:00:02,200` is completely incorrect!
   - **RE-INDEX SEQUENTIALLY:** Renumber all subtitle blocks sequentially starting from `1`.{part_rule}

Output ONLY the refined SRT content without any introductory or concluding comments.

Raw SRT Transcription:
{raw_text}
"""


def srt_chunk_system_instruction(part: int, total: int) -> str:
    return f"""CHUNK RULES:
- This is part {part} of {total} of one longer subtitle file.
- Do not add an introduction, conclusion, or Markdown headings.
- Output only the converted subtitle blocks for this part."""


def compose_prompt(system_instruction: str, contents: str) -> str:
    parts = [SOURCE_RESTRICTION]
    if system_instruction:
        parts.append(system_instruction)
    parts.append(contents)
    return "\n\n".join(parts)
