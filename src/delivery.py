"""Telegram document delivery helpers."""

from pathlib import Path

from telegram import InputFile

import i18n


async def send_text_document(
    bot,
    chat_id: int,
    text: str,
    fmt: str,
    original_name: str,
    kind: str,
    locale: str = i18n.DEFAULT_LOCALE,
):
    extension = "srt" if fmt == "SRT" else "txt"
    stem = Path(original_name).stem or original_name or "transcription"
    kind_label = (
        i18n.t(locale, f"document.{kind}")
        if kind in ("raw", "refined")
        else kind
    )
    data = (text or "").encode("utf-8")
    document = InputFile(
        data, filename=f"{stem} ({kind_label}).{extension}"
    )
    return await bot.send_document(chat_id=chat_id, document=document)
