"""Inline keyboards and callback payloads for the transcription wizard."""

from typing import Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import i18n

CALLBACK_PREFIX = "tb"


def _callback(action: str, value: str = "") -> str:
    return f"{CALLBACK_PREFIX}:{action}" + (f":{value}" if value else "")


def parse_callback(data: str) -> Tuple[str, str, str]:
    """Returns (action, value, job_id) for a callback payload."""
    parts = (data or "").split(":")
    if len(parts) < 2 or parts[0] != CALLBACK_PREFIX:
        return "", "", ""
    action = parts[1]
    value = parts[2] if len(parts) > 2 else ""
    job_id = parts[3] if len(parts) > 3 else ""
    return action, value, job_id


def language_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "language.cantonese"),
                    callback_data=_callback("lang", "cantonese"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "language.mandarin"),
                    callback_data=_callback("lang", "mandarin"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "language.english"),
                    callback_data=_callback("lang", "english"),
                ),
            ]
        ]
    )


def format_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "format.text"),
                    callback_data=_callback("fmt", "text"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "format.srt"),
                    callback_data=_callback("fmt", "srt"),
                ),
            ]
        ]
    )


def zh_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "zh.traditional"),
                    callback_data=_callback("zh", "trad"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "zh.simplified"),
                    callback_data=_callback("zh", "simp"),
                ),
            ]
        ]
    )


def prompt_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "button.skip"),
                    callback_data=_callback("prompt", "skip"),
                )
            ]
        ]
    )


def refine_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "button.yes"),
                    callback_data=_callback("refine", "yes"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "button.no"),
                    callback_data=_callback("refine", "no"),
                ),
            ]
        ]
    )


def c2w_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "button.yes"),
                    callback_data=_callback("c2w", "yes"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "button.no"),
                    callback_data=_callback("c2w", "no"),
                ),
            ]
        ]
    )


def confirm_keyboard(locale: str = i18n.DEFAULT_LOCALE) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "button.confirm"),
                    callback_data=_callback("confirm", "yes"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "button.edit"),
                    callback_data=_callback("confirm", "edit"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, "button.cancel"),
                    callback_data=_callback("confirm", "cancel"),
                ),
            ]
        ]
    )


def _cache_keyboard(
    prefix: str,
    job_id: str,
    first: str,
    second: str,
    locale: str,
):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, first),
                    callback_data=_callback(prefix, f"use:{job_id}"),
                ),
                InlineKeyboardButton(
                    i18n.t(locale, second),
                    callback_data=_callback(prefix, f"new:{job_id}"),
                ),
            ]
        ]
    )


def raw_cache_keyboard(
    job_id: str, locale: str = i18n.DEFAULT_LOCALE
) -> InlineKeyboardMarkup:
    return _cache_keyboard(
        "raw",
        job_id,
        "button.use_transcription",
        "button.start_anew",
        locale,
    )


def refine_cache_keyboard(
    job_id: str, locale: str = i18n.DEFAULT_LOCALE
) -> InlineKeyboardMarkup:
    return _cache_keyboard(
        "ref",
        job_id,
        "button.use_refinement",
        "button.start_anew",
        locale,
    )


def refine_retry_keyboard(
    job_id: str, locale: str = i18n.DEFAULT_LOCALE
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    i18n.t(locale, "button.retry"),
                    callback_data=_callback("refine", f"retry:{job_id}"),
                )
            ]
        ]
    )
