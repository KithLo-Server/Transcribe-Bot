"""Tests for locale resolution and message catalogs."""

import string
import unittest

import _bootstrap  # noqa: F401
import i18n


def _fields(template):
    return {
        name
        for _, name, _, _ in string.Formatter().parse(template)
        if name
    }


class LocaleResolutionTests(unittest.TestCase):
    def test_empty_or_missing_falls_back_to_english(self):
        self.assertEqual(i18n.resolve_locale(None), "en")
        self.assertEqual(i18n.resolve_locale(""), "en")
        self.assertEqual(i18n.resolve_locale("fr"), "en")

    def test_simplified_variants(self):
        for code in ("zh", "zh-hans", "zh-Hans", "zh-CN", "zh-sg"):
            self.assertEqual(i18n.resolve_locale(code), "zh-Hans")

    def test_traditional_variants(self):
        for code in ("zh-hant", "zh-Hant", "zh-HK", "zh-tw", "zh-mo"):
            self.assertEqual(i18n.resolve_locale(code), "zh-Hant")

    def test_unknown_chinese_variant_falls_back_to_english(self):
        self.assertEqual(i18n.resolve_locale("zh-latn"), "en")


class CatalogTests(unittest.TestCase):
    def test_every_locale_defines_every_english_key(self):
        for key in i18n._EN:
            for locale in i18n.LOCALES:
                self.assertIn(key, i18n._CATALOGS[locale], key)

    def test_english_message_is_unchanged(self):
        self.assertEqual(
            i18n.t("en", "wizard.send_file"),
            "Send an audio/video file or a .txt/.srt document to begin.",
        )

    def test_simplified_and_traditional_differ(self):
        simplified = i18n.t("zh-Hans", "auth.authenticated")
        traditional = i18n.t("zh-Hant", "auth.authenticated")
        self.assertIn("验证", simplified)
        self.assertIn("驗證", traditional)
        self.assertNotEqual(simplified, traditional)

    def test_formatting_arguments(self):
        self.assertEqual(
            i18n.t("en", "finalize.submitted", count=3),
            "Submitted 3 files.",
        )
        self.assertEqual(
            i18n.t("zh-Hant", "finalize.submitted", count=3),
            "已提交 3 個檔案。",
        )

    def test_unknown_key_returns_key(self):
        self.assertEqual(i18n.t("en", "missing.key"), "missing.key")

    def test_translations_use_the_same_placeholders_as_english(self):
        for key, english in i18n._EN.items():
            for locale in ("zh-Hans", "zh-Hant"):
                self.assertEqual(
                    _fields(i18n._CATALOGS[locale][key]),
                    _fields(english),
                    f"{locale} placeholder mismatch for {key}",
                )


if __name__ == "__main__":
    unittest.main()
