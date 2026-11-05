"""Tests for output document naming."""

import asyncio
import unittest

import _bootstrap  # noqa: F401
import delivery


class FakeBot:
    def __init__(self):
        self.documents = []

    async def send_document(self, chat_id, document, **kwargs):
        self.documents.append((chat_id, document))
        return None


class DeliveryTests(unittest.TestCase):
    def test_raw_and_refined_names_use_original_stem(self):
        async def run():
            bot = FakeBot()
            await delivery.send_text_document(
                bot, 1, "text", "Text", "talk.mp3", "raw"
            )
            await delivery.send_text_document(
                bot, 1, "text", "SRT", "talk.mp3", "refined"
            )
            return bot

        bot = asyncio.run(run())

        self.assertEqual(bot.documents[0][1].filename, "talk (raw).txt")
        self.assertEqual(bot.documents[1][1].filename, "talk (refined).srt")


if __name__ == "__main__":
    unittest.main()
