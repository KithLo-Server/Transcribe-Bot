"""Telegram bot entrypoint for audio/video transcription and text refinement."""

import asyncio
import logging

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

import config
import core
import handlers
import state
from jobs import JobManager

logger = logging.getLogger("TranscribeBot")


async def _post_init(application: Application) -> None:
    logger.info("Warming up transcription models...")
    await asyncio.to_thread(core.warm_up)
    logger.info("Transcription models are ready.")
    state.jobs = JobManager(application.bot)
    await state.jobs.start()
    await state.jobs.resume()
    await handlers.resume_manual_sessions(application.bot)
    logger.info("Previous jobs restored.")


async def _post_shutdown(application: Application) -> None:
    if state.jobs:
        try:
            state.save_runtime()
        except Exception:
            logger.exception("Could not save runtime state during shutdown")
        await state.jobs.stop()


def _register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", handlers.cmd_start))
    application.add_handler(CommandHandler("status", handlers.cmd_status))
    application.add_handler(CommandHandler("refine", handlers.cmd_refine))
    application.add_handler(CommandHandler("exceed", handlers.cmd_exceed))
    application.add_handler(CallbackQueryHandler(handlers.handle_callback))
    application.add_handler(
        MessageHandler(
            filters.AUDIO
            | filters.VIDEO
            | filters.VOICE
            | filters.Document.AUDIO
            | filters.Document.VIDEO
            | filters.Document.FileExtension("txt")
            | filters.Document.FileExtension("srt"),
            handlers.handle_media,
        )
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_text)
    )


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is missing.")

    state.init_state()

    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .base_url(config.TELEGRAM_API_BASE_URL)
        .base_file_url(config.TELEGRAM_API_FILE_BASE_URL)
        .local_mode(True)
        .read_timeout(config.TELEGRAM_READ_TIMEOUT)
        .write_timeout(config.TELEGRAM_WRITE_TIMEOUT)
        .connect_timeout(config.TELEGRAM_CONNECT_TIMEOUT)
        .pool_timeout(config.TELEGRAM_POOL_TIMEOUT)
        .media_write_timeout(config.TELEGRAM_MEDIA_WRITE_TIMEOUT)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    _register_handlers(application)

    logger.info("Bot is starting polling...")
    application.run_polling()


if __name__ == "__main__":
    main()
