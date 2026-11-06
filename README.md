# Transcribe Bot

A Telegram bot that transcribes audio and video, optionally refines the result with Google Gemini, and returns `.txt` or `.srt` files. It supports Cantonese, Mandarin, and English, handles batched uploads, and can also run entirely manual AI refinement without calling Gemini from the bot.

## Features

- Transcription of audio, video, voice messages, and `.txt`/`.srt` documents.
- Cantonese, Mandarin, and English input.
- Text or SRT output, with optional Traditional/Simplified Chinese conversion via OpenCC.
- Optional AI refinement through Gemini, including long-input chunk planning and parallel chunk processing.
- Batched uploads: one settings wizard applies to every file uploaded during the batch.
- Manual refinement mode (`/refine`) that builds prompts locally and accepts AI responses from any external chatbot.
- `/exceed` splits a pending manual prompt that is too large for an external AI.
- Security-question authentication with persisted lockouts.
- Localized messages in English, Simplified Chinese, and Traditional Chinese.
- Priority transcription queue, parallel refinement, and resumable jobs after a restart.
- Encrypted raw-transcription and refinement caches, with prompts to reuse existing results.
- Retry support for failed refinements and friendly messages for Gemini quota/model cooldowns.

## Commands

| Command | Description |
| --- | --- |
| `/start` | Authenticate with the configured security questions, or confirm that you are already authenticated. |
| `/status` | List your active jobs. The configured admin sees every user's jobs. |
| `/refine` | Start a manual AI refinement session by uploading a `.txt` or `.srt` file. |
| `/exceed` | Split the currently pending manual refinement prompt into smaller prompts. |

## How It Works

1. A user authenticates with `/start`.
2. The user uploads one or more media files or `.txt`/`.srt` documents.
3. The bot walks through language, output format, Chinese format, refinement, and related options, then shows a confirmation summary.
4. On confirmation, files enter a priority queue:
   - English is transcribed with `faster-whisper` (`medium.en`).
   - Cantonese and Mandarin are transcribed with FunASR `SenseVoiceSmall`, including VAD and punctuation models.
5. Raw transcriptions are cached and delivered as Telegram documents.
6. If refinement is enabled, Gemini refines the text, splitting long inputs into planned chunks when needed.
7. Refined output is cached and delivered, and can be reused for identical future requests.

Manual refinement follows the same wizard, but the bot sends prompts for the user to paste into an external AI. The user returns each response as a message or file, and the bot assembles the final document.

## Quick Start with Docker Compose

### Prerequisites

- Docker with Compose.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- A running Telegram Bot API server on the external `telegram_bot` network, reachable as `telegram_bot_api_server` (as referenced by `docker-compose.yml`).
- A Gemini API key for automatic refinement (optional if you only plan to use manual refinement).

### Environment

Create a `.env` file in the repository root. Compose reads these values:

```dotenv
TRANSCRIBE_BOT_TOKEN=123456:your-telegram-bot-token
TRANSCRIBE_ADMIN_USER_ID=123456789          # optional
GEMINI_API_KEY=key-one,key-two              # optional; comma-separated keys are rotated
TRANSCRIBE_DATA=/srv/transcribe             # host directory for bot data
TELEGRAM_BOT_API_DATA=/srv/telegram-bot-api # host directory for the Bot API server data
```

`TRANSCRIBE_BOT_TOKEN` and `GEMINI_API_KEY` are passed through from the environment. The compose file also accepts the optional Gemini model and cooldown overrides listed in the configuration table below.

### Security Questions

Authentication is required before uploads. Create `security_questions.json` inside the bot data directory (by default `$TRANSCRIBE_DATA/bot/security_questions.json`):

```json
[
  {
    "question": "What is the name of your first pet?",
    "answers": ["Fluffy", "fluffy"]
  },
  {
    "question": "In which city were you born?",
    "answers": ["Hong Kong"]
  }
]
```

The file must be a non-empty JSON array. Each entry needs a `question` and a non-empty `answers` array. The singular `answer` key is accepted for compatibility. Answers are matched case-insensitively with whitespace collapsed.

### Run

```bash
docker compose up -d --build
```

The container restarts unless stopped, runs as a lower-priority process, and persists uploads, caches, and state under the mounted data volume.

## Local Development

Python 3.12 is required.

System packages needed by the transcription stack:

- `ffmpeg`
- `libsndfile1`
- `libsox3` (or the equivalent `sox` runtime package for your distribution)
- `libgomp1`

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The source modules use flat imports, so run the bot from the `src` directory:

```bash
python src/bot.py
```

Make sure the required environment variables are exported and that `SECURITY_QUESTIONS_FILE` points to a valid questions file. Model caches default to the data directory and can be overridden with the standard Hugging Face and ModelScope environment variables.

## Configuration

All settings are environment variables read by `src/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `TRANSCRIBE_BOT_TOKEN` | _(required)_ | Telegram bot token. `TELEGRAM_BOT_TOKEN` is also accepted. |
| `TRANSCRIBE_ADMIN_USER_ID` | _(empty)_ | Telegram user ID allowed to see all jobs in `/status`. |
| `TELEGRAM_API_BASE_URL` | `https://api.telegram.org/bot` | Telegram Bot API base URL. |
| `TELEGRAM_API_FILE_BASE_URL` | `https://api.telegram.org/file/bot` | Telegram Bot API file URL. |
| `GEMINI_API_KEY` | _(empty)_ | One or more Gemini API keys, comma-separated. Keys are rotated in a round-robin pool. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model used for single-pass refinement. |
| `GEMINI_CHUNK_MODEL` | `gemini-3.1-flash-lite` | Model used for chunked refinement. |
| `GEMINI_C2W_MODEL` | `gemini-3.8-flash` | Model used for Cantonese-to-written-Chinese conversion. |
| `GEMINI_MODEL_COOLDOWN_SECONDS` | `60` | Cooldown after a model returns 503. |
| `GEMINI_QUOTA_COOLDOWN_SECONDS` | `60` | Cooldown for a key after a short-lived 429. |
| `GEMINI_DAILY_QUOTA_COOLDOWN_SECONDS` | `3600` | Cooldown for a key after daily quota exhaustion. |
| `CPU_THREADS` | `4` | CPU threads for transcription and model inference. |
| `REFINE_CONCURRENCY` | `10` | Maximum concurrent refinement jobs. |
| `REFINE_CHUNK_TOKENS` | `4000` | Target chunk size for refinement. |
| `REFINE_CHUNK_HARD_TOKENS` | `6000` | Hard chunk size limit before forcing a split. |
| `DATA_DIR` | `/data/bot` | Root directory for uploads, cache, and state. |
| `SECURITY_QUESTIONS_FILE` | `$DATA_DIR/security_questions.json` | Path to the authentication questions file. |
| `MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | Maximum accepted upload size. |
| `TELEGRAM_READ_TIMEOUT` | `120` | Telegram read timeout in seconds. |
| `TELEGRAM_WRITE_TIMEOUT` | `600` | Telegram write timeout in seconds. |
| `TELEGRAM_CONNECT_TIMEOUT` | `30` | Telegram connect timeout in seconds. |
| `TELEGRAM_POOL_TIMEOUT` | `30` | Telegram connection pool timeout in seconds. |
| `TELEGRAM_MEDIA_WRITE_TIMEOUT` | `600` | Telegram media upload timeout in seconds. |

## Data Layout

With the default `DATA_DIR=/data/bot`:

```
/data/bot/
├── cache/
│   ├── raw/              # AES-GCM encrypted raw transcriptions
│   └── refine/           # AES-GCM encrypted refinements
│       └── parts/        # Cached intermediate Gemini responses
├── state/
│   ├── auth.json         # Authenticated users and lockout counters
│   └── runtime.json      # Sessions, queued jobs, and completion ordering
├── uploads/              # Temporary uploads, cleaned up after 24 hours
└── security_questions.json
```

Cache files are encrypted with AES-GCM. Raw caches are keyed by the uploaded media digest and refinement caches by the converted transcription text, so identical inputs and settings can reuse prior results.

## Testing

The test suite uses the standard library `unittest` and does not require Telegram or model downloads:

```bash
python -m unittest discover -s tests -v
```

## Deployment

`.github/workflows/deploy.yml` deploys on pushes to `master` and on manual dispatch:

1. Checks out the repository on a self-hosted runner.
2. Verifies every variable referenced by `docker-compose.yml` is defined.
3. Runs `docker compose up -d --build --remove-orphans`.
4. Prints container status.

Required GitHub configuration:

- Secrets: `TRANSCRIBE_BOT_TOKEN`, `GEMINI_API_KEY`
- Variables: `TRANSCRIBE_DATA`, `TELEGRAM_BOT_API_DATA`, `TRANSCRIBE_ADMIN_USER_ID`

## Project Layout

| Path | Responsibility |
| --- | --- |
| `src/bot.py` | Application entrypoint, handler registration, startup/shutdown hooks. |
| `src/handlers.py` | Authentication, uploads, settings wizard, callbacks, manual refinement flow. |
| `src/jobs.py` | Priority transcription queue, parallel refinement, cache review queue. |
| `src/core.py` | Whisper/FunASR transcription, OpenCC conversion, Gemini refinement. |
| `src/manual.py` | Deterministic prompt building and output assembly for manual refinement. |
| `src/cache.py` | AES-GCM encrypted transcription and refinement cache. |
| `src/state.py` | Shared sessions, jobs, auth, caches, and runtime persistence. |
| `src/auth.py` | Security-question authentication and lockout management. |
| `src/i18n.py` | English, Simplified Chinese, and Traditional Chinese messages. |
| `src/chunking.py` | Text/SRT splitting and merging without network dependencies. |
| `src/prompts.py` | Gemini system instructions and prompt templates. |
| `src/delivery.py` | Telegram document delivery helpers. |
| `src/ui.py` | Inline keyboards and callback payloads. |
| `src/models.py` | Dataclasses shared across modules. |
| `src/storage.py` | Atomic JSON persistence helpers. |
| `src/config.py` | Environment configuration, logging setup, and filesystem locations. |
| `tests/` | Unit and flow tests for auth, cache, chunking, delivery, jobs, manual mode, and state. |

## Security Notes

- Keep `.env` out of version control; it is already listed in `.gitignore`.
- Uploads, caches, and state are stored under `DATA_DIR`; protect that directory with appropriate filesystem permissions.
- Authentication state and runtime state files are written atomically with `0600` permissions.
- Cache contents are encrypted at rest with AES-GCM.
- Failed authentication attempts trigger a one-hour lockout after five failures.
- API keys, bot tokens, and security answers are never written to the repository.
