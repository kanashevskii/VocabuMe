# VocabuMe

VocabuMe is an English learning product that now runs in three connected surfaces:
- Telegram bot
- Telegram Mini App
- Website with Telegram-based sign-in

All three surfaces share the same `TelegramUser`, dictionary, settings, and learning progress.

Links:
- Telegram bot: [@VocabuMe_bot](https://t.me/VocabuMe_bot)

## Usage terms

VocabuMe is provided for personal, non-commercial use only.

Commercial use, resale, sublicensing, paid access, white-label use, use inside commercial products, or any other commercial exploitation is prohibited without prior written permission from the author.

If you want to use VocabuMe commercially, obtain explicit written permission first.

## Current product scope

VocabuMe supports:
- adding words and phrases in batch
- flashcards
- multiple-choice practice
- listening mode
- irregular verbs training
- personal dictionary management
- shared progress and achievements
- reminders and learning settings

## Stack

- Python
- Django
- React
- Vite
- python-telegram-bot
- PostgreSQL
- OpenAI
- Nginx

## Repo structure

- [frontend](frontend) - React SPA for Telegram Mini App and website
- [vocab/bot.py](vocab/bot.py) - Telegram bot flows
- [vocab/views.py](vocab/views.py) - SPA entry and API endpoints
- [vocab/services.py](vocab/services.py) - shared application logic
- [vocab/telegram_auth.py](vocab/telegram_auth.py) - Telegram auth verification
- [run.py](run.py) - local combined runner for bot, reminders, and Django server
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) - prioritized execution backlog

## Environment

Minimum requirements:
- Python 3.10+
- PostgreSQL
- Node.js 20+
- Telegram bot token
- OpenAI API key

Key environment variables:
- `DEBUG`
- `SECRET_KEY`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `TELEGRAM_TOKEN`
- `TELEGRAM_BOT_USERNAME`
- `WEBAPP_URL`
- `OPENAI_API_KEY`
- `ALERT_CHAT_ID`
- `CSRF_TRUSTED_ORIGINS`

See [.env.example](.env.example).

Configuration rules:
- `SECRET_KEY`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `TELEGRAM_TOKEN`, and `OPENAI_API_KEY` are required.
- The app fails fast on startup if any required secret/config value is missing or empty.
- Generate a strong Django key with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`.

## Local setup

1. Install backend dependencies.

```bash
pip install -r requirements.txt
```

2. Install frontend dependencies.

```bash
cd frontend && npm install
```

3. Create `.env` from `.env.example`.

4. Apply migrations.

```bash
python manage.py migrate --noinput
```

5. Build frontend assets.

```bash
cd frontend && npm run build
```

6. Run the app locally.

```bash
python run.py
```

This starts:
- Telegram bot
- reminder loop
- Django server on `0.0.0.0:8000`

## Testing

Run the backend safety-net suite:

```bash
python -m pytest -q
```

Run the focused backend coverage report:

```bash
python -m pytest --cov=vocab.services --cov=vocab.views --cov=core.env --cov-report=term-missing -q
```

`pytest` uses [core/test_settings.py](core/test_settings.py), which switches tests to SQLite so the suite can run without PostgreSQL `CREATE DATABASE` privileges.

## Code Quality

Backend:

```bash
python -m black --check core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests
python -m flake8 core/env.py core/logging_config.py core/settings.py core/test_settings.py run.py vocab/services.py vocab/views.py vocab/openai_utils.py vocab/reminders.py vocab/management/commands/send_reminders.py tests
python -m mypy core/env.py core/settings.py core/test_settings.py run.py
```

Frontend:

```bash
cd frontend && npm run lint
cd frontend && npm run build
```

Shortcut commands:

```bash
make format-backend
make quality
```

## Frontend notes

The frontend is a mobile-first app shell intended for Telegram WebView first, but it also works as a normal website.

Auth paths:
- Telegram Mini App via `initData`
- web login via Telegram deep link token flow

The frontend build output is served by Django from `frontend/dist`.

## Production notes

Production checklist:
- set `DEBUG=False`
- build frontend with `cd frontend && npm ci && npm run build`
- configure `WEBAPP_URL`, `TELEGRAM_BOT_USERNAME`, and `CSRF_TRUSTED_ORIGINS`
- proxy `vocabume.k1prod.com` to Django
- serve `/static/` from `frontend/dist`
- enable SSL

For Telegram website login widget flows, the bot domain must be configured in BotFather:
- `/setdomain`
- `vocabume.k1prod.com`

## Operational notes

- If you update normalization logic for old words, run:

```bash
python scripts/clean_existing_words.py
```

- Local audio endpoints require TTS dependencies from [requirements.txt](requirements.txt).

## License

This repository is not provided under an open commercial license.

Use is limited by the terms above: personal non-commercial use only unless separate written permission is granted by the author.
