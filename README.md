# VocabuMe

VocabuMe is an English learning product that now runs in three connected surfaces:
- Telegram bot
- Telegram Mini App
- Website with Telegram-based sign-in

All three surfaces share the same `TelegramUser`, dictionary, settings, and learning progress.

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

- [frontend](/Users/eduard/PycharmProjects/englishbot/frontend) - React SPA for Telegram Mini App and website
- [vocab/bot.py](/Users/eduard/PycharmProjects/englishbot/vocab/bot.py) - Telegram bot flows
- [vocab/views.py](/Users/eduard/PycharmProjects/englishbot/vocab/views.py) - SPA entry and API endpoints
- [vocab/services.py](/Users/eduard/PycharmProjects/englishbot/vocab/services.py) - shared application logic
- [vocab/telegram_auth.py](/Users/eduard/PycharmProjects/englishbot/vocab/telegram_auth.py) - Telegram auth verification
- [run.py](/Users/eduard/PycharmProjects/englishbot/run.py) - local combined runner for bot, reminders, and Django server

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

See [.env.example](/Users/eduard/PycharmProjects/englishbot/.env.example).

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

- Local audio endpoints require TTS dependencies from [requirements.txt](/Users/eduard/PycharmProjects/englishbot/requirements.txt).

## License

MIT
