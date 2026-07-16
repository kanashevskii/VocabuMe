# VocabuMe

VocabuMe is a shared language-learning product for relocants. It has three surfaces backed by one Django application: a Telegram bot, a Telegram Mini App, and a website. Telegram identity, dictionary state, subscription entitlement, and learning progress are shared across all of them.

Production: <https://vocabume.k1prod.com> · Bot: [@VocabuMe_bot](https://t.me/VocabuMe_bot) · API: [`/api/docs`](https://vocabume.k1prod.com/api/docs)

## Product and usage terms

VocabuMe focuses on practical English and Georgian for relocation: scenario packs, vocabulary capture, flashcards, multiple-choice practice, listening, irregular verbs, progress, reminders, and Premium access.

It is provided for **personal, non-commercial use only**. Commercial use, resale, sublicensing, paid access, white-label use, or use inside commercial products requires the author's prior written permission. See [TERMS.md](TERMS.md).

## Architecture

The backend is Django; the Mini App is React + Vite. The browser build in `frontend/dist` is served by Django. PostgreSQL is the source of truth; Redis provides the shared rate-limit/cache store and Celery broker/result backend.

```text
Telegram Bot ─────┐
Telegram Mini App ├─> Django HTTP API ─> application services ─> PostgreSQL
Website ──────────┘          │                       │
                             └─> Redis ─> Celery high-priority worker
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing identity, learning state, background work, or deployment topology.

## Repository map

- [frontend](frontend): React/Vite app shell for the Mini App and website.
- [vocab/services.py](vocab/services.py): shared business logic.
- [vocab/views.py](vocab/views.py): HTTP API and authentication flows.
- [vocab/telegram_auth.py](vocab/telegram_auth.py): verified Telegram auth helpers.
- [vocab/bot.py](vocab/bot.py): Telegram bot interactions.
- [vocab/openapi.py](vocab/openapi.py): hand-maintained public API contract.
- [vocab/migrations](vocab/migrations): Django database migrations.
- [deploy/systemd](deploy/systemd): versioned worker unit templates.

## Quick start

Prerequisites: Python 3.10+, Node.js 20+, PostgreSQL 16+, Redis 7+, and a Telegram bot token. Copy `.env.example` to `.env`; never commit the resulting file. Required values are validated at startup.

```bash
make install-dev
cd frontend && npm ci && cd ..
cp .env.example .env
python manage.py migrate --noinput
make build-frontend
DEBUG=true VOCABUME_PROCESS=web python run.py
```

Open <http://127.0.0.1:8000>. Start bot and worker processes separately:

```bash
VOCABUME_PROCESS=bot python run.py
celery -A core worker --loglevel=INFO --queues=vocabume-high --concurrency=2
```

For a containerized local stack, set real secrets in `.env` and run:

```bash
docker compose up --build migrate
docker compose up --build web bot worker-high
```

`worker-low` and `beat` are intentionally separate because they may trigger background generation or reminders; enable them only after reviewing cost and retry behaviour.

## Configuration

See [.env.example](.env.example) for the complete local template. In production set `DEBUG=False`, secure cookie/HSTS settings, `CSRF_TRUSTED_ORIGINS`, and use a shared Redis cache:

```dotenv
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1
CACHE_BACKEND=django.core.cache.backends.redis.RedisCache
CACHE_LOCATION=redis://127.0.0.1:6379/2
```

Do not print or paste `SECRET_KEY`, database passwords, Telegram tokens, payment provider tokens, or OpenAI keys into logs, issues, or pull requests.

## Quality gate

Run the production-equivalent checks before opening a pull request:

```bash
make quality
DEBUG=false python manage.py check --deploy
DJANGO_SETTINGS_MODULE=core.test_settings python manage.py makemigrations --check --dry-run
```

`make quality` verifies pinned dependencies, formatting, Ruff, MyPy, tests with coverage, dependency vulnerabilities, frontend linting, and the frontend build. The GitHub Actions workflow repeats this against PostgreSQL and deploys `main` only after it succeeds.

For frontend/mobile work, also test a real mobile viewport for Today, Learn, Words, Progress, and More, including a short viewport height.

## API and operations

The live OpenAPI contract and Swagger UI are public at [`/api/openapi.json`](https://vocabume.k1prod.com/api/openapi.json) and [`/api/docs`](https://vocabume.k1prod.com/api/docs). Mutating endpoints require verified Telegram Mini App `initData` or an authenticated session; the contract does not expose secrets.

Deployment, rollback, worker ownership, backups, and smoke-test expectations are documented in [docs/OPERATIONS.md](docs/OPERATIONS.md). Report a vulnerability according to [SECURITY.md](SECURITY.md), not through a public issue.

## First-party product analytics

Product analytics remains inside PostgreSQL: no third-party analytics SDK or user data export is enabled. The backend records authenticated session, dictionary, practice, checkout, and subscription events; the Mini App may submit only the whitelisted `paywall_opened` event through `POST /api/analytics/events`. Event properties are flat, bounded primitives; raw Telegram initData, tokens, audio, and free-form text must never be submitted.

Events are automatically purged after 180 days by the low-priority Celery worker. Staff can inspect individual events in Django Admin or obtain the current funnel without exposing raw rows:

```bash
python manage.py report_analytics_funnel --days 30
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep migrations reversible, preserve cross-surface state, and avoid dependency upgrades unrelated to the change.
