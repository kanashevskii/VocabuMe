# VocabuMe architecture

## Product boundary

VocabuMe is one product with three clients: the Telegram bot, Telegram Mini App, and website. `TelegramUser` is the common identity. Dictionary entries, learning progress, course selection, and entitlement state belong to that user and must not be duplicated per surface.

## Application layers

| Layer | Responsibility | Primary location |
| --- | --- | --- |
| Delivery | Django HTTP handlers, serialization, request limits | `vocab/views.py` |
| Authentication | Telegram initData verification and session bridging | `vocab/telegram_auth.py` |
| Application | Shared use cases and transactions | `vocab/services.py` |
| Domain/persistence | Django models, constraints, migrations | `vocab/models.py`, `vocab/migrations/` |
| Integrations | Bot API, TTS, OpenAI, payments | `vocab/bot.py`, `vocab/tts.py`, `vocab/openai_utils.py` |
| Background execution | Durable jobs and Celery task dispatch | `vocab/jobs.py`, `vocab/tasks.py` |

Views and bot handlers may validate/translate transport data, but business rules belong in services so every client gets identical behaviour. New external I/O must be isolated behind an integration helper and never occur on a read endpoint.

## Runtime topology

Production runs Gunicorn for Django, a separate Telegram bot process, Redis, and a high-priority Celery worker. The high worker owns user-latency-sensitive work such as queued audio preparation. PostgreSQL is the durable source of truth.

`vocabume-worker-low` and `vocabume-beat` templates are versioned but not enabled by default: low-priority pack generation can consume paid API quota, and beat dispatches reminders. Before enabling either, review task cost, idempotency, retry limits, queue isolation, and an operational owner.

## Trust boundaries

- Browser-provided Telegram `initData` is untrusted until server-side HMAC verification and freshness checks pass.
- Session cookies are a transport convenience, not a separate identity system.
- Browser score, streak, payment, and entitlement claims are untrusted; the server computes and persists authoritative results.
- Redis improves rate limiting and queueing but cannot be the sole durable record of an action that affects user progress or billing.

## Data and performance rules

- Use migrations for every schema/index change; inspect PostgreSQL `EXPLAIN (ANALYZE, BUFFERS)` for hot queries before declaring an index effective.
- Avoid `order_by("?")`, unbounded scans, and date casts on indexed timestamps.
- Use `select_related`/`prefetch_related` deliberately and test query counts for list endpoints.
- Background jobs must be idempotent and persist enough state to recover from a worker restart.

## API contract

`vocab/openapi.py` is the reviewed source for the public HTTP contract. Update it with every public endpoint or response-shape change and add a focused endpoint test. The served contract is `/api/openapi.json`; Swagger UI is `/api/docs`.
