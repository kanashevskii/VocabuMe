# Contributing

## Before coding

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and the relevant service/model code. Preserve shared Telegram identity and progress across bot, Mini App, and website. Do not add a second identity system.

## Change rules

- Keep HTTP handlers thin; place reusable business rules in `vocab/services.py`.
- Add a Django migration for every model/index change. Do not edit an applied migration.
- Treat browser input, Telegram payloads, and payment callbacks as untrusted.
- Do not run paid/external I/O from a GET endpoint or a database transaction.
- Keep user-latency work on the high-priority queue; review cost before enabling background generation or scheduling.
- Never commit `.env`, credentials, media, database dumps, or private logs.

## Verification

Run `make quality`, the relevant migration checks, and focused tests. Frontend changes also require mobile viewport validation. Production-affecting changes need a post-deploy smoke test and a realistic user flow before they are complete.

## Pull requests

Explain the user-visible impact, migration/rollback plan, security implications, tests run, and any new environment variables. Keep dependency upgrades scoped to the task and update `requirements-prod.lock` only when required.
