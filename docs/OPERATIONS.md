# Operations runbook

## Deploy

GitHub Actions runs the full quality gate on every pull request and deploys a push to `main` only after that gate passes. The deployment script builds the frontend, synchronizes application code without `.env` or media, installs the locked production dependencies, runs `check --deploy` and migrations, restarts the high-priority worker when enabled, then restarts Gunicorn and checks `/api/app-config` locally.

The SSH transport account and runtime account may differ. `RUNTIME_USER` must be the account that owns the virtualenv, application files, and service-written media. Do not change it without verifying the systemd unit `User=` values.

After every production deployment:

1. Confirm GitHub Actions quality and deploy jobs are successful.
2. Check `https://vocabume.k1prod.com/api/app-config` and the changed endpoint.
3. Confirm the live frontend serves the newly built asset hash when frontend code changed.
4. Exercise a realistic affected user flow. For audio/assets, verify non-empty response bytes, not only a database path.
5. Inspect the relevant service and worker logs for errors without copying secrets.

## Rollback

1. Identify the last known-good Git commit and its completed deployment run.
2. Revert the offending commit with a new commit; do not rewrite shared history.
3. Let the normal quality gate and deploy path run, then repeat the smoke path.
4. If a migration is involved, do not reverse it blindly. First establish whether it is reversible and whether newer data depends on it; prefer a forward repair.

## Backups and restore drills

PostgreSQL backups are operationally mandatory. Keep encrypted, access-controlled off-host backups and test a restore into an isolated database regularly.

```bash
pg_dump --format=custom --file=vocabume-YYYY-MM-DD.dump "$DATABASE_URL"
createdb vocabume_restore_test
pg_restore --clean --if-exists --no-owner --dbname=vocabume_restore_test vocabume-YYYY-MM-DD.dump
```

Record the backup timestamp, restore duration, migration version, row-count sanity checks, and who validated the restore. Never use a restore drill against the live database.

## Error-log retention

Client diagnostics are authenticated and redacted before they are stored in `AppErrorLog`, but they still contain operational metadata. Run the following daily through the approved scheduler/operations process; do not enable a new worker or cron loop without an owner and a cost/retry review:

```bash
python manage.py purge_error_logs --days 30
```

The command deletes only records older than the selected retention window and reports the number deleted. Test it first against an isolated database when changing the retention period.

## Worker and queue safety

High-priority user work is isolated on `vocabume-high`. Do not send pack warm-up, backfill, or other paid/bulk work there. Before enabling low-priority workers or Celery beat, identify whether tasks call Telegram, OpenAI, TTS, or a payment provider; cap retries and concurrency first.

## Incident triage

For a 5xx/502, establish facts in this order: reverse proxy status, Gunicorn service status, local `/api/app-config`, recent application logs, database connectivity, Redis/worker status, then the changed deploy/migration. Use one batched SSH session for an investigation rather than repeated reconnects.
