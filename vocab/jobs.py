"""Database-backed durable jobs until the Redis worker transport is deployed."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.db import transaction
from django.conf import settings
from django.utils import timezone

from .models import BackgroundJob

logger = logging.getLogger(__name__)


def enqueue_job(
    *, kind: str, deduplication_key: str, payload: dict[str, Any], priority: int
) -> BackgroundJob:
    with transaction.atomic():
        job, created = BackgroundJob.objects.get_or_create(
            deduplication_key=deduplication_key,
            defaults={
                "kind": kind,
                "priority": priority,
                "payload": payload,
                "run_after": timezone.now(),
            },
        )
        requeued = False
        if not created and job.status == "failed":
            requeued = (
                BackgroundJob.objects.filter(id=job.id, status="failed").update(
                    status="queued",
                    attempts=0,
                    run_after=timezone.now(),
                    locked_at=None,
                    last_error="",
                )
                == 1
            )
            if requeued:
                job.refresh_from_db()

    if (created or requeued) and settings.CELERY_BROKER_URL:
        queue = "vocabume-high" if priority < 100 else "vocabume-low"

        def dispatch() -> None:
            from .tasks import (
                process_background_job_high,
                process_background_job_low,
            )

            task = (
                process_background_job_high
                if priority < 100
                else process_background_job_low
            )
            task.apply_async(args=[job.id], queue=queue)

        transaction.on_commit(dispatch)
    return job


def _execute(job: BackgroundJob) -> None:
    # Delayed imports avoid importing service integrations in web process startup.
    from . import services

    if job.kind == "draft_image":
        services._run_draft_image_generation(
            job.payload["draft_id"], job.payload["version"]
        )
    elif job.kind == "word_image":
        services._run_word_image_generation(
            job.payload["item_id"], job.payload["version"]
        )
    elif job.kind == "pack_prepare":
        services._prepare_pack_level_sync(
            job.payload["pack_id"],
            job.payload["level_id"],
            course_code=job.payload["course_code"],
        )
    elif job.kind == "image_optimize":
        services._optimize_image_to_webp(Path(job.payload["source_path"]))
    elif job.kind == "tts_audio":
        from .tts import generate_tts_audio

        asyncio.run(
            generate_tts_audio(
                job.payload["text"], language_code=job.payload["language_code"]
            )
        )
    else:
        raise ValueError(f"Unsupported background job kind: {job.kind}")


def _complete_job(job: BackgroundJob, *, external_retry: bool) -> None:
    try:
        _execute(job)
    except Exception as exc:
        logger.exception("Background job %s (%s) failed", job.id, job.kind)
        retry = job.attempts < job.max_attempts
        job.status = "queued" if retry else "failed"
        job.run_after = (
            timezone.now()
            if external_retry
            else timezone.now() + timedelta(seconds=min(300, 2**job.attempts))
        )
        job.last_error = str(exc)[:2_000]
        job.locked_at = None
        job.save(
            update_fields=[
                "status",
                "run_after",
                "last_error",
                "locked_at",
                "updated_at",
            ]
        )
        if external_retry and retry:
            raise
    else:
        job.status = "succeeded"
        job.locked_at = None
        job.last_error = ""
        job.save(update_fields=["status", "locked_at", "last_error", "updated_at"])


def _claim_job(queryset, now):
    job = queryset.select_for_update(skip_locked=True).first()
    if job is None:
        return None
    job.status = "running"
    job.locked_at = now
    job.attempts += 1
    job.save(update_fields=["status", "locked_at", "attempts", "updated_at"])
    return job


def run_one_job() -> bool:
    """Claim and execute one job. Safe for multiple worker processes on Postgres."""
    now = timezone.now()
    with transaction.atomic():
        job = _claim_job(
            BackgroundJob.objects.filter(status="queued", run_after__lte=now).order_by(
                "priority", "run_after", "id"
            ),
            now,
        )
        if job is None:
            return False
    _complete_job(job, external_retry=False)
    return True


def run_job_by_id(job_id: int, *, external_retry: bool = False) -> bool:
    with transaction.atomic():
        job = _claim_job(
            BackgroundJob.objects.filter(
                id=job_id, status="queued", run_after__lte=timezone.now()
            ),
            timezone.now(),
        )
    if job is None:
        return False
    _complete_job(job, external_retry=external_retry)
    return True
