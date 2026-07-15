"""Database-backed durable jobs until the Redis worker transport is deployed."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.db import transaction
from django.utils import timezone

from .models import BackgroundJob

logger = logging.getLogger(__name__)


def enqueue_job(
    *, kind: str, deduplication_key: str, payload: dict[str, Any], priority: int
) -> BackgroundJob:
    job, _ = BackgroundJob.objects.get_or_create(
        deduplication_key=deduplication_key,
        defaults={
            "kind": kind,
            "priority": priority,
            "payload": payload,
            "run_after": timezone.now(),
        },
    )
    return job


def _execute(job: BackgroundJob) -> None:
    # Delayed imports avoid importing service integrations in web process startup.
    from . import services

    if job.kind == "draft_image":
        services._run_draft_image_generation(job.payload["draft_id"], job.payload["version"])
    elif job.kind == "word_image":
        services._run_word_image_generation(job.payload["item_id"], job.payload["version"])
    elif job.kind == "pack_prepare":
        services._prepare_pack_level_sync(
            job.payload["pack_id"],
            job.payload["level_id"],
            course_code=job.payload["course_code"],
        )
    elif job.kind == "image_optimize":
        services._optimize_image_to_webp(Path(job.payload["source_path"]))
    else:
        raise ValueError(f"Unsupported background job kind: {job.kind}")


def run_one_job() -> bool:
    """Claim and execute one job. Safe for multiple worker processes on Postgres."""
    now = timezone.now()
    with transaction.atomic():
        job = (
            BackgroundJob.objects.select_for_update(skip_locked=True)
            .filter(status="queued", run_after__lte=now)
            .order_by("priority", "run_after", "id")
            .first()
        )
        if job is None:
            return False
        job.status = "running"
        job.locked_at = now
        job.attempts += 1
        job.save(update_fields=["status", "locked_at", "attempts", "updated_at"])

    try:
        _execute(job)
    except Exception as exc:
        logger.exception("Background job %s (%s) failed", job.id, job.kind)
        retry = job.attempts < job.max_attempts
        job.status = "queued" if retry else "failed"
        job.run_after = timezone.now() + timedelta(seconds=min(300, 2 ** job.attempts))
        job.last_error = str(exc)[:2_000]
        job.locked_at = None
        job.save(update_fields=["status", "run_after", "last_error", "locked_at", "updated_at"])
    else:
        job.status = "succeeded"
        job.locked_at = None
        job.last_error = ""
        job.save(update_fields=["status", "locked_at", "last_error", "updated_at"])
    return True
