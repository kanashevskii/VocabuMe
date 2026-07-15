from __future__ import annotations

from celery import shared_task
from django.core.management import call_command

from .jobs import run_job_by_id


@shared_task(
    bind=True,
    name="vocab.tasks.process_background_job_high",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=270,
    time_limit=300,
)
def process_background_job_high(self, job_id: int) -> bool:
    return run_job_by_id(job_id, external_retry=True)


@shared_task(
    bind=True,
    name="vocab.tasks.process_background_job_low",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=270,
    time_limit=300,
)
def process_background_job_low(self, job_id: int) -> bool:
    return run_job_by_id(job_id, external_retry=True)


@shared_task(
    name="vocab.tasks.send_reminders",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=120,
    time_limit=150,
)
def send_reminders() -> None:
    call_command("send_reminders")
