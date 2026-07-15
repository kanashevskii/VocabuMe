from __future__ import annotations

import pytest

from vocab.jobs import enqueue_job, run_one_job
from vocab.models import BackgroundJob


@pytest.mark.django_db
def test_enqueue_job_is_idempotent():
    first = enqueue_job(
        kind="word_image",
        deduplication_key="word-image:1:2",
        payload={"item_id": 1, "version": 2},
        priority=10,
    )
    second = enqueue_job(
        kind="word_image",
        deduplication_key="word-image:1:2",
        payload={"item_id": 1, "version": 2},
        priority=10,
    )

    assert first.id == second.id
    assert BackgroundJob.objects.count() == 1


@pytest.mark.django_db
def test_run_one_job_marks_success(monkeypatch):
    job = enqueue_job(
        kind="word_image",
        deduplication_key="word-image:2:1",
        payload={"item_id": 2, "version": 1},
        priority=10,
    )
    monkeypatch.setattr(
        "vocab.services._run_word_image_generation", lambda item_id, version: None
    )

    assert run_one_job() is True

    job.refresh_from_db()
    assert job.status == "succeeded"
    assert job.attempts == 1


@pytest.mark.django_db
def test_run_one_job_retries_after_failure(monkeypatch):
    job = enqueue_job(
        kind="word_image",
        deduplication_key="word-image:3:1",
        payload={"item_id": 3, "version": 1},
        priority=10,
    )
    monkeypatch.setattr(
        "vocab.services._run_word_image_generation",
        lambda item_id, version: (_ for _ in ()).throw(RuntimeError("temporary")),
    )

    assert run_one_job() is True

    job.refresh_from_db()
    assert job.status == "queued"
    assert job.attempts == 1
    assert "temporary" in job.last_error


@pytest.mark.django_db
def test_run_one_job_generates_tts_in_worker(monkeypatch):
    job = enqueue_job(
        kind="tts_audio",
        deduplication_key="tts:example",
        payload={"text": "apple", "language_code": "en"},
        priority=10,
    )
    generated: list[tuple[str, str]] = []

    async def fake_generate(text: str, language_code: str) -> str:
        generated.append((text, language_code))
        return "/tmp/apple.mp3"

    monkeypatch.setattr("vocab.tts.generate_tts_audio", fake_generate)

    assert run_one_job() is True
    job.refresh_from_db()
    assert generated == [("apple", "en")]
    assert job.status == "succeeded"
