"""Read-only media delivery and durable audio-preparation API endpoints."""

from __future__ import annotations

import logging
from hashlib import sha256

from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.alphabets import get_alphabet_letter
from vocab.api.common import enforce_request_limit, json_body, json_error, require_user
from vocab.jobs import enqueue_job
from vocab.models import TelegramUser
from vocab.services import get_active_course_code, get_user_word

logger = logging.getLogger(__name__)


def _serve_cached_audio(text: str, language_code: str) -> JsonResponse | FileResponse:
    """Serve a cached audio asset; GET handlers never generate audio."""
    from vocab.tts import get_audio_path, is_audio_ready

    audio_path = get_audio_path(text, language_code=language_code)
    if not is_audio_ready(text, language_code=language_code):
        return json_error(
            "Audio is being prepared.", status=404, code="audio_not_ready"
        )
    try:
        return FileResponse(open(audio_path, "rb"), content_type="audio/mpeg")
    except OSError:
        logger.exception("Cached audio file disappeared: %s", audio_path)
        return json_error("Audio is temporarily unavailable.", status=503)


def _enqueue_audio_generation(text: str, language_code: str) -> tuple[bool, int | None]:
    from vocab.tts import is_audio_ready

    if is_audio_ready(text, language_code=language_code):
        return True, None
    fingerprint = sha256(f"{language_code}:{text}".encode("utf-8")).hexdigest()
    job = enqueue_job(
        kind="tts_audio",
        deduplication_key=f"tts:{fingerprint}",
        payload={"text": text, "language_code": language_code},
        priority=10,
    )
    return False, job.id


@require_GET
def word_audio(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    item = get_user_word(user, word_id)
    if item is None:
        return json_error("Word not found.", status=404)
    return _serve_cached_audio(item.word, item.course_code)


@require_POST
def word_audio_prepare(request: HttpRequest, word_id: int) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="tts-prepare", limit=20, window=60, user=user
    ):
        return limited
    item = get_user_word(user, word_id)
    if item is None:
        return json_error("Word not found.", status=404)
    ready, job_id = _enqueue_audio_generation(item.word, item.course_code)
    return JsonResponse(
        {"ok": True, "ready": ready, "job_id": job_id},
        status=200 if ready else 202,
    )


def _get_alphabet_audio_text(
    user: TelegramUser, symbol: str
) -> tuple[str, str] | JsonResponse:
    if not symbol:
        return json_error("Alphabet symbol is required.", status=400)
    active_course = get_active_course_code(user)
    letter = get_alphabet_letter(active_course, symbol)
    if letter is None:
        return json_error("Alphabet letter not found.", status=404)
    return str(letter.get("name") or letter["symbol"]).strip(), active_course


@require_GET
def alphabet_audio(request: HttpRequest) -> JsonResponse | FileResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    audio_spec = _get_alphabet_audio_text(user, request.GET.get("symbol", "").strip())
    if isinstance(audio_spec, JsonResponse):
        return audio_spec
    return _serve_cached_audio(*audio_spec)


@require_POST
def alphabet_audio_prepare(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if limited := enforce_request_limit(
        request, scope="tts-prepare", limit=20, window=60, user=user
    ):
        return limited
    try:
        payload = json_body(request)
        symbol = str(payload.get("symbol", "")).strip()
    except (TypeError, ValueError):
        return json_error("Invalid alphabet audio payload.")
    audio_spec = _get_alphabet_audio_text(user, symbol)
    if isinstance(audio_spec, JsonResponse):
        return audio_spec
    ready, job_id = _enqueue_audio_generation(*audio_spec)
    return JsonResponse(
        {"ok": True, "ready": ready, "job_id": job_id},
        status=200 if ready else 202,
    )
