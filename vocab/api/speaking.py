"""Speech-recognition learning endpoints and upload boundary."""

from __future__ import annotations

import logging
import os
import tempfile

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from vocab.api.common import enforce_request_limit, json_error, require_user
from vocab.openai_limits import openai_user_scope
from vocab.openai_utils import transcribe_speech_file
from vocab.services import (
    build_speaking_question,
    get_issued_speaking_question,
    submit_issued_speaking_answer,
)

logger = logging.getLogger(__name__)
MAX_AUDIO_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
}
ALLOWED_AUDIO_SUFFIXES = {".webm", ".ogg", ".mp3", ".wav", ".m4a", ".mp4"}


def _safe_unlink(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        logger.warning("Failed to remove speech upload temp file %s", path)


@require_GET
def speaking_question(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    question = build_speaking_question(user)
    if question is None:
        return JsonResponse({"ok": True, "empty": True})
    return JsonResponse({"ok": True, "question": question})


@require_POST
def speaking_answer(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    if limited := enforce_request_limit(
        request, scope="speech-transcription", limit=10, window=60, user=user
    ):
        return limited
    question_id = str(request.POST.get("question_id", ""))
    try:
        get_issued_speaking_question(user, question_id)
    except ValueError as exc:
        return json_error(str(exc), status=400)

    audio_file = request.FILES.get("audio")
    if audio_file is None:
        return json_error("Audio file is required.")
    if audio_file.size > MAX_AUDIO_UPLOAD_BYTES:
        return json_error("Audio file is too large.", status=413)
    if audio_file.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        return json_error("Unsupported audio format.", status=415)

    suffix = os.path.splitext(audio_file.name or "")[1].lower() or ".webm"
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        return json_error("Unsupported audio format.", status=415)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        with openai_user_scope(user.id):
            transcript = transcribe_speech_file(temp_path)
        return JsonResponse(
            {"ok": True, **submit_issued_speaking_answer(user, question_id, transcript)}
        )
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception:
        logger.exception(
            "Speaking recognition failed for user=%s question_id=%s",
            user.id,
            question_id,
        )
        return json_error("Speech recognition is temporarily unavailable.", status=503)
    finally:
        _safe_unlink(temp_path)
