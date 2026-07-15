"""Read-only image delivery endpoints for user-owned vocabulary assets."""

from __future__ import annotations

import logging
import mimetypes

from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.http import require_GET

from vocab.api.common import json_error, require_user
from vocab.services import (
    get_draft_image_file,
    get_user_draft,
    get_user_word,
    get_word_image_file,
)

logger = logging.getLogger(__name__)


@require_GET
def word_image(request: HttpRequest, word_id: int) -> JsonResponse | FileResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    item = get_user_word(user, word_id)
    if item is None:
        return json_error("Word not found.", status=404)
    image_path = get_word_image_file(item)
    if image_path is None:
        return json_error("Image not found.", status=404)

    content_type, _ = mimetypes.guess_type(image_path.name)
    try:
        return FileResponse(
            open(image_path, "rb"), content_type=content_type or "image/jpeg"
        )
    except OSError:
        logger.exception(
            "Image file open failed for user=%s word_id=%s path=%s",
            user.id,
            word_id,
            image_path,
        )
        return json_error("Image is temporarily unavailable.", status=503)


@require_GET
def draft_image(request: HttpRequest, draft_id: int) -> JsonResponse | FileResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user

    draft = get_user_draft(user, draft_id)
    if draft is None:
        return json_error("Draft not found.", status=404)
    image_path = get_draft_image_file(draft)
    if image_path is None:
        return json_error("Image not found.", status=404)

    content_type, _ = mimetypes.guess_type(image_path.name)
    try:
        return FileResponse(
            open(image_path, "rb"), content_type=content_type or "image/png"
        )
    except OSError:
        logger.exception(
            "Draft image open failed for user=%s draft_id=%s path=%s",
            user.id,
            draft_id,
            image_path,
        )
        return json_error("Image is temporarily unavailable.", status=503)
