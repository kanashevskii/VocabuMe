"""User settings and profile-avatar API endpoints."""

from __future__ import annotations

import logging
import mimetypes

from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.http import require_http_methods

from vocab.api.common import json_body, json_error, require_user
from vocab.services import (
    apply_user_settings,
    delete_user_avatar,
    get_profile_avatar_file,
    get_user_settings_payload,
    save_user_avatar,
    serialize_user,
)

logger = logging.getLogger(__name__)


@require_http_methods(["GET", "POST"])
def settings_view(request: HttpRequest) -> JsonResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if request.method == "GET":
        return JsonResponse({"ok": True, "settings": get_user_settings_payload(user)})
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))
    try:
        apply_user_settings(user, payload)
    except (TypeError, ValueError) as exc:
        return json_error(f"Invalid settings payload: {exc}")
    return JsonResponse({"ok": True})


@require_http_methods(["GET", "POST", "DELETE"])
def profile_avatar(request: HttpRequest) -> JsonResponse | FileResponse:
    user = require_user(request)
    if isinstance(user, JsonResponse):
        return user
    if request.method == "GET":
        avatar_path = get_profile_avatar_file(user)
        if avatar_path is None:
            return json_error("Avatar not found.", status=404)
        content_type, _ = mimetypes.guess_type(avatar_path.name)
        try:
            return FileResponse(
                open(avatar_path, "rb"), content_type=content_type or "image/webp"
            )
        except OSError:
            logger.exception(
                "Avatar file open failed for user=%s path=%s", user.id, avatar_path
            )
            return json_error("Avatar is temporarily unavailable.", status=503)
    if request.method == "DELETE":
        delete_user_avatar(user)
        return JsonResponse(
            {
                "ok": True,
                "user": serialize_user(user),
                "settings": get_user_settings_payload(user),
            }
        )
    uploaded_file = request.FILES.get("avatar")
    if uploaded_file is None:
        return json_error("Avatar file is required.")
    try:
        save_user_avatar(user, uploaded_file)
    except ValueError as exc:
        return json_error(str(exc))
    return JsonResponse(
        {
            "ok": True,
            "user": serialize_user(user),
            "settings": get_user_settings_payload(user),
        }
    )
