"""Safe application-error recording and the client diagnostic endpoint."""

from __future__ import annotations

import logging

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from vocab.api.common import (
    current_user,
    enforce_request_limit,
    json_body,
    json_error,
)
from vocab.models import AppErrorLog, TelegramUser

logger = logging.getLogger(__name__)

MAX_CLIENT_ERROR_BODY_BYTES = 16 * 1024
MAX_CLIENT_ERROR_MESSAGE_LENGTH = 1_000
CLIENT_ERROR_CATEGORIES = {"client", "network", "ui", "api"}
CLIENT_ERROR_LEVELS = {"warning", "error"}
_SENSITIVE_CONTEXT_KEYS = {"password", "token", "init_data", "audio"}


def safe_context(payload: dict | None) -> dict:
    """Bound and redact context before it crosses into the persistent error log."""
    if not payload:
        return {}
    safe: dict[str, object] = {}
    for key, value in payload.items():
        if key.lower() in _SENSITIVE_CONTEXT_KEYS:
            continue
        if isinstance(value, str):
            safe[key] = value[:500]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def log_app_error(
    request: HttpRequest,
    *,
    message: str,
    category: str = "server",
    level: str = "error",
    status_code: int | None = None,
    user: TelegramUser | None = None,
    context: dict | None = None,
) -> None:
    """Persist diagnostics without allowing logging failures to affect requests."""
    try:
        AppErrorLog.objects.create(
            user=user or current_user(request),
            category=category,
            level=level,
            message=message[:4000],
            path=request.path[:255],
            method=request.method[:10],
            status_code=status_code,
            context=safe_context(context),
        )
    except Exception:
        logger.exception("Failed to persist AppErrorLog for %s", request.path)


@csrf_exempt
@require_POST
def client_error_log(request: HttpRequest) -> JsonResponse:
    """Accept a bounded, authenticated client diagnostic without secret leakage."""
    user = current_user(request)
    if user is None:
        return json_error("Authentication required.", status=401)
    if limited := enforce_request_limit(
        request, scope="client-error", limit=30, window=60, user=user
    ):
        return limited
    if int(request.META.get("CONTENT_LENGTH") or 0) > MAX_CLIENT_ERROR_BODY_BYTES:
        return json_error("Client error payload is too large.", status=413)
    try:
        payload = json_body(request)
    except ValueError as exc:
        return json_error(str(exc))

    category = str(payload.get("category") or "client").lower()
    level = str(payload.get("level") or "error").lower()
    if category not in CLIENT_ERROR_CATEGORIES or level not in CLIENT_ERROR_LEVELS:
        return json_error("Invalid client error payload.")
    status_code = payload.get("status_code")
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None
    log_app_error(
        request,
        user=user,
        category=category,
        level=level,
        status_code=status_code,
        message=str(payload.get("message") or "Client-side error")[
            :MAX_CLIENT_ERROR_MESSAGE_LENGTH
        ],
        context={
            "url": payload.get("url", ""),
            "detail": str(payload.get("detail", ""))[:1_000],
            "meta": payload.get("meta", {}),
        },
    )
    return JsonResponse({"ok": True})
