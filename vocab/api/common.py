"""Shared HTTP primitives for VocabuMe API endpoints.

This module deliberately stays framework-light: endpoint modules can reuse the
same request validation, Telegram identity resolution, and rate-limit policy
without importing the legacy ``vocab.views`` facade.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from django.conf import settings
from django.http import HttpRequest, JsonResponse

from core.env import get_telegram_token
from vocab.analytics import record_product_event
from vocab.models import TelegramUser
from vocab.ratelimit import RateLimitExceeded, enforce_rate_limit
from vocab.services import (
    build_user_progress,
    get_telegram_user_by_id,
    serialize_user,
)
from vocab.telegram_auth import TelegramAuthError, verify_webapp_init_data

SESSION_USER_KEY = "telegram_user_id"


def json_body(request: HttpRequest) -> dict:
    """Parse a bounded JSON object request body."""
    content_length = request.META.get("CONTENT_LENGTH")
    if content_length and int(content_length) > settings.MAX_JSON_BODY_BYTES:
        raise ValueError("JSON body is too large.")
    if not request.body:
        return {}
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def json_error(
    message: str, status: int = 400, *, code: str = "", **extra: object
) -> JsonResponse:
    payload: dict[str, object] = {"ok": False, "error": message}
    if code:
        payload["code"] = code
    payload.update(extra)
    return JsonResponse(payload, status=status)


def current_user(
    request: HttpRequest,
    *,
    verify_init_data: Callable[..., dict[str, Any]] = verify_webapp_init_data,
) -> TelegramUser | None:
    """Resolve a Telegram user, preferring verified Mini App identity to cookies."""
    init_data = request.headers.get("X-Telegram-Init-Data", "").strip()
    if init_data:
        try:
            verified = verify_init_data(
                init_data,
                get_telegram_token(),
                max_age_seconds=settings.TELEGRAM_AUTH_MAX_AGE_SECONDS,
            )
            telegram_user = verified.get("user") or {}
            telegram_id = int(telegram_user["id"])
        except (KeyError, TypeError, ValueError, TelegramAuthError):
            return None
        return TelegramUser.objects.filter(chat_id=telegram_id).first()

    user_id = request.session.get(SESSION_USER_KEY)
    return get_telegram_user_by_id(user_id) if user_id else None


def rate_limit_subject(request: HttpRequest, user: TelegramUser | None = None) -> str:
    if user is not None:
        return f"user:{user.id}"
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    client_ip = forwarded_for.split(",", 1)[0].strip() or request.META.get(
        "REMOTE_ADDR", ""
    )
    return f"ip:{client_ip or 'unknown'}"


def enforce_request_limit(
    request: HttpRequest,
    *,
    scope: str,
    limit: int,
    window: int,
    user: TelegramUser | None = None,
) -> JsonResponse | None:
    try:
        enforce_rate_limit(
            scope=scope,
            subject=rate_limit_subject(request, user),
            limit=limit,
            window=window,
        )
    except RateLimitExceeded:
        response = json_error(
            "Too many requests. Please try again shortly.", status=429
        )
        response["Retry-After"] = str(window)
        return response
    return None


def require_user(request: HttpRequest) -> TelegramUser | JsonResponse:
    user = current_user(request)
    return (
        user if user is not None else json_error("Authentication required.", status=401)
    )


def login(request: HttpRequest, user: TelegramUser) -> JsonResponse:
    request.session.cycle_key()
    request.session[SESSION_USER_KEY] = user.id
    record_product_event(user, "authenticated")
    return JsonResponse(
        {
            "ok": True,
            "user": serialize_user(user),
            "progress": build_user_progress(user),
        }
    )
