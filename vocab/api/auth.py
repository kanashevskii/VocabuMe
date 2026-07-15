"""Telegram-only authentication and account-link HTTP endpoints."""

from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from core.env import get_telegram_bot_username, get_telegram_token
from vocab.api.common import (
    SESSION_USER_KEY,
    current_user,
    enforce_request_limit,
    json_body,
    json_error,
    login,
)
from vocab.services import (
    build_user_progress,
    consume_web_login_token,
    create_web_login_token,
    serialize_user,
    upsert_telegram_user,
)
from vocab.telegram_auth import (
    TelegramAuthError,
    verify_login_widget,
    verify_webapp_init_data,
)


@csrf_exempt
@require_POST
def auth_telegram_widget(request: HttpRequest) -> JsonResponse:
    if limited := enforce_request_limit(
        request, scope="auth-widget", limit=20, window=60
    ):
        return limited
    try:
        payload = json_body(request)
        verified = verify_login_widget(
            payload, get_telegram_token(), settings.TELEGRAM_AUTH_MAX_AGE_SECONDS
        )
    except (ValueError, TelegramAuthError) as exc:
        return json_error(str(exc), status=400)
    telegram_id = int(verified["id"])
    user = upsert_telegram_user(chat_id=telegram_id, username=verified.get("username"))
    return login(request, user)


@csrf_exempt
@require_POST
def auth_telegram_webapp(request: HttpRequest) -> JsonResponse:
    if limited := enforce_request_limit(
        request, scope="auth-webapp", limit=20, window=60
    ):
        return limited
    try:
        payload = json_body(request)
        verified = verify_webapp_init_data(
            payload.get("init_data", ""),
            get_telegram_token(),
            settings.TELEGRAM_AUTH_MAX_AGE_SECONDS,
        )
    except (ValueError, TelegramAuthError) as exc:
        return json_error(str(exc), status=400)
    telegram_user = verified.get("user")
    if not isinstance(telegram_user, dict):
        return json_error("Telegram user was not provided.", status=400)
    telegram_id = telegram_user.get("id")
    if isinstance(telegram_id, bool) or not isinstance(telegram_id, int):
        return json_error("Telegram user was not provided.", status=400)
    username = telegram_user.get("username")
    return login(
        request,
        upsert_telegram_user(
            chat_id=telegram_id,
            username=username if isinstance(username, str) else None,
        ),
    )


@require_POST
def auth_logout(request: HttpRequest) -> JsonResponse:
    request.session.flush()
    return JsonResponse({"ok": True})


@require_GET
def auth_me(request: HttpRequest) -> JsonResponse:
    user = current_user(request, verify_init_data=verify_webapp_init_data)
    return JsonResponse(
        {
            "ok": True,
            "authenticated": user is not None,
            "user": serialize_user(user) if user else None,
            "progress": build_user_progress(user) if user else None,
        }
    )


@require_POST
def auth_request_link(request: HttpRequest) -> JsonResponse:
    if limited := enforce_request_limit(request, scope="auth-link", limit=5, window=60):
        return limited
    login_token = create_web_login_token()
    return JsonResponse(
        {
            "ok": True,
            "token": login_token.token,
            "deep_link": f"https://t.me/{get_telegram_bot_username()}?start=login_{login_token.token}",
            "expires_at": login_token.expires_at.isoformat(),
        }
    )


@require_POST
def auth_web_register(request: HttpRequest) -> JsonResponse:
    return json_error(
        "Email/password registration is no longer supported. Use Telegram login.",
        status=410,
    )


@require_POST
def auth_web_login(request: HttpRequest) -> JsonResponse:
    return json_error(
        "Email/password login is no longer supported. Use Telegram login.", status=410
    )


@require_POST
def auth_poll_link(request: HttpRequest, token: str) -> JsonResponse:
    user = consume_web_login_token(token)
    if user is None:
        return JsonResponse({"ok": True, "authenticated": False})
    request.session.cycle_key()
    request.session[SESSION_USER_KEY] = user.id
    return JsonResponse(
        {
            "ok": True,
            "authenticated": True,
            "user": serialize_user(user),
            "progress": build_user_progress(user),
        }
    )
