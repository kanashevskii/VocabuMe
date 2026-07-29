"""Transactional Telegram web-login token lifecycle."""

from __future__ import annotations

from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from vocab.models import TelegramUser, WebLoginToken


def create_web_login_token() -> WebLoginToken:
    return WebLoginToken.objects.create(
        expires_at=timezone.now() + timedelta(minutes=15)
    )


@transaction.atomic
def bind_web_login_token(token: str, user: TelegramUser) -> WebLoginToken | None:
    """Attach a Telegram user to a valid, still-unconsumed login token."""
    try:
        login_token = WebLoginToken.objects.select_for_update().get(
            token=token,
            expires_at__gt=timezone.now(),
            consumed_at__isnull=True,
        )
    except WebLoginToken.DoesNotExist:
        return None

    login_token.user = user
    login_token.save(update_fields=["user"])
    return login_token


@transaction.atomic
def consume_web_login_token(token: str) -> TelegramUser | None:
    """Consume a bound token exactly once, including under concurrent polling."""
    try:
        login_token = (
            WebLoginToken.objects.select_for_update()
            .select_related("user")
            .get(
                token=token,
                expires_at__gt=timezone.now(),
                consumed_at__isnull=True,
            )
        )
    except WebLoginToken.DoesNotExist:
        return None

    if login_token.user is None:
        return None

    consumed_at = timezone.now()
    claimed = WebLoginToken.objects.filter(
        pk=login_token.pk,
        consumed_at__isnull=True,
    ).update(consumed_at=consumed_at)
    if claimed != 1:
        return None
    return login_token.user
