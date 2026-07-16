"""First-party product analytics with bounded, non-sensitive event payloads."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db.models import Count
from django.utils import timezone

from vocab.models import ProductEvent, TelegramUser

CLIENT_EVENT_NAMES = frozenset({"paywall_opened"})
SERVER_EVENT_NAMES = frozenset(
    {
        "authenticated",
        "words_created",
        "pack_words_added",
        "practice_completed",
        "checkout_started",
        "subscription_activated",
    }
)
ALL_EVENT_NAMES = CLIENT_EVENT_NAMES | SERVER_EVENT_NAMES
MAX_PROPERTIES = 12
MAX_PROPERTY_KEY_LENGTH = 48
MAX_PROPERTY_VALUE_LENGTH = 160
EVENT_RETENTION_DAYS = 180
SENSITIVE_PROPERTY_KEYS = frozenset(
    {"token", "password", "secret", "init_data", "audio", "message", "text"}
)


def _clean_properties(properties: dict[str, Any] | None) -> dict[str, str | int | bool]:
    if not isinstance(properties, dict):
        return {}

    cleaned: dict[str, str | int | bool] = {}
    for key, value in properties.items():
        if len(cleaned) >= MAX_PROPERTIES or not isinstance(key, str):
            break
        key = key.strip().lower()
        if (
            not key
            or len(key) > MAX_PROPERTY_KEY_LENGTH
            or key in SENSITIVE_PROPERTY_KEYS
        ):
            continue
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, str):
            cleaned[key] = value.strip()[:MAX_PROPERTY_VALUE_LENGTH]
    return cleaned


def record_product_event(
    user: TelegramUser,
    name: str,
    *,
    properties: dict[str, Any] | None = None,
) -> ProductEvent:
    """Record a whitelisted event; never accept arbitrary event names or payloads."""
    if name not in ALL_EVENT_NAMES:
        raise ValueError("Unsupported product analytics event.")
    return ProductEvent.objects.create(
        user=user,
        name=name,
        properties=_clean_properties(properties),
    )


def build_funnel_report(*, days: int = 30) -> dict[str, object]:
    """Return a small admin/export-friendly funnel for a bounded recent window."""
    if days < 1 or days > 365:
        raise ValueError("days must be between 1 and 365.")
    since = timezone.now() - timedelta(days=days)
    rows = (
        ProductEvent.objects.filter(occurred_at__gte=since)
        .values("name")
        .annotate(events=Count("id"), users=Count("user_id", distinct=True))
    )
    by_name = {
        row["name"]: {"events": row["events"], "users": row["users"]} for row in rows
    }
    return {
        "since": since.isoformat(),
        "days": days,
        "events": {
            name: by_name.get(name, {"events": 0, "users": 0})
            for name in sorted(ALL_EVENT_NAMES)
        },
    }


def purge_expired_product_events(*, now=None) -> int:
    """Delete telemetry past its documented retention period."""
    cutoff = (now or timezone.now()) - timedelta(days=EVENT_RETENTION_DAYS)
    deleted, _ = ProductEvent.objects.filter(occurred_at__lt=cutoff).delete()
    return deleted
