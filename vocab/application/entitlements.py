"""Premium access gates and concurrency-safe daily entitlement accounting."""

from __future__ import annotations

from datetime import date, datetime, time as datetime_time, timedelta

from django.db import IntegrityError, transaction
from django.db.models import F
from django.utils import timezone

from vocab.application.billing import get_entitlements_for_user, user_has_premium
from vocab.models import TelegramUser, UserDailyEntitlementUsage, VocabularyItem


class EntitlementError(ValueError):
    def __init__(self, code: str, message: str, *, paywall_trigger: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.paywall_trigger = paywall_trigger


def _application_day_window(day: date) -> tuple[datetime, datetime]:
    """Return the application's timezone-aware UTC-compatible day range."""
    app_timezone = timezone.get_current_timezone()
    start = datetime.combine(day, datetime_time.min, tzinfo=app_timezone)
    return start, start + timedelta(days=1)


def get_daily_entitlement_usage(
    user: TelegramUser, target_date: date | None = None
) -> UserDailyEntitlementUsage:
    usage_date = target_date or timezone.localdate()
    day_start, day_end = _application_day_window(usage_date)
    usage, _ = UserDailyEntitlementUsage.objects.get_or_create(
        user=user,
        usage_date=usage_date,
        defaults={
            "new_items_added": VocabularyItem.objects.filter(
                user=user,
                created_at__gte=day_start,
                created_at__lt=day_end,
            ).count(),
            "extra_image_regenerations": 0,
        },
    )
    return usage


def get_remaining_new_items_for_today(user: TelegramUser) -> int | None:
    max_items = get_entitlements_for_user(user).get("max_new_items_per_day")
    if max_items is None:
        return None
    usage = get_daily_entitlement_usage(user)
    return max(0, int(max_items) - int(usage.new_items_added or 0))


def get_remaining_extra_image_regenerations_for_today(user: TelegramUser) -> int | None:
    max_regenerations = get_entitlements_for_user(user).get(
        "max_extra_image_regenerations_per_day"
    )
    if max_regenerations is None:
        return None
    usage = get_daily_entitlement_usage(user)
    return max(0, int(max_regenerations) - int(usage.extra_image_regenerations or 0))


def reserve_new_items_for_today(user: TelegramUser, count: int) -> None:
    if count <= 0:
        return
    usage_date = timezone.localdate()
    day_start, day_end = _application_day_window(usage_date)
    with transaction.atomic():
        try:
            usage, _ = UserDailyEntitlementUsage.objects.get_or_create(
                user=user,
                usage_date=usage_date,
                defaults={
                    "new_items_added": VocabularyItem.objects.filter(
                        user=user,
                        created_at__gte=day_start,
                        created_at__lt=day_end,
                    ).count(),
                },
            )
        except IntegrityError:
            usage = UserDailyEntitlementUsage.objects.get(
                user=user, usage_date=usage_date
            )
        usage = UserDailyEntitlementUsage.objects.select_for_update().get(pk=usage.pk)
        max_items = get_entitlements_for_user(user).get("max_new_items_per_day")
        if max_items is not None and usage.new_items_added + count > int(max_items):
            raise EntitlementError(
                "paywall_daily_new_items_limit",
                "В free-плане можно добавить до 10 новых слов и фраз в день. Открой Premium, чтобы снять лимит.",
                paywall_trigger="daily_new_items_limit",
            )
        UserDailyEntitlementUsage.objects.filter(pk=usage.pk).update(
            new_items_added=F("new_items_added") + count
        )


def reserve_extra_image_regeneration_for_today(user: TelegramUser) -> None:
    usage_date = timezone.localdate()
    with transaction.atomic():
        try:
            usage, _ = UserDailyEntitlementUsage.objects.get_or_create(
                user=user, usage_date=usage_date
            )
        except IntegrityError:
            usage = UserDailyEntitlementUsage.objects.get(
                user=user, usage_date=usage_date
            )
        usage = UserDailyEntitlementUsage.objects.select_for_update().get(pk=usage.pk)
        max_regenerations = get_entitlements_for_user(user).get(
            "max_extra_image_regenerations_per_day"
        )
        if max_regenerations is not None and usage.extra_image_regenerations >= int(
            max_regenerations
        ):
            raise EntitlementError(
                "paywall_extra_image_regeneration_limit",
                "В free-плане закончились дополнительные обновления фото на сегодня. Открой Premium, чтобы снять лимит.",
                paywall_trigger="extra_image_regeneration_limit",
            )
        UserDailyEntitlementUsage.objects.filter(pk=usage.pk).update(
            extra_image_regenerations=F("extra_image_regenerations") + 1
        )


def pack_requires_premium(
    user: TelegramUser | None, pack_definition: dict | None
) -> bool:
    if not pack_definition or user_has_premium(user):
        return False
    if pack_definition.get("track") != "relocation":
        return False
    return not bool(pack_definition.get("starter_pack"))


def ensure_pack_is_accessible(user: TelegramUser, pack_definition: dict | None) -> None:
    if not pack_requires_premium(user, pack_definition):
        return
    raise EntitlementError(
        "paywall_premium_pack_gate",
        "Этот сценарий доступен в Premium. Открой полный доступ к сценариям для переезда.",
        paywall_trigger="premium_pack_gate",
    )
