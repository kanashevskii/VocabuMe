"""Subscription and Telegram Stars payment application services.

The module owns billing state transitions. HTTP and Telegram handlers only
adapt their transport data and call these use cases.
"""

from __future__ import annotations

import secrets
from datetime import timedelta
from decimal import Decimal

from asgiref.sync import async_to_sync
from django.db import transaction
from django.utils import timezone
from telegram import Bot, LabeledPrice

from core.env import get_telegram_token
from vocab.models import (
    PaymentAttempt,
    SubscriptionPlan,
    TelegramUser,
    UserSubscription,
)
from vocab.monetization import (
    PLAN_DEFINITIONS,
    TELEGRAM_STARS_CURRENCY,
    get_telegram_stars_prices_for_user,
)


def sync_subscription_plans() -> list[SubscriptionPlan]:
    premium_price = PLAN_DEFINITIONS["premium"]["price"]
    plan_specs = [
        {
            "code": "premium_monthly",
            "name": "Premium Monthly",
            "billing_period": "monthly",
            "currency": premium_price["monthly"]["currency"],
            "price_amount": Decimal(premium_price["monthly"]["amount"]),
            "duration_days": 30,
        },
        {
            "code": "premium_yearly",
            "name": "Premium Yearly",
            "billing_period": "yearly",
            "currency": premium_price["yearly"]["currency"],
            "price_amount": Decimal(premium_price["yearly"]["amount"]),
            "duration_days": 365,
        },
    ]
    plans: list[SubscriptionPlan] = []
    for spec in plan_specs:
        plan, _ = SubscriptionPlan.objects.update_or_create(
            code=spec["code"],
            defaults={
                "name": spec["name"],
                "billing_period": spec["billing_period"],
                "currency": spec["currency"],
                "price_amount": spec["price_amount"],
                "duration_days": spec["duration_days"],
                "is_active": True,
                "metadata": {"base_plan_code": "premium"},
            },
        )
        plans.append(plan)
    return plans


def get_subscription_plans() -> list[SubscriptionPlan]:
    plans = list(
        SubscriptionPlan.objects.filter(is_active=True).order_by("price_amount", "id")
    )
    return plans or sync_subscription_plans()


def expire_user_subscriptions(user: TelegramUser) -> None:
    current_time = timezone.now()
    UserSubscription.objects.filter(
        user=user,
        status="active",
        expires_at__isnull=False,
        expires_at__lte=current_time,
    ).update(status="expired", updated_at=current_time)


def get_active_subscription(user: TelegramUser) -> UserSubscription | None:
    expire_user_subscriptions(user)
    return (
        UserSubscription.objects.select_related("plan")
        .filter(user=user, status="active")
        .order_by("-expires_at", "-id")
        .first()
    )


def user_has_premium(user: TelegramUser | None) -> bool:
    return user is not None and get_active_subscription(user) is not None


def serialize_subscription(subscription: UserSubscription | None) -> dict | None:
    if subscription is None:
        return None
    return {
        "plan_code": subscription.plan.code,
        "plan_name": subscription.plan.name,
        "billing_period": subscription.plan.billing_period,
        "status": subscription.status,
        "started_at": (
            subscription.started_at.isoformat() if subscription.started_at else None
        ),
        "expires_at": (
            subscription.expires_at.isoformat() if subscription.expires_at else None
        ),
        "activated_at": (
            subscription.activated_at.isoformat() if subscription.activated_at else None
        ),
    }


def get_billing_payload(user: TelegramUser) -> dict:
    active_subscription = get_active_subscription(user)
    plans = get_subscription_plans()
    return {
        "premium_active": active_subscription is not None,
        "active_subscription": serialize_subscription(active_subscription),
        "plans": [
            {
                "code": plan.code,
                "name": plan.name,
                "billing_period": plan.billing_period,
                "currency": plan.currency,
                "price_amount": format(plan.price_amount, ".2f"),
                "duration_days": plan.duration_days,
            }
            for plan in plans
        ],
    }


def get_plan_definition_for_user(user: TelegramUser | None) -> dict:
    return PLAN_DEFINITIONS["premium" if user_has_premium(user) else "free"]


def get_entitlements_for_user(user: TelegramUser | None) -> dict:
    return get_plan_definition_for_user(user)["entitlements"]


def _get_subscription_plan(plan_code: str, billing_period: str) -> SubscriptionPlan:
    normalized_period = (billing_period or "").strip().lower()
    if plan_code != "premium" or normalized_period not in {"monthly", "yearly"}:
        raise ValueError("Unsupported plan selection.")
    desired_code = f"premium_{normalized_period}"
    for plan in get_subscription_plans():
        if plan.code == desired_code:
            return plan
    raise ValueError("Subscription plan is unavailable.")


def _build_payment_payload(user: TelegramUser, plan: SubscriptionPlan) -> str:
    return f"sub:{user.id}:{plan.code}:{secrets.token_hex(8)}"


def _telegram_stars_amount_for_plan(user: TelegramUser, plan: SubscriptionPlan) -> int:
    amount = get_telegram_stars_prices_for_user(user.chat_id).get(plan.billing_period)
    if amount is None:
        raise ValueError("Telegram Stars price is unavailable.")
    return int(amount)


def create_bot_payment_attempt(
    user: TelegramUser, *, plan_code: str, billing_period: str
) -> dict:
    plan = _get_subscription_plan(plan_code, billing_period)
    payload = _build_payment_payload(user, plan)
    amount_minor = _telegram_stars_amount_for_plan(user, plan)
    attempt = PaymentAttempt.objects.create(
        user=user,
        plan=plan,
        provider="telegram",
        status="pending",
        invoice_payload=payload,
        amount_minor=amount_minor,
        currency=TELEGRAM_STARS_CURRENCY,
        metadata={
            "return_source": "bot",
            "catalog_price_amount": format(plan.price_amount, ".2f"),
            "catalog_currency": plan.currency,
            "payment_method": "telegram_stars",
        },
    )
    return {
        "attempt_id": attempt.id,
        "invoice_payload": payload,
        "amount_minor": amount_minor,
        "plan": plan,
        "currency": TELEGRAM_STARS_CURRENCY,
    }


def create_checkout_session(
    user: TelegramUser,
    *,
    plan_code: str,
    billing_period: str,
    return_source: str = "miniapp",
) -> dict:
    plan = _get_subscription_plan(plan_code, billing_period)
    prepared = create_bot_payment_attempt(
        user, plan_code=plan_code, billing_period=billing_period
    )
    payload = prepared["invoice_payload"]
    amount_minor = prepared["amount_minor"]

    bot = Bot(token=get_telegram_token())
    invoice_link = async_to_sync(bot.create_invoice_link)(
        title=f"{plan.name} for VocabuMe",
        description="Premium для безлимитного добавления и всех relocation-сценариев.",
        payload=payload,
        provider_token="",
        currency=TELEGRAM_STARS_CURRENCY,
        prices=[LabeledPrice(label=plan.name, amount=amount_minor)],
    )

    attempt = PaymentAttempt.objects.get(id=prepared["attempt_id"])
    attempt.invoice_link = invoice_link
    attempt.metadata = {**attempt.metadata, "return_source": return_source}
    attempt.save(update_fields=["invoice_link", "metadata", "updated_at"])
    return {
        "attempt_id": attempt.id,
        "invoice_payload": payload,
        "invoice_link": invoice_link,
        "plan": {
            "code": "premium",
            "billing_period": plan.billing_period,
            "price_amount": format(plan.price_amount, ".2f"),
            "currency": plan.currency,
            "telegram_stars_amount": amount_minor,
            "telegram_stars_currency": TELEGRAM_STARS_CURRENCY,
        },
    }


def activate_subscription_for_successful_payment(
    *,
    invoice_payload: str,
    telegram_payment_charge_id: str,
    provider_payment_charge_id: str,
    amount_minor: int,
    currency: str,
) -> UserSubscription:
    if not invoice_payload or not telegram_payment_charge_id:
        raise ValueError("Payment identifiers are required.")
    with transaction.atomic():
        attempt = (
            PaymentAttempt.objects.select_for_update()
            .select_related("user", "plan")
            .filter(invoice_payload=invoice_payload)
            .first()
        )
        if attempt is None:
            raise ValueError("Payment attempt not found.")
        if attempt.currency != currency or attempt.amount_minor != amount_minor:
            raise ValueError("Payment amount or currency does not match the invoice.")
        if attempt.status == "paid":
            existing = UserSubscription.objects.filter(
                invoice_payload=invoice_payload
            ).first()
            if existing is None:
                raise ValueError("Paid payment attempt has no subscription.")
            return existing
        if attempt.status != "pending":
            raise ValueError("Payment attempt cannot be activated.")
        if (
            PaymentAttempt.objects.filter(
                telegram_payment_charge_id=telegram_payment_charge_id
            )
            .exclude(pk=attempt.pk)
            .exists()
        ):
            raise ValueError("Telegram payment charge was already processed.")

        current_time = timezone.now()
        TelegramUser.objects.select_for_update().get(pk=attempt.user_id)
        attempt.status = "paid"
        attempt.paid_at = current_time
        attempt.telegram_payment_charge_id = telegram_payment_charge_id
        attempt.provider_payment_charge_id = provider_payment_charge_id
        attempt.save(
            update_fields=[
                "status",
                "paid_at",
                "telegram_payment_charge_id",
                "provider_payment_charge_id",
                "updated_at",
            ]
        )
        UserSubscription.objects.select_for_update().filter(
            user=attempt.user, status="active"
        ).update(status="expired", updated_at=current_time)
        subscription = UserSubscription.objects.create(
            user=attempt.user,
            plan=attempt.plan,
            status="active",
            started_at=current_time,
            activated_at=current_time,
            expires_at=current_time + timedelta(days=attempt.plan.duration_days),
            source="telegram",
            invoice_payload=invoice_payload,
            telegram_payment_charge_id=telegram_payment_charge_id,
            provider_payment_charge_id=provider_payment_charge_id,
            metadata={"attempt_id": attempt.id},
        )
        from vocab.analytics import record_product_event

        transaction.on_commit(
            lambda: record_product_event(
                attempt.user,
                "subscription_activated",
                properties={"billing_period": attempt.plan.billing_period},
            )
        )
        return subscription


def validate_telegram_pre_checkout(
    *, invoice_payload: str, amount_minor: int, currency: str
) -> tuple[bool, str]:
    """Validate a Telegram Stars invoice before Telegram charges the user."""
    attempt = PaymentAttempt.objects.filter(invoice_payload=invoice_payload).first()
    if attempt is None or attempt.status != "pending":
        return False, "Счёт недействителен или уже обработан. Откройте оплату заново."
    if attempt.amount_minor != amount_minor or attempt.currency != currency:
        return False, "Сумма счёта не совпадает. Откройте оплату заново."
    return True, ""
