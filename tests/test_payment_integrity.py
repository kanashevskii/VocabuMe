import pytest

from vocab.models import PaymentAttempt, TelegramUser, UserSubscription
from vocab.services import (
    activate_subscription_for_successful_payment,
    create_bot_payment_attempt,
    validate_telegram_pre_checkout,
)


def _payment_attempt(user: TelegramUser) -> dict:
    return create_bot_payment_attempt(
        user, plan_code="premium", billing_period="monthly"
    )


def _activate(attempt: dict, *, charge_id: str = "telegram-charge"):
    return activate_subscription_for_successful_payment(
        invoice_payload=attempt["invoice_payload"],
        telegram_payment_charge_id=charge_id,
        provider_payment_charge_id=f"provider-{charge_id}",
        amount_minor=attempt["amount_minor"],
        currency=attempt["currency"],
    )


@pytest.mark.django_db
def test_payment_activation_is_idempotent_for_telegram_delivery_retries():
    user = TelegramUser.objects.create(chat_id=31_001, username="payment-retry")
    attempt = _payment_attempt(user)

    first = _activate(attempt)
    second = _activate(attempt)

    assert second.id == first.id
    assert UserSubscription.objects.filter(user=user, status="active").count() == 1
    assert PaymentAttempt.objects.get(id=attempt["attempt_id"]).status == "paid"


@pytest.mark.django_db
def test_payment_activation_rejects_charge_id_already_bound_to_another_invoice():
    first_user = TelegramUser.objects.create(chat_id=31_002, username="payment-one")
    second_user = TelegramUser.objects.create(chat_id=31_003, username="payment-two")
    first_attempt = _payment_attempt(first_user)
    second_attempt = _payment_attempt(second_user)

    _activate(first_attempt, charge_id="shared-telegram-charge")

    with pytest.raises(ValueError, match="already processed"):
        _activate(second_attempt, charge_id="shared-telegram-charge")

    assert (
        PaymentAttempt.objects.get(id=second_attempt["attempt_id"]).status == "pending"
    )
    assert UserSubscription.objects.filter(user=second_user).count() == 0


@pytest.mark.django_db
def test_payment_activation_rejects_wrong_amount_without_marking_attempt_paid():
    user = TelegramUser.objects.create(chat_id=31_004, username="payment-amount")
    attempt = _payment_attempt(user)

    with pytest.raises(ValueError, match="amount or currency"):
        activate_subscription_for_successful_payment(
            invoice_payload=attempt["invoice_payload"],
            telegram_payment_charge_id="wrong-amount-charge",
            provider_payment_charge_id="wrong-amount-provider",
            amount_minor=attempt["amount_minor"] + 1,
            currency=attempt["currency"],
        )

    assert PaymentAttempt.objects.get(id=attempt["attempt_id"]).status == "pending"
    assert UserSubscription.objects.filter(user=user).count() == 0


@pytest.mark.django_db
def test_pre_checkout_rejects_an_attempt_that_was_already_paid():
    user = TelegramUser.objects.create(chat_id=31_005, username="payment-precheckout")
    attempt = _payment_attempt(user)

    assert validate_telegram_pre_checkout(
        invoice_payload=attempt["invoice_payload"],
        amount_minor=attempt["amount_minor"],
        currency=attempt["currency"],
    ) == (True, "")

    _activate(attempt)

    valid, reason = validate_telegram_pre_checkout(
        invoice_payload=attempt["invoice_payload"],
        amount_minor=attempt["amount_minor"],
        currency=attempt["currency"],
    )
    assert valid is False
    assert "недействителен" in reason
