import asyncio
from types import SimpleNamespace

from vocab import bot
from vocab.integrations.telegram import payments


def test_bot_keeps_payment_handler_compatibility_imports():
    assert bot.subscribe is payments.subscribe
    assert bot.start_subscription_checkout is payments.start_subscription_checkout
    assert bot.handle_pre_checkout_query is payments.handle_pre_checkout_query
    assert bot.handle_successful_payment is payments.handle_successful_payment


def test_pre_checkout_handler_answers_with_service_validation(monkeypatch):
    received: dict[str, object] = {}

    def validate(**kwargs):
        assert kwargs == {
            "invoice_payload": "invoice-1",
            "amount_minor": 500,
            "currency": "XTR",
        }
        return False, "Счёт больше недействителен."

    class Query:
        invoice_payload = "invoice-1"
        total_amount = 500
        currency = "XTR"

        async def answer(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(payments, "validate_telegram_pre_checkout", validate)

    asyncio.run(
        payments.handle_pre_checkout_query(
            SimpleNamespace(pre_checkout_query=Query()), SimpleNamespace()
        )
    )

    assert received == {
        "ok": False,
        "error_message": "Счёт больше недействителен.",
    }
