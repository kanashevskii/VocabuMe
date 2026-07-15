from types import SimpleNamespace

import pytest

from vocab.models import OpenAIUsageEvent, TelegramUser
from vocab.openai_budget import OpenAIBudgetExceeded, openai_budget_reservation


class FakeRedis:
    def __init__(self):
        self.values: dict[str, int] = {}

    def eval(self, script, _num_keys, key, *args):
        if "current + amount > maximum" in script:
            amount, maximum, _ttl = map(int, args)
            current = self.values.get(key, 0)
            if current + amount > maximum:
                return 0
            self.values[key] = current + amount
            return 1
        if "local adjusted" in script:
            adjusted = self.values.get(key, 0) + int(args[0])
            if adjusted <= 0:
                self.values.pop(key, None)
            else:
                self.values[key] = adjusted
            return adjusted
        raise AssertionError("Unexpected Redis script")


@pytest.mark.django_db
def test_daily_budget_reserves_before_request_and_reconciles_actual_usage(monkeypatch):
    client = FakeRedis()
    user = TelegramUser.objects.create(chat_id=9913, username="budget-test")
    monkeypatch.setattr("vocab.openai_budget._redis_client", lambda: client)
    monkeypatch.setattr("vocab.openai_budget.OPENAI_DAILY_BUDGET_MICRO_USD", 100)
    monkeypatch.setattr("vocab.openai_budget.OPENAI_TEXT_REQUEST_RESERVE_MICRO_USD", 60)

    with openai_budget_reservation(
        "generate-word-data", model="gpt-5-mini", user_id=user.id
    ) as reservation:
        reservation._finalize(cost_microusd=40, usage_available=True)

    assert sum(client.values.values()) == 40
    event = OpenAIUsageEvent.objects.get()
    assert event.cost_microusd == 40
    assert event.user == user

    monkeypatch.setattr("vocab.openai_budget.OPENAI_TEXT_REQUEST_RESERVE_MICRO_USD", 70)
    with pytest.raises(OpenAIBudgetExceeded, match="daily budget is exhausted"):
        with openai_budget_reservation(
            "generate-word-data", model="gpt-5-mini", user_id=user.id
        ):
            pass


@pytest.mark.django_db
def test_chat_usage_is_persisted_with_current_model_pricing(monkeypatch):
    monkeypatch.setattr("vocab.openai_budget._redis_client", lambda: None)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
    )

    with openai_budget_reservation(
        "generate-word-data", model="gpt-5-mini", user_id=None
    ) as reservation:
        reservation.record_chat_response(response)

    event = OpenAIUsageEvent.objects.get()
    assert event.usage_available is True
    assert event.input_tokens == 1_000_000
    assert event.output_tokens == 1_000_000
    assert event.cost_microusd == 2_250_000
