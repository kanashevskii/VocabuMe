"""Cross-process OpenAI spend guard and auditable usage accounting."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from django.utils import timezone
from redis import Redis
from redis.exceptions import RedisError

from core.env import env
from vocab.models import OpenAIUsageEvent
from vocab.openai_limits import _redis_client

logger = logging.getLogger(__name__)

MICRO_USD = Decimal("1000000")
TEXT_PRICES_PER_MILLION_USD = {
    "gpt-5-mini": (Decimal("0.25"), Decimal("2.00"), Decimal("0.025")),
    "gpt-4o-mini-transcribe": (Decimal("1.25"), Decimal("5.00"), Decimal("0")),
}
IMAGE_PRICES_USD = {
    ("gpt-image-1.5", "low", "1024x1024"): Decimal("0.009"),
    ("gpt-image-1.5", "low", "1024x1536"): Decimal("0.013"),
    ("gpt-image-1.5", "low", "1536x1024"): Decimal("0.013"),
}


def _usd_to_microusd(value: str | Decimal) -> int:
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(
            "OpenAI budget configuration must be a decimal USD amount."
        ) from exc
    if decimal_value < 0:
        raise ValueError("OpenAI budget configuration cannot be negative.")
    return int((decimal_value * MICRO_USD).to_integral_value(rounding=ROUND_HALF_UP))


OPENAI_DAILY_BUDGET_MICRO_USD = _usd_to_microusd(
    env("OPENAI_DAILY_BUDGET_USD", default="5.00")
)
OPENAI_TEXT_REQUEST_RESERVE_MICRO_USD = _usd_to_microusd(
    env("OPENAI_TEXT_REQUEST_RESERVE_USD", default="0.02")
)
OPENAI_TRANSCRIPTION_REQUEST_RESERVE_MICRO_USD = _usd_to_microusd(
    env("OPENAI_TRANSCRIPTION_REQUEST_RESERVE_USD", default="0.25")
)

_RESERVE_BUDGET = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
local maximum = tonumber(ARGV[2])
if current + amount > maximum then
    return 0
end
redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
"""
_ADJUST_BUDGET = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local adjusted = current + tonumber(ARGV[1])
if adjusted <= 0 then
    redis.call('DEL', KEYS[1])
else
    redis.call('SET', KEYS[1], adjusted, 'KEEPTTL')
end
return adjusted
"""


class OpenAIBudgetExceeded(RuntimeError):
    """Raised before a request when the configured daily spend cap is exhausted."""


def _seconds_to_next_utc_day() -> int:
    now = timezone.now()
    next_day = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((next_day - now).total_seconds()))


def _budget_key() -> str:
    return f"vocabume:openai:budget:{timezone.now().date().isoformat()}"


def _estimate_microusd(
    operation: str, *, model: str, size: str = "", quality: str = ""
) -> int:
    if operation == "generate-card-image":
        price = IMAGE_PRICES_USD.get((model, quality, size))
        if price is None:
            raise OpenAIBudgetExceeded(
                "OpenAI image price is not configured for this request."
            )
        return _usd_to_microusd(price)
    if operation == "transcribe-speech":
        return OPENAI_TRANSCRIPTION_REQUEST_RESERVE_MICRO_USD
    return OPENAI_TEXT_REQUEST_RESERVE_MICRO_USD


def _extract_usage(response: object) -> tuple[int, int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None))
    output_tokens = getattr(
        usage, "completion_tokens", getattr(usage, "output_tokens", None)
    )
    if input_tokens is None and output_tokens is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
    return int(input_tokens or 0), int(output_tokens or 0), int(cached_tokens or 0)


def _text_cost_microusd(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int
) -> int:
    try:
        input_price, output_price, cached_input_price = TEXT_PRICES_PER_MILLION_USD[
            model
        ]
    except KeyError as exc:
        raise OpenAIBudgetExceeded(
            f"OpenAI price is not configured for model {model}."
        ) from exc
    uncached_input = max(0, input_tokens - cached_tokens)
    cost = (
        Decimal(uncached_input) * input_price
        + Decimal(cached_tokens) * cached_input_price
        + Decimal(output_tokens) * output_price
    ) / Decimal("1000000")
    return _usd_to_microusd(cost)


@dataclass
class OpenAIBudgetReservation:
    client: Redis | None
    key: str
    operation: str
    model: str
    user_id: int | None
    reserved_microusd: int
    finalized: bool = False

    def record_chat_response(self, response: object) -> None:
        usage = _extract_usage(response)
        if usage is None:
            self._finalize(cost_microusd=self.reserved_microusd, usage_available=False)
            return
        input_tokens, output_tokens, cached_tokens = usage
        self._finalize(
            cost_microusd=_text_cost_microusd(
                self.model, input_tokens, output_tokens, cached_tokens
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_tokens,
            usage_available=True,
        )

    def record_image_response(
        self, _response: object, *, size: str, quality: str
    ) -> None:
        self._finalize(
            cost_microusd=_estimate_microusd(
                self.operation, model=self.model, size=size, quality=quality
            ),
            image_count=1,
            usage_available=True,
        )

    def release(self) -> None:
        if self.finalized:
            return
        self._adjust(-self.reserved_microusd)
        self.finalized = True

    def _finalize(
        self,
        *,
        cost_microusd: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        image_count: int = 0,
        usage_available: bool,
    ) -> None:
        if self.finalized:
            return
        self._adjust(cost_microusd - self.reserved_microusd)
        try:
            OpenAIUsageEvent.objects.create(
                user_id=self.user_id,
                operation=self.operation,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                image_count=image_count,
                cost_microusd=cost_microusd,
                usage_available=usage_available,
            )
        except Exception:
            logger.exception(
                "Failed to persist OpenAI usage event operation=%s", self.operation
            )
        self.finalized = True

    def _adjust(self, delta_microusd: int) -> None:
        if self.client is None or not delta_microusd:
            return
        try:
            self.client.eval(_ADJUST_BUDGET, 1, self.key, delta_microusd)
        except RedisError as exc:
            raise OpenAIBudgetExceeded(
                "OpenAI budget ledger is temporarily unavailable."
            ) from exc


@contextmanager
def openai_budget_reservation(
    operation: str,
    *,
    model: str,
    user_id: int | None,
    size: str = "",
    quality: str = "",
) -> Iterator[OpenAIBudgetReservation]:
    """Reserve daily budget before calling OpenAI, then reconcile actual usage."""
    reserved_microusd = _estimate_microusd(
        operation, model=model, size=size, quality=quality
    )
    client = _redis_client()
    key = _budget_key()
    if client is not None and OPENAI_DAILY_BUDGET_MICRO_USD:
        try:
            reserved = bool(
                client.eval(
                    _RESERVE_BUDGET,
                    1,
                    key,
                    reserved_microusd,
                    OPENAI_DAILY_BUDGET_MICRO_USD,
                    _seconds_to_next_utc_day(),
                )
            )
        except RedisError as exc:
            raise OpenAIBudgetExceeded(
                "OpenAI budget ledger is temporarily unavailable."
            ) from exc
        if not reserved:
            raise OpenAIBudgetExceeded(
                "OpenAI daily budget is exhausted. Please try again tomorrow."
            )
    reservation = OpenAIBudgetReservation(
        client=client,
        key=key,
        operation=operation,
        model=model,
        user_id=user_id,
        reserved_microusd=reserved_microusd,
    )
    try:
        yield reservation
    except Exception:
        reservation.release()
        raise
    else:
        if not reservation.finalized:
            reservation._finalize(
                cost_microusd=reserved_microusd, usage_available=False
            )
