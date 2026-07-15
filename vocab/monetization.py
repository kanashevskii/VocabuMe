from __future__ import annotations

import copy
from typing import Any

from core.env import env, env_csv

PRICE_CURRENCY = "USD"
TELEGRAM_STARS_CURRENCY = "XTR"
TELEGRAM_STARS_PRICES = {
    "monthly": 199,
    "yearly": 999,
}
DEFAULT_EXTRA_IMAGE_REGENERATIONS_PER_DAY = 3
DEFAULT_FREE_RELOCATION_PACKS = 2
DEFAULT_FREE_NEW_ITEMS_PER_DAY = 10

PLAN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "free": {
        "code": "free",
        "label": "Free",
        "price": None,
        "entitlements": {
            "max_new_items_per_day": DEFAULT_FREE_NEW_ITEMS_PER_DAY,
            "max_extra_image_regenerations_per_day": DEFAULT_EXTRA_IMAGE_REGENERATIONS_PER_DAY,
            "max_relocation_packs": DEFAULT_FREE_RELOCATION_PACKS,
            "review_unlimited": True,
            "practice_unlimited": True,
            "reminders_unlimited": True,
            "alphabet_unlimited": True,
            "irregular_verbs_unlimited": True,
            "first_auto_image_generation_included": True,
            "premium_relocation_packs": False,
            "ai_explanations": False,
            "ai_dialogue": False,
        },
    },
    "premium": {
        "code": "premium",
        "label": "Premium",
        "price": {
            "monthly": {"amount": "6.99", "currency": PRICE_CURRENCY},
            "yearly": {"amount": "39.99", "currency": PRICE_CURRENCY},
        },
        "telegram_stars_price": {
            "monthly": {
                "amount": TELEGRAM_STARS_PRICES["monthly"],
                "currency": TELEGRAM_STARS_CURRENCY,
            },
            "yearly": {
                "amount": TELEGRAM_STARS_PRICES["yearly"],
                "currency": TELEGRAM_STARS_CURRENCY,
            },
        },
        "entitlements": {
            "max_new_items_per_day": None,
            "max_extra_image_regenerations_per_day": None,
            "max_relocation_packs": None,
            "review_unlimited": True,
            "practice_unlimited": True,
            "reminders_unlimited": True,
            "alphabet_unlimited": True,
            "irregular_verbs_unlimited": True,
            "first_auto_image_generation_included": True,
            "premium_relocation_packs": True,
            "ai_explanations": True,
            "ai_dialogue": True,
        },
    },
}

PAYWALL_TRIGGERS: list[dict[str, str]] = [
    {
        "code": "daily_new_items_limit",
        "kind": "hard",
        "surface": "bot,miniapp,website",
        "description": "Show paywall when the free user tries to add the 11th new word or phrase in the same day.",
    },
    {
        "code": "premium_pack_gate",
        "kind": "hard",
        "surface": "miniapp,website",
        "description": "Show paywall when the free user tries to open a relocation scenario pack beyond the free starter allowance.",
    },
    {
        "code": "extra_image_regeneration_limit",
        "kind": "hard",
        "surface": "miniapp,website",
        "description": "Show paywall when the free user exceeds the daily cap for manual image regenerations.",
    },
    {
        "code": "post_first_value_moment",
        "kind": "soft",
        "surface": "miniapp,website",
        "description": "Offer Premium after the user finishes the first meaningful scenario or first successful practice loop.",
    },
]


def get_plan_catalog() -> dict[str, dict[str, Any]]:
    return PLAN_DEFINITIONS


def get_telegram_stars_prices_for_user(user_chat_id: int | str | None = None) -> dict[str, int]:
    prices = dict(TELEGRAM_STARS_PRICES)
    if user_chat_id is None:
        return prices

    test_user_ids = set(env_csv("TELEGRAM_STARS_TEST_USER_IDS"))
    if str(user_chat_id) not in test_user_ids:
        return prices

    test_price = env("TELEGRAM_STARS_TEST_PRICE", default=1, cast=int)
    if test_price < 1:
        test_price = 1
    return {period: test_price for period in prices}


def get_monetization_payload(user_chat_id: int | str | None = None) -> dict[str, Any]:
    plans = copy.deepcopy(get_plan_catalog())
    stars_prices = get_telegram_stars_prices_for_user(user_chat_id)
    premium = plans.get("premium")
    if premium:
        premium["telegram_stars_price"] = {
            period: {"amount": amount, "currency": TELEGRAM_STARS_CURRENCY}
            for period, amount in stars_prices.items()
        }
    return {
        "currency": PRICE_CURRENCY,
        "telegram_stars_currency": TELEGRAM_STARS_CURRENCY,
        "plans": plans,
        "paywall_triggers": PAYWALL_TRIGGERS,
        "default_free_plan_code": "free",
        "default_paid_plan_code": "premium",
    }
