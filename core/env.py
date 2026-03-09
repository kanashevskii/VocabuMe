from __future__ import annotations

from functools import lru_cache
from typing import Any

from decouple import Csv, UndefinedValueError, config as decouple_config
from django.core.exceptions import ImproperlyConfigured


def env(
    name: str, *, default: Any = None, cast: Any = str, required: bool = False
) -> Any:
    try:
        value = decouple_config(name, default=default, cast=cast)
    except UndefinedValueError as exc:
        raise ImproperlyConfigured(
            f"Missing required environment variable: {name}"
        ) from exc

    if cast is str and isinstance(value, str):
        value = value.strip()

    if required and (value is None or value == ""):
        raise ImproperlyConfigured(f"Environment variable {name} must not be empty.")

    return value


def env_csv(name: str, *, default: str = "") -> list[str]:
    values = env(name, default=default, cast=Csv())
    return [item for item in values if item]


@lru_cache(maxsize=1)
def get_secret_key() -> str:
    return env("SECRET_KEY", required=True)


@lru_cache(maxsize=1)
def get_telegram_token() -> str:
    return env("TELEGRAM_TOKEN", required=True)


@lru_cache(maxsize=1)
def get_telegram_bot_username() -> str:
    return env("TELEGRAM_BOT_USERNAME", default="")


@lru_cache(maxsize=1)
def get_webapp_url() -> str:
    return env("WEBAPP_URL", default="")


@lru_cache(maxsize=1)
def get_openai_api_key() -> str:
    return env("OPENAI_API_KEY", required=True)


@lru_cache(maxsize=1)
def get_telegram_payments_provider_token() -> str:
    return env("TELEGRAM_PAYMENTS_PROVIDER_TOKEN", default="")
