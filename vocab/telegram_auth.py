from __future__ import annotations

import hashlib
import hmac
import json
from time import time
from urllib.parse import parse_qsl


class TelegramAuthError(ValueError):
    pass


def _build_check_string(data: dict) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(data.items()))


def verify_login_widget(payload: dict, bot_token: str, max_age_seconds: int = 86400) -> dict:
    data = {key: value for key, value in payload.items() if value not in (None, "")}
    provided_hash = data.pop("hash", None)
    if not provided_hash:
        raise TelegramAuthError("Missing Telegram hash.")

    auth_date = int(data.get("auth_date", 0))
    if not auth_date or time() - auth_date > max_age_seconds:
        raise TelegramAuthError("Telegram auth data is too old.")

    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    expected_hash = hmac.new(
        secret_key,
        _build_check_string(data).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise TelegramAuthError("Invalid Telegram auth signature.")
    return data


def verify_webapp_init_data(init_data: str, bot_token: str, max_age_seconds: int = 86400) -> dict:
    parsed = dict(parse_qsl(init_data, strict_parsing=True))
    provided_hash = parsed.pop("hash", None)
    if not provided_hash:
        raise TelegramAuthError("Missing WebApp hash.")

    auth_date = int(parsed.get("auth_date", 0))
    if not auth_date or time() - auth_date > max_age_seconds:
        raise TelegramAuthError("Telegram WebApp auth data is too old.")

    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key,
        _build_check_string(parsed).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise TelegramAuthError("Invalid Telegram WebApp signature.")

    if "user" in parsed:
        parsed["user"] = json.loads(parsed["user"])
    return parsed
