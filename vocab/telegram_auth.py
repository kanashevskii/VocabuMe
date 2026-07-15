from __future__ import annotations

import hashlib
import hmac
import json
from time import time
from urllib.parse import parse_qsl


class TelegramAuthError(ValueError):
    pass


MAX_INIT_DATA_LENGTH = 8_192
MAX_CLOCK_SKEW_SECONDS = 30


def _build_check_string(data: dict) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(data.items()))


def _validate_auth_date(auth_date_raw: object, max_age_seconds: int) -> None:
    if not isinstance(auth_date_raw, (str, int, float)) or isinstance(
        auth_date_raw, bool
    ):
        raise TelegramAuthError("Invalid Telegram auth date.")
    try:
        auth_date = int(auth_date_raw)
    except (TypeError, ValueError) as exc:
        raise TelegramAuthError("Invalid Telegram auth date.") from exc
    now = time()
    if not auth_date or auth_date > now + MAX_CLOCK_SKEW_SECONDS:
        raise TelegramAuthError("Invalid Telegram auth date.")
    if now - auth_date > max_age_seconds:
        raise TelegramAuthError("Telegram auth data is too old.")


def verify_login_widget(payload: dict, bot_token: str, max_age_seconds: int = 300) -> dict:
    data = {key: value for key, value in payload.items() if value not in (None, "")}
    provided_hash = data.pop("hash", None)
    if not provided_hash:
        raise TelegramAuthError("Missing Telegram hash.")

    _validate_auth_date(data.get("auth_date"), max_age_seconds)

    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    expected_hash = hmac.new(
        secret_key,
        _build_check_string(data).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise TelegramAuthError("Invalid Telegram auth signature.")
    return data


def verify_webapp_init_data(init_data: str, bot_token: str, max_age_seconds: int = 300) -> dict:
    if not isinstance(init_data, str) or not init_data or len(init_data) > MAX_INIT_DATA_LENGTH:
        raise TelegramAuthError("Invalid WebApp init data.")
    pairs = parse_qsl(init_data, strict_parsing=True, keep_blank_values=True)
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise TelegramAuthError("Duplicate WebApp init-data field.")
    parsed = dict(pairs)
    provided_hash = parsed.pop("hash", None)
    if not provided_hash:
        raise TelegramAuthError("Missing WebApp hash.")

    _validate_auth_date(parsed.get("auth_date"), max_age_seconds)

    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key,
        _build_check_string(parsed).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise TelegramAuthError("Invalid Telegram WebApp signature.")

    if "user" in parsed:
        try:
            parsed["user"] = json.loads(parsed["user"])
        except json.JSONDecodeError as exc:
            raise TelegramAuthError("Invalid Telegram user payload.") from exc
        if not isinstance(parsed["user"], dict):
            raise TelegramAuthError("Invalid Telegram user payload.")
    return parsed
