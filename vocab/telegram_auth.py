from __future__ import annotations

import hashlib
import hmac
import json
from time import time
from typing import NotRequired, TypedDict
from urllib.parse import parse_qsl


class TelegramAuthError(ValueError):
    pass


class TelegramWebAppUser(TypedDict):
    """Validated subset of Telegram's Web App user object used by VocabuMe."""

    id: int
    username: NotRequired[str]
    first_name: NotRequired[str]
    last_name: NotRequired[str]
    language_code: NotRequired[str]
    is_bot: NotRequired[bool]
    is_premium: NotRequired[bool]


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


def _parse_webapp_user(raw_user: object) -> TelegramWebAppUser:
    if not isinstance(raw_user, dict):
        raise TelegramAuthError("Invalid Telegram user payload.")
    user_id = raw_user.get("id")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise TelegramAuthError("Invalid Telegram user payload.")

    user: TelegramWebAppUser = {"id": user_id}
    for key in ("username", "first_name", "last_name", "language_code"):
        value = raw_user.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TelegramAuthError("Invalid Telegram user payload.")
        if key == "username":
            user["username"] = value
        elif key == "first_name":
            user["first_name"] = value
        elif key == "last_name":
            user["last_name"] = value
        else:
            user["language_code"] = value
    for key in ("is_bot", "is_premium"):
        value = raw_user.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise TelegramAuthError("Invalid Telegram user payload.")
        if key == "is_bot":
            user["is_bot"] = value
        else:
            user["is_premium"] = value
    return user


def verify_login_widget(
    payload: dict, bot_token: str, max_age_seconds: int = 300
) -> dict:
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


def verify_webapp_init_data(
    init_data: str, bot_token: str, max_age_seconds: int = 300
) -> dict[str, object]:
    if (
        not isinstance(init_data, str)
        or not init_data
        or len(init_data) > MAX_INIT_DATA_LENGTH
    ):
        raise TelegramAuthError("Invalid WebApp init data.")
    pairs = parse_qsl(init_data, strict_parsing=True, keep_blank_values=True)
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise TelegramAuthError("Duplicate WebApp init-data field.")
    parsed: dict[str, object] = dict(pairs)
    provided_hash = parsed.pop("hash", None)
    if not isinstance(provided_hash, str) or not provided_hash:
        raise TelegramAuthError("Missing WebApp hash.")

    _validate_auth_date(parsed.get("auth_date"), max_age_seconds)

    secret_key = hmac.new(
        b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_hash = hmac.new(
        secret_key,
        _build_check_string(parsed).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise TelegramAuthError("Invalid Telegram WebApp signature.")

    if "user" in parsed:
        try:
            raw_user = json.loads(str(parsed["user"]))
        except json.JSONDecodeError as exc:
            raise TelegramAuthError("Invalid Telegram user payload.") from exc
        parsed["user"] = _parse_webapp_user(raw_user)
    return parsed
