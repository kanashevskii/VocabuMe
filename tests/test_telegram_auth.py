from __future__ import annotations

import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

import pytest

from vocab.telegram_auth import (
    TelegramAuthError,
    verify_login_widget,
    verify_webapp_init_data,
)


def _widget_hash(payload: dict, bot_token: str) -> str:
    data = {
        key: value
        for key, value in payload.items()
        if key != "hash" and value not in (None, "")
    }
    check_string = "\n".join(f"{key}={value}" for key, value in sorted(data.items()))
    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    return hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _webapp_hash(payload: dict, bot_token: str) -> str:
    data = {key: value for key, value in payload.items() if key != "hash"}
    check_string = "\n".join(f"{key}={value}" for key, value in sorted(data.items()))
    secret_key = hmac.new(
        b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256
    ).digest()
    return hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def test_verify_login_widget_accepts_valid_payload():
    bot_token = "test-token"
    payload = {
        "id": "123",
        "username": "tester",
        "auth_date": str(int(time())),
    }
    payload["hash"] = _widget_hash(payload, bot_token)

    verified = verify_login_widget(payload, bot_token)

    assert verified["id"] == "123"
    assert verified["username"] == "tester"


def test_verify_login_widget_rejects_invalid_signature():
    with pytest.raises(TelegramAuthError, match="Invalid Telegram auth signature"):
        verify_login_widget(
            {
                "id": "123",
                "username": "tester",
                "auth_date": str(int(time())),
                "hash": "bad",
            },
            "test-token",
        )


def test_verify_webapp_init_data_accepts_valid_payload():
    bot_token = "test-token"
    payload = {
        "auth_date": str(int(time())),
        "query_id": "abc",
        "user": json.dumps({"id": 123, "username": "tester"}),
    }
    payload["hash"] = _webapp_hash(payload, bot_token)
    init_data = urlencode(payload)

    verified = verify_webapp_init_data(init_data, bot_token)

    assert verified["query_id"] == "abc"
    assert verified["user"]["id"] == 123


def test_verify_webapp_init_data_rejects_missing_hash():
    with pytest.raises(TelegramAuthError, match="Missing WebApp hash"):
        verify_webapp_init_data("auth_date=123", "test-token")
