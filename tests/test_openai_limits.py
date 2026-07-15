import pytest
from contextlib import nullcontext

from vocab.openai_limits import (
    OpenAIConcurrencyExceeded,
    openai_slot,
    openai_user_scope,
)
from vocab.openai_utils import openai_request_slot


class FakeRedis:
    def __init__(self):
        self.values: dict[str, int] = {}

    def eval(self, script, _num_keys, key, *args):
        if "current >= maximum" in script:
            maximum = int(args[0])
            current = self.values.get(key, 0)
            if current >= maximum:
                return 0
            self.values[key] = current + 1
            return 1
        current = self.values.get(key, 0)
        if current <= 1:
            self.values.pop(key, None)
        else:
            self.values[key] = current - 1
        return 1

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    def expire(self, _key, _seconds):
        return True

    def delete(self, key):
        self.values.pop(key, None)
        return 1


def test_openai_slot_releases_global_and_user_capacity(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr("vocab.openai_limits._redis_client", lambda: client)

    with openai_slot("image", user_id=12):
        assert client.values == {
            "vocabume:openai:inflight": 1,
            "vocabume:openai:user:12:inflight": 1,
        }

    assert client.values == {}


def test_openai_slot_rejects_parallel_operation_for_same_user(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr("vocab.openai_limits._redis_client", lambda: client)
    monkeypatch.setattr("vocab.openai_limits.OPENAI_SLOT_WAIT_SECONDS", 0)

    with openai_slot("word", user_id=12):
        with pytest.raises(OpenAIConcurrencyExceeded):
            with openai_slot("image", user_id=12):
                pass

    assert client.values == {}


def test_openai_request_slot_uses_trusted_user_scope(monkeypatch):
    captured: dict[str, int | str | None] = {}

    def fake_slot(label, *, user_id=None):
        captured.update(label=label, user_id=user_id)
        return nullcontext()

    monkeypatch.setattr("vocab.openai_utils.openai_slot", fake_slot)

    with openai_user_scope(42):
        with openai_request_slot("word-data"):
            pass

    assert captured == {"label": "word-data", "user_id": 42}


def test_openai_slot_opens_circuit_after_configured_failures(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr("vocab.openai_limits._redis_client", lambda: client)
    monkeypatch.setattr("vocab.openai_limits.OPENAI_CIRCUIT_FAILURE_THRESHOLD", 1)

    with pytest.raises(RuntimeError):
        with openai_slot("word"):
            raise RuntimeError("provider failure")

    with pytest.raises(OpenAIConcurrencyExceeded):
        with openai_slot("word"):
            pass
