import pytest

from vocab.openai_limits import OpenAIConcurrencyExceeded, openai_slot


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
