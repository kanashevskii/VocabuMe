import requests

from vocab.utils import translate_to_ru


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> list:
        return [[["Привет ", "Hello", None, None, 1], ["мир", "world", None, None, 1]]]


def test_translate_to_ru_uses_bounded_google_request(monkeypatch):
    captured: dict = {}

    def fake_get(url, *, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("vocab.utils.requests.get", fake_get)

    assert translate_to_ru("Hello world") == "Привет мир"
    assert captured == {
        "url": "https://translate.googleapis.com/translate_a/single",
        "params": {
            "client": "gtx",
            "sl": "auto",
            "tl": "ru",
            "dt": "t",
            "q": "Hello world",
        },
        "timeout": 5,
    }


def test_translate_to_ru_fails_closed_when_provider_is_unavailable(monkeypatch):
    def failing_get(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("vocab.utils.requests.get", failing_get)

    assert translate_to_ru("Hello") == ""
    assert translate_to_ru("   ") == ""
