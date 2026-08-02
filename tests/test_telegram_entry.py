import asyncio
from types import SimpleNamespace

from vocab import bot
from vocab.integrations.telegram import entry


def test_bot_keeps_entry_handler_compatibility_imports():
    assert bot.start is entry.start
    assert bot._webapp_url_with_source is entry._webapp_url_with_source


def test_webapp_deep_link_source_is_bounded_and_preserves_existing_query(monkeypatch):
    monkeypatch.setattr(entry, "WEBAPP_URL", "https://example.test/open?lang=ru")

    url = entry._webapp_url_with_source("source" * 30)

    assert url.startswith("https://example.test/open?")
    assert "lang=ru" in url
    assert f"src={'source' * 13}so" in url


def test_bot_callback_word_lookup_is_scoped_to_the_current_user(monkeypatch):
    owner = SimpleNamespace(id=123, active_studied_language="en")
    captured: dict[str, object] = {}

    def get_user_word(user, word_id):
        captured["user"] = user
        captured["word_id"] = word_id
        return None

    monkeypatch.setattr(bot, "get_user_word_service", get_user_word)

    assert asyncio.run(bot.get_word_by_id(owner, 999)) is None
    assert captured == {"user": owner, "word_id": 999}
