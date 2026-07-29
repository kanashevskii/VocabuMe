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
