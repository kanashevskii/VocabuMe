import asyncio
from datetime import date
from types import SimpleNamespace

from vocab import bot
from vocab.integrations.telegram import progress_handlers


def test_bot_keeps_progress_handler_compatibility_imports():
    assert bot.progress is progress_handlers.progress
    assert bot.get_user_progress is progress_handlers.get_user_progress
    assert bot.get_user_achievements is progress_handlers.get_user_achievements


def test_progress_renders_achievements_and_rank(monkeypatch):
    replies = []

    async def get_or_create_user(_chat_id, _username):
        return SimpleNamespace(id=1), False

    async def get_user_progress(_user):
        return {
            "total": 12,
            "learned": 8,
            "learning": 4,
            "start_date": date(2026, 7, 1),
            "irregular": 3,
            "rank_percent": 10,
        }

    async def get_user_achievements(_user):
        return ["Первые 10 слов"]

    async def safe_reply(_update, message, **kwargs):
        replies.append((message, kwargs))

    monkeypatch.setattr(progress_handlers, "get_or_create_user", get_or_create_user)
    monkeypatch.setattr(progress_handlers, "get_user_progress", get_user_progress)
    monkeypatch.setattr(
        progress_handlers, "get_user_achievements", get_user_achievements
    )
    monkeypatch.setattr(progress_handlers, "safe_reply", safe_reply)

    update = SimpleNamespace(effective_chat=SimpleNamespace(id=123, username="alice"))
    asyncio.run(progress_handlers.progress(update, SimpleNamespace()))

    message, kwargs = replies[0]
    assert "*12*" in message
    assert "*10%*" in message
    assert "Первые 10 слов" in message
    assert kwargs["parse_mode"] == "Markdown"
