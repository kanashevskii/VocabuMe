import asyncio
from types import SimpleNamespace

from vocab import bot
from vocab.integrations.telegram import word_handlers


def test_bot_keeps_word_handler_compatibility_imports():
    assert bot.mywords is word_handlers.mywords
    assert bot.handle_mywords_pagination is word_handlers.handle_mywords_pagination
    assert bot.handle_mywords_delete is word_handlers.handle_mywords_delete
    assert bot.handle_mywords_edit is word_handlers.handle_mywords_edit
    assert bot.update_word_translation is word_handlers.update_word_translation


def test_edit_callback_rejects_word_outside_current_user(monkeypatch):
    edited_messages = []

    async def safe_answer(_query):
        return None

    async def get_or_create_user(_chat_id, _username):
        return SimpleNamespace(id=1), False

    async def get_word_for_user(_user, _word_id):
        return None

    async def edit_message_text(message, **_kwargs):
        edited_messages.append(message)

    monkeypatch.setattr(word_handlers, "safe_answer", safe_answer)
    monkeypatch.setattr(word_handlers, "get_or_create_user", get_or_create_user)
    monkeypatch.setattr(word_handlers, "get_word_for_user", get_word_for_user)

    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="mywords_edit_choose|999|0", edit_message_text=edit_message_text
        ),
        effective_chat=SimpleNamespace(id=123, username="alice"),
    )
    context = SimpleNamespace(user_data={})

    asyncio.run(word_handlers.handle_mywords_edit(update, context))

    assert edited_messages == ["⚠️ Слово не найдено."]
    assert context.user_data == {}


def test_delete_callback_rejects_word_outside_current_user(monkeypatch):
    edited_messages = []

    async def safe_answer(_query):
        return None

    async def get_or_create_user(_chat_id, _username):
        return SimpleNamespace(id=1), False

    async def get_word_for_user(_user, _word_id):
        return None

    async def edit_message_text(message, **_kwargs):
        edited_messages.append(message)

    monkeypatch.setattr(word_handlers, "safe_answer", safe_answer)
    monkeypatch.setattr(word_handlers, "get_or_create_user", get_or_create_user)
    monkeypatch.setattr(word_handlers, "get_word_for_user", get_word_for_user)

    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data="mywords_delete_one_confirm|999|0",
            edit_message_text=edit_message_text,
        ),
        effective_chat=SimpleNamespace(id=123, username="alice"),
    )
    context = SimpleNamespace(user_data={})

    asyncio.run(word_handlers.handle_mywords_delete(update, context))

    assert edited_messages == ["⚠️ Слово не найдено."]
    assert context.user_data == {}
