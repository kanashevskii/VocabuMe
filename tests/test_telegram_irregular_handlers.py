import asyncio
from types import SimpleNamespace

from vocab import bot
from vocab.integrations.telegram import irregular_handlers
from vocab.irregular_verbs import IRREGULAR_VERBS


def test_bot_keeps_irregular_handler_compatibility_imports():
    assert bot.irregular_menu is irregular_handlers.irregular_menu
    assert bot.irregular_repeat is irregular_handlers.irregular_repeat
    assert bot.handle_irregular_list is irregular_handlers.handle_irregular_list
    assert bot.irregular_train is irregular_handlers.irregular_train
    assert bot.handle_irregular_answer is irregular_handlers.handle_irregular_answer
    assert bot.get_praise is irregular_handlers.get_praise
    assert bot.user_lessons is irregular_handlers.user_lessons


def test_irregular_answer_updates_shared_session_and_persists_only_correct_answer(
    monkeypatch,
):
    calls = {"progress": [], "achievements": [], "continued": 0, "edited": []}
    word = IRREGULAR_VERBS[0]
    correct_pair = f"{word['past']} {word['participle']}"

    async def safe_answer(_query):
        return None

    async def get_or_create_user(_chat_id, _username):
        return SimpleNamespace(id=1), False

    async def update_progress(user, base):
        calls["progress"].append((user.id, base))

    async def get_achievements(_user):
        return ["Первый глагол"]

    async def safe_reply(_update, message, **_kwargs):
        calls["achievements"].append(message)

    async def continue_training(_update, _context):
        calls["continued"] += 1

    async def edit_message_text(message):
        calls["edited"].append(message)

    monkeypatch.setattr(irregular_handlers, "safe_answer", safe_answer)
    monkeypatch.setattr(irregular_handlers, "get_or_create_user", get_or_create_user)
    monkeypatch.setattr(
        irregular_handlers, "_update_irregular_progress", update_progress
    )
    monkeypatch.setattr(irregular_handlers, "_get_new_achievements", get_achievements)
    monkeypatch.setattr(irregular_handlers, "safe_reply", safe_reply)
    monkeypatch.setattr(irregular_handlers, "irregular_train", continue_training)

    update = SimpleNamespace(
        callback_query=SimpleNamespace(
            data=f"irrans|{word['base']}|{correct_pair}",
            message=SimpleNamespace(chat=SimpleNamespace(id=123)),
            edit_message_text=edit_message_text,
        ),
        effective_chat=SimpleNamespace(id=123, username="alice"),
    )
    context = SimpleNamespace(user_data={"irr_info_123": {"answered": 0, "correct": 0}})

    asyncio.run(irregular_handlers.handle_irregular_answer(update, context))

    assert context.user_data["irr_info_123"] == {"answered": 1, "correct": 1}
    assert calls["progress"] == [(1, word["base"])]
    assert calls["achievements"] == ["🏆 Первый глагол"]
    assert calls["continued"] == 1
    assert calls["edited"] == [f"✅ Верно! {word['base']} → {correct_pair}"]
