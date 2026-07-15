import asyncio
from types import SimpleNamespace

from vocab.integrations.telegram.messaging import safe_edit_message_text, safe_reply


def test_safe_reply_uses_message_target_and_preserves_keyword_arguments():
    received: dict[str, object] = {}

    class Message:
        async def reply_text(self, text, **kwargs):
            received["text"] = text
            received["kwargs"] = kwargs
            return "sent"

    update = SimpleNamespace(
        message=Message(), callback_query=None, effective_chat=SimpleNamespace(id=42)
    )

    result = asyncio.run(safe_reply(update, "Hello", parse_mode="HTML"))

    assert result == "sent"
    assert received == {"text": "Hello", "kwargs": {"parse_mode": "HTML"}}


def test_safe_reply_without_message_target_is_a_noop():
    update = SimpleNamespace(message=None, callback_query=None, effective_chat=None)

    assert asyncio.run(safe_reply(update, "Hello")) is None


def test_safe_edit_retries_without_parse_mode_after_telegram_parse_error():
    calls: list[dict[str, object]] = []

    class Query:
        message = SimpleNamespace(chat_id=42)

        async def edit_message_text(self, text, **kwargs):
            calls.append(kwargs)
            if "parse_mode" in kwargs:
                from telegram.error import BadRequest

                raise BadRequest("Can't parse entities")
            return text

    result = asyncio.run(safe_edit_message_text(Query(), "Hello", parse_mode="HTML"))

    assert result == "Hello"
    assert calls == [{"parse_mode": "HTML"}, {}]
