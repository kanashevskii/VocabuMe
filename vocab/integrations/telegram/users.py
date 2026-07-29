"""Shared Telegram-identity adapters for bot handlers."""

from asgiref.sync import sync_to_async

from vocab.services import upsert_telegram_user


@sync_to_async
def get_or_create_user(chat_id: int, username: str | None):
    """Resolve the shared Telegram identity without blocking the event loop."""
    return upsert_telegram_user(chat_id=chat_id, username=username), False
