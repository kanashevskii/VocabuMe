"""Resilient Telegram delivery primitives shared by bot handlers.

The handlers own product conversation flow; this module owns the transport
boundary, including Telegram-specific retry and error semantics.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

from telegram.error import (
    BadRequest,
    Forbidden,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

logger = logging.getLogger(__name__)


async def safe_telegram_request(
    action: str,
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    chat_id: int | None = None,
    attempts: int = 3,
    swallow_bad_request: bool = True,
) -> Any | None:
    """Execute a Telegram request with bounded retry for transient failures."""
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except RetryAfter as exc:
            delay = float(getattr(exc, "retry_after", 1.0)) + 0.2
            logger.warning("%s: RetryAfter %.2fs chat_id=%s", action, delay, chat_id)
            await asyncio.sleep(delay)
        except BadRequest as exc:
            if swallow_bad_request:
                logger.warning(
                    "%s: bad request chat_id=%s err=%s", action, chat_id, exc
                )
                return None
            raise
        except (TimedOut, NetworkError) as exc:
            if attempt >= attempts - 1:
                logger.exception(
                    "%s: network timeout exhausted chat_id=%s err=%s",
                    action,
                    chat_id,
                    exc,
                )
                return None
            delay = min(2**attempt, 8) + random.random() * 0.25
            logger.warning(
                "%s: network error, retry in %.2fs chat_id=%s err=%s",
                action,
                delay,
                chat_id,
                exc,
            )
            await asyncio.sleep(delay)
        except Forbidden as exc:
            logger.warning("%s: forbidden chat_id=%s err=%s", action, chat_id, exc)
            return None
        except TelegramError as exc:
            logger.exception(
                "%s: TelegramError chat_id=%s err=%s", action, chat_id, exc
            )
            return None
        except Exception:
            logger.exception("%s: unexpected error chat_id=%s", action, chat_id)
            return None
    return None


def _message_target(update: Any) -> Any | None:
    if update.message:
        return update.message
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return None


async def safe_reply(update: Any, text: str, **kwargs: Any) -> Any | None:
    target = _message_target(update)
    if target is None:
        logger.warning("safe_reply: no message target (update=%s)", update)
        return None

    chat_id = update.effective_chat.id if update.effective_chat else None

    async def send(send_kwargs: dict[str, Any]) -> Any:
        return await target.reply_text(text, **send_kwargs)

    try:
        return await safe_telegram_request(
            "reply_text",
            lambda: send(kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        if "parse" in str(exc).lower() and "parse_mode" in kwargs:
            logger.warning(
                "safe_reply: parse error, retrying without parse_mode chat_id=%s err=%s",
                chat_id,
                exc,
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("parse_mode", None)
            return await safe_telegram_request(
                "reply_text(no-parse)",
                lambda: send(fallback_kwargs),
                chat_id=chat_id,
                attempts=2,
            )
        logger.warning("safe_reply: BadRequest chat_id=%s err=%s", chat_id, exc)
        return None


async def safe_answer(query: Any) -> None:
    try:
        await query.answer()
    except BadRequest as exc:
        logger.warning("Callback answer failed (possibly stale): %s", exc)
    except (TimedOut, NetworkError, RetryAfter, Forbidden, TelegramError) as exc:
        logger.warning("Callback answer failed: %s", exc)
    except Exception:
        logger.exception("Callback answer crashed")


async def safe_photo_reply(update: Any, photo: bytes, **kwargs: Any) -> Any | None:
    target = _message_target(update)
    if target is None:
        logger.warning("safe_photo_reply: no message target (update=%s)", update)
        return None

    chat_id = update.effective_chat.id if update.effective_chat else None

    async def send(send_kwargs: dict[str, Any]) -> Any:
        return await target.reply_photo(photo=photo, **send_kwargs)

    try:
        return await safe_telegram_request(
            "reply_photo",
            lambda: send(kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        logger.warning("safe_photo_reply: BadRequest chat_id=%s err=%s", chat_id, exc)
        return None


async def safe_edit_message_text(query: Any, text: str, **kwargs: Any) -> Any | None:
    chat_id = query.message.chat_id if getattr(query, "message", None) else None
    try:
        return await safe_telegram_request(
            "edit_message_text",
            lambda: query.edit_message_text(text, **kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        if "parse" in str(exc).lower() and "parse_mode" in kwargs:
            logger.warning(
                "safe_edit_message_text: parse error, retrying without parse_mode chat_id=%s err=%s",
                chat_id,
                exc,
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("parse_mode", None)
            return await safe_telegram_request(
                "edit_message_text(no-parse)",
                lambda: query.edit_message_text(text, **fallback_kwargs),
                chat_id=chat_id,
                attempts=2,
            )
        logger.warning(
            "safe_edit_message_text: BadRequest chat_id=%s err=%s", chat_id, exc
        )
        return None
