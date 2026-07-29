"""Telegram entrypoint handlers for Mini App opening and web-login linking."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from asgiref.sync import sync_to_async
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from core.env import get_webapp_url
from vocab.integrations.telegram.messaging import safe_reply
from vocab.integrations.telegram.users import get_or_create_user
from vocab.services import bind_web_login_token

WEBAPP_URL = get_webapp_url()


def _webapp_url_with_source(source: str) -> str:
    if not WEBAPP_URL or not source:
        return WEBAPP_URL
    parts = urlsplit(WEBAPP_URL)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["src"] = source[:80]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle `/start`, login tokens, acquisition tags, and Mini App launch."""
    if update.callback_query:
        try:
            await update.callback_query.answer()
        except BadRequest as exc:
            logging.warning("Callback answer failed (possibly stale): %s", exc)

    start_source = ""
    if context.args:
        start_arg = context.args[0]
        if start_arg.startswith("login_"):
            token = start_arg.removeprefix("login_")
            user, _ = await get_or_create_user(
                update.effective_chat.id, update.effective_chat.username
            )
            login_token = await sync_to_async(bind_web_login_token)(token, user)
            if login_token:
                await safe_reply(
                    update,
                    "Вход для сайта подтверждён. Возвращайся в браузер, страница авторизуется автоматически.",
                )
            else:
                await safe_reply(
                    update,
                    "Ссылка для входа недействительна или уже использована. Запроси новую на сайте.",
                )
            return
        start_source = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", start_arg).strip("-")[:80]
        if start_source and update.effective_chat:
            user, _ = await get_or_create_user(
                update.effective_chat.id, update.effective_chat.username
            )
            logging.info(
                "Acquisition start source=%s user_id=%s chat_id=%s username=%s",
                start_source,
                user.id,
                update.effective_chat.id,
                update.effective_chat.username or "",
            )

    keyboard = []
    if WEBAPP_URL:
        keyboard.append(
            [
                InlineKeyboardButton(
                    "🚀 Открыть VocabuMe",
                    web_app=WebAppInfo(url=_webapp_url_with_source(start_source)),
                )
            ]
        )

    await safe_reply(
        update,
        "👋 VocabuMe теперь работает как Telegram Mini App.\n\n"
        "Открывай приложение, чтобы добавлять слова, проходить практику, смотреть прогресс и управлять словарём.\n"
        "Этот бот остаётся для входа и напоминаний о занятиях.",
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
    )
