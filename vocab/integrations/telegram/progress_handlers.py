"""Telegram handler for the user's learning progress summary."""

from __future__ import annotations

from asgiref.sync import sync_to_async
from telegram import Update
from telegram.ext import ContextTypes

from vocab.integrations.telegram.messaging import safe_reply
from vocab.integrations.telegram.users import get_or_create_user
from vocab.services import (
    build_user_progress as build_user_progress_service,
    get_user_achievements as get_user_achievements_service,
)


@sync_to_async
def get_user_progress(user):
    return build_user_progress_service(user)


@sync_to_async
def get_user_achievements(user):
    return get_user_achievements_service(user)


async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    stats = await get_user_progress(user)

    if stats["total"] == 0:
        await safe_reply(update, "📜 У тебя пока нет слов. Добавь их через /add")
        return

    started = (
        stats["start_date"].strftime("%d.%m.%Y")
        if stats["start_date"]
        else "неизвестно"
    )
    message = (
        f"📊 Твоя статистика:\n\n"
        f"🔹 Всего слов: *{stats['total']}*\n"
        f"✅ Выучено: *{stats['learned']}*\n"
        f"🧠 В процессе: *{stats['learning']}*\n"
        f"📅 Начало обучения: *{started}*\n"
        f"🔤 Неправильные глаголы: *{stats['irregular']}*"
    )
    if stats["rank_percent"] is not None:
        message += f"\n🏅 Ты входишь в *{stats['rank_percent']}%* лучших учеников!"

    achievements = await get_user_achievements(user)
    if achievements:
        message += "\n\n🎖 *Твои достижения:*\n" + "\n".join(
            f"• {achievement}" for achievement in achievements
        )

    await safe_reply(update, message, parse_mode="Markdown")
