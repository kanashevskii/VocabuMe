"""Telegram handlers for learning preferences and reminder configuration."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from asgiref.sync import sync_to_async
from telegram import InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from vocab.integrations.telegram.messaging import safe_answer, safe_reply
from vocab.integrations.telegram.settings_ui import (
    main_settings_keyboard,
    main_settings_text,
    reminder_menu_text,
    reminder_settings_keyboard,
    repeat_menu_text,
    repeat_settings_keyboard,
    review_menu_text,
    review_settings_keyboard,
)
from vocab.integrations.telegram.users import get_or_create_user
from vocab.services import (
    save_user as save_user_service,
    set_user_repeat_threshold,
    update_user_reminder_time as update_user_reminder_time_service,
    update_user_timezone as update_user_timezone_service,
)
from vocab.utils import normalize_timezone_value, timezone_from_name

SET_REMINDER_TIME = 1
SET_REMINDER_TZ = 2


@sync_to_async
def save_user(user):
    return save_user_service(user)


@sync_to_async
def update_user_repeat_threshold(user, value: int):
    return set_user_repeat_threshold(user, value)


@sync_to_async
def update_user_reminder_time(user, time_obj):
    return update_user_reminder_time_service(user, time_obj)


@sync_to_async
def update_user_timezone(user, tz_value: str):
    return update_user_timezone_service(user, tz_value)


def parse_reminder_time(value: str):
    clean = value.replace(" ", "")
    clean = (
        clean.replace(".", ":").replace("-", ":").replace("—", ":").replace("–", ":")
    )
    if ":" not in clean and len(clean) == 4 and clean.isdigit():
        clean = f"{clean[:2]}:{clean[2:]}"
    parts = clean.split(":")
    if len(parts) != 2:
        raise ValueError("Wrong time format")
    hours, minutes = parts
    if not (hours.isdigit() and minutes.isdigit()):
        raise ValueError("Time must contain digits")
    return datetime.strptime(f"{int(hours):02d}:{int(minutes):02d}", "%H:%M").time()


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    await safe_reply(
        update,
        main_settings_text(user),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(main_settings_keyboard()),
    )


async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    data = query.data
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )

    if data == "settings_repeat":
        await query.edit_message_text(
            repeat_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(repeat_settings_keyboard()),
        )
        return
    if data == "settings_review":
        await query.edit_message_text(
            review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(review_settings_keyboard(user)),
        )
        return
    if data == "settings_reminders":
        await query.edit_message_text(
            reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(reminder_settings_keyboard(user)),
        )
        return
    if data == "back_to_settings":
        await query.edit_message_text(
            main_settings_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(main_settings_keyboard()),
        )
        return
    if data.startswith("set_repeat_"):
        await update_user_repeat_threshold(user, int(data.split("_")[-1]))
        await query.edit_message_text(
            repeat_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(repeat_settings_keyboard()),
        )
        return
    if data == "toggle_review":
        user.enable_review_old_words = not user.enable_review_old_words
        await save_user(user)
        await query.edit_message_text(
            review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(review_settings_keyboard(user)),
        )
        return
    if data.startswith("set_review_days_"):
        user.days_before_review = int(data.split("_")[-1])
        await save_user(user)
        await query.edit_message_text(
            review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(review_settings_keyboard(user)),
        )
        return
    if data == "toggle_reminder":
        user.reminder_enabled = not user.reminder_enabled
        await save_user(user)
        await query.edit_message_text(
            reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(reminder_settings_keyboard(user)),
        )
        return
    if data.startswith("set_reminder_interval_"):
        user.reminder_interval_days = int(data.split("_")[-1])
        await save_user(user)
        await query.edit_message_text(
            reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(reminder_settings_keyboard(user)),
        )
        return
    if data == "set_reminder_time":
        await query.edit_message_text(
            "🕒 Введите время в формате `HH:MM`, например: `08:30` или `21:00`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TIME
    if data == "set_reminder_tz":
        await query.edit_message_text(
            "🌍 Введите часовой пояс. Примеры: `Europe/Moscow`, `UTC+03`, `-5`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TZ


async def set_reminder_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    try:
        parsed_time = parse_reminder_time(text)
        await update_user_reminder_time(user, parsed_time)
        await safe_reply(
            update,
            f"✅ Напоминания будут приходить в *{parsed_time.strftime('%H:%M')}*.",
            parse_mode="Markdown",
        )
        await settings(update, context)
        return ConversationHandler.END
    except Exception as exc:  # noqa: BLE001 - keep retry UX for all handler failures
        logging.exception("Failed to parse reminder time: %s", exc)
        await safe_reply(
            update,
            "⚠️ Неверный формат. Попробуй ещё раз в формате `HH:MM`, например `09:00`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TIME


async def set_reminder_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    try:
        normalized = normalize_timezone_value(text)
        await update_user_timezone(user, normalized)
        tzinfo = timezone_from_name(normalized)
        offset = (
            (datetime.now(tzinfo).utcoffset() or timedelta(0))
            if tzinfo
            else timedelta(0)
        )
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        hours, minutes = divmod(abs(total_minutes), 60)
        await safe_reply(
            update,
            f"✅ Часовой пояс сохранён: *{normalized}* (UTC{sign}{hours:02d}:{minutes:02d}).",
            parse_mode="Markdown",
        )
        await settings(update, context)
        return ConversationHandler.END
    except Exception as exc:  # noqa: BLE001 - keep retry UX for all handler failures
        logging.exception("Failed to parse timezone: %s", exc)
        await safe_reply(
            update,
            "⚠️ Не удалось распознать часовой пояс. Примеры: `Europe/Moscow`, `UTC+03`, `-5`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TZ
