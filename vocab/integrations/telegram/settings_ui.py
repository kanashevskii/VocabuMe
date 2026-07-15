"""Telegram settings presentation: text and keyboard construction only."""

from __future__ import annotations

from telegram import InlineKeyboardButton

from vocab.models import TelegramUser
from vocab.utils import format_timezone_short


def main_settings_text(user: TelegramUser) -> str:
    review_text = "включено" if user.enable_review_old_words else "выключено"
    reminder_text = "включены" if user.reminder_enabled else "отключены"
    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(
        user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней"
    )
    time_text = (
        user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"
    )
    return (
        "⚙️ *Настройки обучения и напоминаний:*\n\n"
        f"🔁 Слово изучается после *{user.repeat_threshold}* правильных ответов\n"
        f"📅 Повтор старых слов: *{review_text}*\n"
        f"⏰ Напоминания: *{reminder_text}*\n"
        f"🌍 Часовой пояс: *{format_timezone_short(user.reminder_timezone or 'UTC')}*\n"
        f"📅 Интервал: *{interval_text}*\n"
        f"🕒 Время: *{time_text}*"
    )


def main_settings_keyboard() -> list[list[InlineKeyboardButton]]:
    return [
        [InlineKeyboardButton("🔁 Повтор", callback_data="settings_repeat")],
        [
            InlineKeyboardButton(
                "📅 Повтор старых слов", callback_data="settings_review"
            )
        ],
        [InlineKeyboardButton("⏰ Напоминания", callback_data="settings_reminders")],
    ]


def repeat_settings_keyboard() -> list[list[InlineKeyboardButton]]:
    return [
        [
            InlineKeyboardButton(str(value), callback_data=f"set_repeat_{value}")
            for value in range(1, 6)
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]


def repeat_menu_text(user: TelegramUser) -> str:
    return (
        "🔁 *Повтор слов*\n\n"
        f"Текущий порог: *{user.repeat_threshold}*\n"
        "Выберите значение:"
    )


def review_settings_keyboard(user: TelegramUser) -> list[list[InlineKeyboardButton]]:
    toggle_label = "🔁 Выключить" if user.enable_review_old_words else "🔁 Включить"
    return [
        [InlineKeyboardButton(toggle_label, callback_data="toggle_review")],
        [
            InlineKeyboardButton("⏱ Неделя", callback_data="set_review_days_7"),
            InlineKeyboardButton("📆 Месяц", callback_data="set_review_days_30"),
            InlineKeyboardButton("🗓 3 месяца", callback_data="set_review_days_90"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]


def review_menu_text(user: TelegramUser) -> str:
    status = "включено" if user.enable_review_old_words else "выключено"
    return (
        "📅 *Повтор старых слов*\n\n"
        f"Сейчас: *{status}*\n"
        f"Интервал: {user.days_before_review} дней"
    )


def reminder_settings_keyboard(user: TelegramUser) -> list[list[InlineKeyboardButton]]:
    toggle_label = "🔔 Выключить" if user.reminder_enabled else "🔔 Включить"
    return [
        [InlineKeyboardButton(toggle_label, callback_data="toggle_reminder")],
        [
            InlineKeyboardButton(
                "📅 Каждый день", callback_data="set_reminder_interval_1"
            ),
            InlineKeyboardButton(
                "📅 Через день", callback_data="set_reminder_interval_2"
            ),
        ],
        [InlineKeyboardButton("🌍 Часовой пояс", callback_data="set_reminder_tz")],
        [
            InlineKeyboardButton(
                "🕒 Установить время", callback_data="set_reminder_time"
            )
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]


def reminder_menu_text(user: TelegramUser) -> str:
    reminder_text = "включены" if user.reminder_enabled else "отключены"
    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(
        user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней"
    )
    time_text = (
        user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"
    )
    return (
        "⏰ *Напоминания*\n\n"
        f"Сейчас: *{reminder_text}*\n"
        f"Интервал: *{interval_text}*\n"
        f"Время: *{time_text}*\n"
        f"Часовой пояс: *{format_timezone_short(user.reminder_timezone or 'UTC')}*"
    )
