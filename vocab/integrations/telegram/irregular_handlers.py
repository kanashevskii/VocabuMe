"""Legacy Telegram bot handlers for irregular-verb practice."""

from __future__ import annotations

import random

from asgiref.sync import sync_to_async
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from vocab.irregular_verbs import IRREGULAR_VERBS, get_random_pairs
from vocab.integrations.telegram.messaging import safe_answer, safe_reply
from vocab.integrations.telegram.session_state import user_lessons
from vocab.integrations.telegram.users import get_or_create_user
from vocab.services import get_new_achievements, update_irregular_progress

MAX_IRREGULAR_PER_SESSION = 10
IRREGULARS_PER_PAGE = 20


def get_praise(correct: int, total: int) -> str:
    if total == 0:
        return ""
    ratio = correct / total
    if ratio >= 0.9:
        return "🌟 Великолепно! Ты мастер слова!"
    if ratio >= 0.75:
        return "👍 Отличный результат!"
    if ratio >= 0.5:
        return "🙂 Хорошая работа!"
    if ratio >= 0.25:
        return "😐 Продолжай практиковаться!"
    return "💡 Не сдавайся и попробуй ещё раз!"


@sync_to_async
def _update_irregular_progress(user, base: str) -> None:
    update_irregular_progress(user, base, True)


@sync_to_async
def _get_new_achievements(user) -> list[str]:
    return get_new_achievements(user)


async def irregular_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the irregular-verb practice menu."""
    keyboard = [
        [InlineKeyboardButton("🔁 Повторять", callback_data="irregular_repeat")],
        [InlineKeyboardButton("🔥 Тренироваться", callback_data="irregular_train")],
    ]
    await safe_reply(
        update, "Выбери режим:", reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def irregular_repeat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["irrlist_page"] = 0
    await _show_irregular_page(update, context)


async def handle_irregular_list(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await safe_answer(query)
    page = context.user_data.get("irrlist_page", 0)
    if query.data == "irrlist_prev":
        page = max(0, page - 1)
    elif query.data == "irrlist_next":
        page += 1
    context.user_data["irrlist_page"] = page
    await _show_irregular_page(update, context)


async def _show_irregular_page(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    page = context.user_data.get("irrlist_page", 0)
    start = page * IRREGULARS_PER_PAGE
    end = start + IRREGULARS_PER_PAGE
    verbs = IRREGULAR_VERBS[start:end]
    lines = [
        f"🔹 *{verb['base']}* — {verb['past']} — {verb['participle']} — {verb['translation']}"
        for verb in verbs
    ]

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️ Назад", callback_data="irrlist_prev"))
    if end < len(IRREGULAR_VERBS):
        nav.append(InlineKeyboardButton("Вперёд ▶️", callback_data="irrlist_next"))
    keyboard = [nav] if nav else []
    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="start_irregular")])

    target = update.message or update.callback_query.message
    await target.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def irregular_train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Train a user on server-owned V2/V3 answer options."""
    chat_id = update.effective_chat.id
    lesson_key = f"irr_{chat_id}"
    info_key = f"irr_info_{chat_id}"
    lesson = user_lessons.get(lesson_key)
    session_info = context.user_data.get(info_key)

    if not lesson:
        if session_info:
            correct = session_info.get("correct", 0)
            total = session_info.get("total", 0)
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🏠 Главное меню", callback_data="start")]]
            )
            await safe_reply(
                update,
                f"📊 Результат: {correct} из {total} слов угадано.\n{get_praise(correct, total)}",
                reply_markup=keyboard,
            )
            context.user_data.pop(info_key, None)
            return

        words = random.sample(
            IRREGULAR_VERBS, min(len(IRREGULAR_VERBS), MAX_IRREGULAR_PER_SESSION)
        )
        user_lessons[lesson_key] = words
        context.user_data[info_key] = {"correct": 0, "total": len(words), "answered": 0}
        lesson = words

    word = lesson.pop(0)
    correct_pair = f"{word['past']} {word['participle']}"
    options = list(dict.fromkeys([correct_pair, *word["wrong_pairs"]]))
    while len(options) < 4:
        extra = get_random_pairs(word, 1, options)
        if not extra:
            break
        options.extend(extra)
    options = options[:4]
    random.shuffle(options)
    keyboard = [
        [InlineKeyboardButton(option, callback_data=f"irrans|{word['base']}|{option}")]
        for option in options
    ]
    keyboard.extend(
        [
            [
                InlineKeyboardButton(
                    "⏭ Пропустить", callback_data=f"irrskip|{word['base']}"
                )
            ],
            [InlineKeyboardButton("⏹ Завершить", callback_data="start")],
        ]
    )
    await safe_reply(
        update,
        f"🔤 *{word['base']}* — выбери правильную пару V2/V3:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_irregular_answer(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await safe_answer(query)
    data = query.data
    if data.startswith("irrskip|"):
        _, base = data.split("|", maxsplit=1)
        word = next((item for item in IRREGULAR_VERBS if item["base"] == base), None)
        if word is None:
            return
        correct_pair = f"{word['past']} {word['participle']}"
        await query.edit_message_text(f"⏭ Пропущено: {word['base']} → {correct_pair}")
        info_key = f"irr_info_{query.message.chat.id}"
        if session := context.user_data.get(info_key):
            session["answered"] += 1
        await irregular_train(update, context)
        return

    if not data.startswith("irrans|"):
        return
    _, base, chosen = data.split("|", maxsplit=2)
    word = next((item for item in IRREGULAR_VERBS if item["base"] == base), None)
    if word is None:
        return
    correct_pair = f"{word['past']} {word['participle']}"
    is_correct = chosen == correct_pair
    await query.edit_message_text(
        f"✅ Верно! {word['base']} → {correct_pair}"
        if is_correct
        else f"❌ Неверно. {word['base']} → {correct_pair}"
    )

    info_key = f"irr_info_{query.message.chat.id}"
    if session := context.user_data.get(info_key):
        session["answered"] += 1
        if is_correct:
            session["correct"] += 1

    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    if is_correct:
        await _update_irregular_progress(user, base)
    for achievement in await _get_new_achievements(user):
        await safe_reply(update, f"🏆 {achievement}")
    await irregular_train(update, context)
