"""Telegram handlers for listing and maintaining a user's vocabulary."""

from __future__ import annotations

from asgiref.sync import sync_to_async
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from vocab.integrations.telegram.messaging import safe_answer, safe_reply
from vocab.integrations.telegram.users import get_or_create_user
from vocab.services import (
    delete_all_words as delete_all_words_service,
    delete_word as delete_word_service,
    get_user_word,
    get_user_word_page as get_user_word_page_service,
    update_word_translation as update_word_translation_service,
)

WORDS_PER_PAGE = 10


@sync_to_async
def get_user_word_page(user, page: int):
    return get_user_word_page_service(user, page, WORDS_PER_PAGE)


@sync_to_async
def get_word_for_user(user, word_id: int):
    return get_user_word(user, word_id)


@sync_to_async
def delete_single_word(user, word_id: int):
    return delete_word_service(user, word_id)


@sync_to_async
def delete_all_words(user):
    return delete_all_words_service(user)


@sync_to_async
def update_word_translation(user, word_id: int, translation: str):
    return update_word_translation_service(user, word_id, translation)


async def mywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    page = context.user_data.get("mywords_page", 0)

    words, total = await get_user_word_page(user, page)
    if not words:
        await safe_reply(
            update, "📭 У тебя пока нет слов для изучения. Добавь их через /add"
        )
        return

    lines = []
    for _, word, transcription, translation in words:
        transcription_part = f" /{transcription}/" if transcription else ""
        lines.append(f"📘 *{word}*{transcription_part} — {translation}")

    navigation = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton("◀️ Назад", callback_data="mywords_prev")
        )
    if (page + 1) * WORDS_PER_PAGE < total:
        navigation.append(
            InlineKeyboardButton("Вперёд ▶️", callback_data="mywords_next")
        )

    keyboard = []
    if navigation:
        keyboard.append(navigation)
    keyboard.append(
        [InlineKeyboardButton("✏️ Изменить перевод", callback_data="mywords_edit")]
    )
    keyboard.append(
        [
            InlineKeyboardButton(
                "🗑 Удалить все", callback_data="mywords_delete_all_confirm"
            ),
            InlineKeyboardButton("❌ Удалить одно", callback_data="mywords_delete_one"),
        ]
    )

    target = update.message or update.callback_query.message
    await target.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def _show_delete_one_menu(query, user, page: int):
    items, total = await get_user_word_page(user, page)
    if not items:
        await query.edit_message_text("📭 Нет слов для удаления.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                f"❌ {word}",
                callback_data=f"mywords_delete_one_confirm|{word_id}|{page}",
            )
        ]
        for word_id, word, _, _ in items
    ]

    navigation = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton(
                "◀️ Назад", callback_data=f"mywords_delete_one_page|{page - 1}"
            )
        )
    if (page + 1) * WORDS_PER_PAGE < total:
        navigation.append(
            InlineKeyboardButton(
                "Вперёд ▶️", callback_data=f"mywords_delete_one_page|{page + 1}"
            )
        )
    if navigation:
        keyboard.append(navigation)

    keyboard.append([InlineKeyboardButton("⬅️ Отмена", callback_data="start_mywords")])
    await query.edit_message_text(
        "Выбери слово для удаления:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def _show_edit_translation_menu(query, user, page: int):
    items, total = await get_user_word_page(user, page)
    if not items:
        await query.edit_message_text("📭 Нет слов для изменения.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                f"✏️ {word}",
                callback_data=f"mywords_edit_choose|{word_id}|{page}",
            )
        ]
        for word_id, word, _, _ in items
    ]

    navigation = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton(
                "◀️ Назад", callback_data=f"mywords_edit_page|{page - 1}"
            )
        )
    if (page + 1) * WORDS_PER_PAGE < total:
        navigation.append(
            InlineKeyboardButton(
                "Вперёд ▶️", callback_data=f"mywords_edit_page|{page + 1}"
            )
        )
    if navigation:
        keyboard.append(navigation)

    keyboard.append([InlineKeyboardButton("⬅️ Отмена", callback_data="start_mywords")])
    await query.edit_message_text(
        "Выбери слово для изменения перевода:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_mywords_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    page = context.user_data.get("mywords_page", 0)
    if query.data == "mywords_prev":
        page = max(0, page - 1)
    elif query.data == "mywords_next":
        page += 1

    context.user_data["mywords_page"] = page
    await mywords(update, context)


async def handle_mywords_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    data = query.data

    if data == "mywords_delete_all_confirm":
        await query.edit_message_text(
            "Удалить ВСЕ твои слова? Это действие необратимо.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Да, удалить все", callback_data="mywords_delete_all"
                        ),
                        InlineKeyboardButton(
                            "❌ Отмена", callback_data="start_mywords"
                        ),
                    ]
                ]
            ),
        )
        return

    if data == "mywords_delete_all":
        await delete_all_words(user)
        context.user_data["mywords_page"] = 0
        await query.edit_message_text("🗑 Все слова удалены.")
        return

    if data == "mywords_delete_one":
        await _show_delete_one_menu(query, user, 0)
        return

    if data.startswith("mywords_delete_one_page|"):
        _, page = data.split("|", 1)
        await _show_delete_one_menu(query, user, int(page))
        return

    if data.startswith("mywords_delete_one_confirm|"):
        _, word_id, page = data.split("|", 2)
        word = await get_word_for_user(user, int(word_id))
        if word is None:
            await query.edit_message_text("⚠️ Слово не найдено.")
            return
        await query.edit_message_text(
            f"Удалить *{word.word}*?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Да, удалить",
                            callback_data=f"mywords_delete_one_do|{word_id}|{page}",
                        ),
                        InlineKeyboardButton(
                            "❌ Отмена", callback_data="start_mywords"
                        ),
                    ]
                ]
            ),
        )
        return

    if data.startswith("mywords_delete_one_do|"):
        _, word_id, page = data.split("|", 2)
        await delete_single_word(user, int(word_id))
        await query.edit_message_text("🗑 Слово удалено.")
        context.user_data["mywords_page"] = 0
        await _show_delete_one_menu(query, user, int(page))


async def handle_mywords_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    user, _ = await get_or_create_user(
        update.effective_chat.id, update.effective_chat.username
    )
    data = query.data

    if data == "mywords_edit":
        await _show_edit_translation_menu(query, user, 0)
        return

    if data.startswith("mywords_edit_page|"):
        _, page = data.split("|", 1)
        await _show_edit_translation_menu(query, user, int(page))
        return

    if data.startswith("mywords_edit_choose|"):
        _, word_id, page = data.split("|", 2)
        word = await get_word_for_user(user, int(word_id))
        if word is None:
            await query.edit_message_text("⚠️ Слово не найдено.")
            return
        context.user_data["edit_translation_word_id"] = word_id
        context.user_data["edit_translation_page"] = int(page)
        await query.edit_message_text(
            f"Введи новый перевод для *{word.word}*.\n"
            f"Текущий: *{word.translation}*",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "⬅️ Отмена", callback_data="mywords_edit_cancel"
                        )
                    ]
                ]
            ),
        )
        return

    if data == "mywords_edit_cancel":
        context.user_data.pop("edit_translation_word_id", None)
        context.user_data.pop("edit_translation_page", None)
        await mywords(update, context)
