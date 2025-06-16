import os
import random
import django
from decouple import config
from .irregular_verbs import IRREGULAR_VERBS, get_random_pairs
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from asgiref.sync import sync_to_async
from .models import TelegramUser, VocabularyItem, Achievement
from .openai_utils import generate_word_data
from .utils import clean_word, translate_to_ru
from .tts import generate_tts_audio, generate_temp_audio
from django.db import IntegrityError
from django.db.models import Count, Q, Min
from django.utils.timezone import now
from datetime import timedelta, datetime

TELEGRAM_TOKEN = config("TELEGRAM_TOKEN")
ADD_WORDS, LEARNING = range(2)
WORDS_PER_PAGE = 10
MAX_WORDS_PER_SESSION = 20

# Память сессии (временно)
user_lessons = {}

SET_REMINDER_TIME = 1

MAX_IRREGULAR_PER_SESSION = 10


def esc(text: str) -> str:
    """Escape text for safe use in MarkdownV2 messages."""
    return escape_markdown(text, version=2) if text else ""

@sync_to_async
def get_or_create_user(chat_id, username):
    return TelegramUser.objects.get_or_create(chat_id=chat_id, defaults={"username": username})

@sync_to_async
def word_already_exists(user, word):
    norm = clean_word(word)
    return VocabularyItem.objects.filter(user=user, normalized_word=norm).exists()

@sync_to_async
def save_word(user, original_input, data):
    word = clean_word(data["word"])  # sanitized
    normalized = word
    tr = data["transcription"]
    if any(c in tr for c in "абвгдеёжзийклмнопрстуфхцчшщыэюя"):
        tr = ""

    example_trans = data.get("example_translation")
    if not example_trans:
        example_trans = translate_to_ru(data["example"])

    return VocabularyItem.objects.create(
        user=user,
        word=word,
        normalized_word=normalized,
        translation=data["translation"],
        transcription=tr,
        example=data["example"],
        example_translation=example_trans,
        part_of_speech=data.get("part_of_speech", "unknown")
    )

@sync_to_async
def get_fake_translations(user, exclude_word, part_of_speech=None, count=3):
    qs = VocabularyItem.objects.exclude(word__iexact=exclude_word)
    if part_of_speech:
        qs = qs.filter(part_of_speech=part_of_speech)

    translations = list(
        qs.values_list("translation", flat=True)
        .distinct()
        .order_by("?")[:count]
    )

    if len(translations) < count:
        remaining = count - len(translations)
        extra_qs = VocabularyItem.objects.exclude(word__iexact=exclude_word)
        extras = list(
            extra_qs.values_list("translation", flat=True)
            .distinct()
            .order_by("?")[:remaining]
        )
        for t in extras:
            if t not in translations:
                translations.append(t)
                if len(translations) == count:
                    break

    return translations

@sync_to_async
def update_correct_count(item_id, correct: bool):
    item = VocabularyItem.objects.get(id=item_id)
    if correct:
        item.correct_count += 1
        threshold = item.user.repeat_threshold if hasattr(item.user, "repeat_threshold") else 3
        if item.correct_count >= threshold:
            item.is_learned = True
    item.save()

@sync_to_async
def get_word_by_id(item_id):
    return VocabularyItem.objects.get(id=item_id)

async def safe_reply(update: Update, text: str, **kwargs):
    if update.message:
        await update.message.reply_text(text, **kwargs)
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, **kwargs)

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

# --- START ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("➕ Добавить", callback_data="start_add"),
            InlineKeyboardButton("🎯 Учить", callback_data="start_learn"),
        ],
        [
            InlineKeyboardButton("🔄 Обратный режим", callback_data="start_learnreverse"),
            InlineKeyboardButton("🎧 Аудирование", callback_data="start_listening"),
        ],
        [InlineKeyboardButton("🔥 Неправильные глаголы", callback_data="start_irregular")],
        [
            InlineKeyboardButton("📘 Мои слова", callback_data="start_mywords"),
            InlineKeyboardButton("📊 Прогресс", callback_data="start_progress"),
        ],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="start_settings")],
    ]

    await safe_reply(
        update,
        "👋 Привет! Я помогу тебе выучить английские слова — просто и эффективно.\n\n"
        "Вот что я умею:\n"
        "➕ /add — добавить новые слова\n"
        "🎯 /learn — начать тренировку (перевод с англ. на рус.)\n"
        "🔄 /learnreverse — обратный режим (с рус. на англ.)\n"
        "🎧 /listening — аудирование по твоим словам\n"
        "📘 /mywords — список слов, которые ты учишь\n"
        "📊 /progress — посмотреть свою статистику и достижения\n"
        "⚙️ /settings — изменить настройки обучения и напоминаний\n\n"
        "🔥 /irregular — тренировать неправильные глаголы\n"
        "⏰ Я могу напоминать тебе о занятиях каждый день или через день — настрой это через /settings!\n\n"
        "🚀 Готов начать? Жми /add или /learn!",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# --- ADD ---
async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        "✍️ Введи слово или фразу. Можно несколько — каждое с новой строки.\n\nКогда закончишь — просто отправь сообщение.",
    )
    return ADD_WORDS

async def process_words(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )

    words = update.message.text.strip().split("\n")
    words = [w.strip() for w in words if w.strip()]
    replies = []

    await update.message.reply_text("⏳ Обрабатываем слова, это может занять несколько секунд...")

    for original_input in words:
        # определим язык и получим все данные
        data = generate_word_data(original_input)
        if not data:
            replies.append(f"⚠️ Не удалось получить данные для: *{original_input}*")
            continue

        norm = clean_word(data["word"])
        if await word_already_exists(user, norm):
            replies.append(f"⛔ Слово уже есть у тебя: *{norm}*")
            continue

        try:
            await save_word(user, original_input, data)
            reply = f"""✅ *{norm}*
📖 {data['translation']}
🗣️ /{data['transcription']}/
✏️ _{data['example']}_"""
        except IntegrityError:
            reply = f"⛔ Ошибка сохранения для: *{norm}*"

        replies.append(reply)

    final_message = "\n\n".join(replies) + "\n\n🧠 Все слова добавлены. Чтобы начать изучение — напиши /learn"
    await update.message.reply_text(final_message, parse_mode="Markdown")
    return ConversationHandler.END

# --- LEARN ---
async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("learning_stopped"):
        context.user_data["learning_stopped"] = False
        return

    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )

    lesson = user_lessons.get(update.effective_chat.id)
    session_info = context.user_data.get("session_info")

    if not lesson:
        if session_info:
            correct = session_info.get("correct", 0)
            total = session_info.get("total", 0)
            praise = get_praise(correct, total)
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="start")]
            ])
            await safe_reply(
                update,
                f"📊 Результат: {correct} из {total} слов угадано.\n{praise}",
                reply_markup=keyboard,
            )
            context.user_data.pop("session_info", None)
            return

        parts = await get_available_parts(user)
        selected_part = random.choice(parts) if parts else None
        context.user_data["session_part"] = selected_part
        word_list = await get_unlearned_words(user, count=MAX_WORDS_PER_SESSION, part_of_speech=selected_part)

        if not word_list:
            await safe_reply(update, "🎉 Все слова выучены! Добавь новые через /add.")
            return

        user_lessons[update.effective_chat.id] = word_list
        context.user_data["session_info"] = {"correct": 0, "total": len(word_list), "answered": 0}
        lesson = word_list

    word_obj = lesson.pop(0)

    if not word_obj.example_translation:
        word_obj.example_translation = translate_to_ru(word_obj.example)
        await sync_to_async(word_obj.save)()

    try:
        fakes = await get_fake_translations(
            user,
            exclude_word=word_obj.word,
            part_of_speech=word_obj.part_of_speech,
        )
        all_options = list(set(fakes + [word_obj.translation]))
        random.shuffle(all_options)
        if len(all_options) < 2:
            logging.error("Not enough options for word %s", word_obj.word)
            await safe_reply(update, "⚠️ Не удалось подготовить варианты ответа. Попробуй позже.")
            return
    except Exception as e:
        logging.exception("Failed to prepare options for %s: %s", word_obj.word, e)
        await safe_reply(update, "⚠️ Ошибка подготовки вопроса. Попробуй позже.")
        return

    options_map = context.user_data.setdefault("options", {})
    options_map[str(word_obj.id)] = all_options

    keyboard = [
        [InlineKeyboardButton(text=opt, callback_data=f"ans|{word_obj.id}|{i}")]
        for i, opt in enumerate(all_options)
    ]
    keyboard.append([InlineKeyboardButton("⏭ Пропустить", callback_data=f"skip|{word_obj.id}")])

    try:
        audio_path = await generate_tts_audio(word_obj.word)
        with open(audio_path, "rb") as audio:
            if update.message:
                await update.message.reply_audio(audio)
            elif update.callback_query:
                await update.callback_query.message.reply_audio(audio)
    except Exception as e:
        logging.exception("Failed to send audio for %s: %s", word_obj.word, e)

    msg = (
        f"💬 *{esc(word_obj.word)}*\n"
        f"🗣️ /{esc(word_obj.transcription)}/\n"
        f"✏️ _{esc(word_obj.example)}_\n"
        f"||{esc(word_obj.example_translation)}||\n\n"
        "Выбери правильный перевод:"
    )
    await safe_reply(
        update,
        msg,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

# --- HANDLE ANSWER ---
async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data.startswith("skip|"):
        _, item_id = query.data.split("|")
        item = await get_word_by_id(item_id)
        await query.edit_message_text(
            f"⏭ Пропущено: *{esc(item.word)}* — {esc(item.translation)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        session = context.user_data.get("session_info")
        if session:
            session["answered"] += 1
            context.user_data["session_info"] = session
        await learn(update, context)
        return

    if query.data.startswith("revskip|"):
        _, item_id = query.data.split("|")
        item = await get_word_by_id(item_id)
        await query.edit_message_text(
            f"⏭ Пропущено: *{esc(item.translation)}* — {esc(item.word)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

        session = context.user_data.get("session_info")
        if session:
            session["answered"] += 1
            context.user_data["session_info"] = session

        # 🗣️ Озвучка пропущенного
        audio_path = await generate_tts_audio(item.word)
        with open(audio_path, "rb") as audio:
            await query.message.reply_audio(audio)

        await learn_reverse(update, context)
        return

    if query.data.startswith("rev|"):
        _, item_id, idx = query.data.split("|")
        options = context.user_data.get("rev_options", {}).get(item_id, [])
        chosen = options[int(idx)] if int(idx) < len(options) else ""
        context.user_data.get("rev_options", {}).pop(item_id, None)
        item = await get_word_by_id(item_id)
        is_correct = chosen == item.word
        await update_correct_count(item.id, correct=is_correct)

        response = (
            f"✅ Верно! *{esc(item.translation)}* = {esc(item.word)}"
            if is_correct else
            f"❌ Неверно. *{esc(item.translation)}* = {esc(item.word)}"
        )

        await query.edit_message_text(response, parse_mode=ParseMode.MARKDOWN_V2)
        session = context.user_data.get("session_info")
        if session:
            session["answered"] += 1
            if is_correct:
                session["correct"] += 1
            context.user_data["session_info"] = session

        # 🗣️ Озвучка после ответа
        audio_path = await generate_tts_audio(item.word)
        with open(audio_path, "rb") as audio:
            await query.message.reply_audio(audio)

        await learn_reverse(update, context)
        return

    if query.data.startswith("ans|"):
        _, item_id, idx = query.data.split("|")
        options = context.user_data.get("options", {}).get(item_id, [])
        chosen = options[int(idx)] if int(idx) < len(options) else ""
        context.user_data.get("options", {}).pop(item_id, None)
    else:
        item_id, chosen = query.data.split("|")
    item = await get_word_by_id(item_id)
    is_correct = chosen == item.translation

    await update_correct_count(item.id, correct=is_correct)

    if is_correct:
        response = f"✅ Верно! *{esc(item.word)}* = {esc(item.translation)}"
    else:
        response = f"❌ Неверно. *{esc(item.word)}* = {esc(item.translation)}"

    await query.edit_message_text(response, parse_mode=ParseMode.MARKDOWN_V2)
    session = context.user_data.get("session_info")
    if session:
        session["answered"] += 1
        if is_correct:
            session["correct"] += 1
        context.user_data["session_info"] = session
    await learn(update, context)

    # 🎖️ Проверка новых достижений
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    new_achievements = await get_new_achievements(user)
    for a in new_achievements:
        await safe_reply(update, f"🏆 {a}")

# --- STOP ---
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["learning_stopped"] = True
    user_lessons.pop(update.effective_chat.id, None)
    user_lessons.pop(f"aud_{update.effective_chat.id}", None)
    user_lessons.pop(f"irr_{update.effective_chat.id}", None)
    context.user_data.pop("session_info", None)
    context.user_data.pop("aud_session_info", None)
    context.user_data.pop("aud_current_word", None)
    context.user_data.pop(f"irr_info_{update.effective_chat.id}", None)
    await update.message.reply_text("🛑 Обучение остановлено. Возвращайся, когда будешь готов 🙌")

# --- CANCEL ---
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Отменено.")
    return ConversationHandler.END

# --- RUN ---
def run_telegram_bot():
    # Application.run_polling() relies on an asyncio event loop. When running
    # the bot in a separate thread (as done in run.py) there is no loop by
    # default, which results in "There is no current event loop" errors.
    import asyncio

    asyncio.set_event_loop(asyncio.new_event_loop())

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("add", add_command),
            CallbackQueryHandler(add_command, pattern="^start_add$"),
        ],
        states={
            ADD_WORDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_words)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    reminder_time_conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(handle_settings_callback, pattern="^set_reminder_time$")],
        states={
            SET_REMINDER_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_reminder_time)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(reminder_time_conv)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(start, pattern="^start$") )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("learn", learn))
    app.add_handler(CallbackQueryHandler(learn, pattern="^start_learn$"))
    app.add_handler(CommandHandler("learnreverse", learn_reverse))
    app.add_handler(CallbackQueryHandler(learn_reverse, pattern="^start_learnreverse$"))
    app.add_handler(CommandHandler("listening", listening_menu))
    app.add_handler(CallbackQueryHandler(listening_menu, pattern="^start_listening$"))
    app.add_handler(CallbackQueryHandler(listening_word, pattern="^listening_word$"))
    app.add_handler(CallbackQueryHandler(listening_translate, pattern="^listening_translate$"))
    app.add_handler(CommandHandler("irregular", irregular_menu))
    app.add_handler(CallbackQueryHandler(irregular_menu, pattern="^start_irregular$"))
    app.add_handler(CommandHandler("stop", stop))
    # Support both new and legacy callback data formats
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^ans\|"))
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^\d+\|"))
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^skip\|"))
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^rev\|"))
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^rev_\d+\|"))
    app.add_handler(CallbackQueryHandler(handle_answer, pattern=r"^revskip\|"))
    app.add_handler(CallbackQueryHandler(handle_listening_skip, pattern="^audskip$"))
    app.add_handler(CallbackQueryHandler(handle_irregular_answer, pattern=r"^irr"))
    app.add_handler(CommandHandler("mywords", mywords))
    app.add_handler(CallbackQueryHandler(mywords, pattern="^start_mywords$"))
    app.add_handler(CallbackQueryHandler(handle_mywords_pagination, pattern="^mywords_"))
    app.add_handler(CommandHandler("settings", settings))
    app.add_handler(CallbackQueryHandler(settings, pattern="^start_settings$"))
    app.add_handler(CommandHandler("progress", progress))
    app.add_handler(CallbackQueryHandler(progress, pattern="^start_progress$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_listening_answer))
    app.add_handler(
        CallbackQueryHandler(
            handle_settings_callback,
            pattern=(
                "^(settings_repeat|settings_review|settings_reminders|back_to_settings|"
                "set_repeat_|toggle_review|toggle_reminder|set_review_days_|"
                "set_reminder_interval_|set_reminder_time$)"
            ),
        )
    )
    logging.info("Telegram bot is running...")
    # When running inside a background thread (see run.py) the default
    # signal handlers used by run_polling() can't be registered. Setting
    # ``stop_signals=None`` prevents the library from trying to register
    # them and avoids "set_wakeup_fd" errors.
    app.run_polling(stop_signals=None)

@sync_to_async
def save_user(user):
    user.save()

@sync_to_async
def get_user_word_list(user):
    return list(
        VocabularyItem.objects
        .filter(user=user, is_learned=False)
        .values_list("word", "transcription", "translation")
        .order_by("word")
    )

async def mywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    page = context.user_data.get("mywords_page", 0)

    words, total = await get_user_word_page(user, page)
    if not words:
        await update.message.reply_text("📭 У тебя пока нет слов для изучения. Добавь их через /add")
        return

    lines = []
    for word, tr, trans in words:
        tr_part = f" /{tr}/" if tr else ""
        lines.append(f"📘 *{word}*{tr_part} — {trans}")

    keyboard = []
    if page > 0:
        keyboard.append(InlineKeyboardButton("◀️ Назад", callback_data="mywords_prev"))
    if (page + 1) * WORDS_PER_PAGE < total:
        keyboard.append(InlineKeyboardButton("Вперёд ▶️", callback_data="mywords_next"))

    reply_markup = InlineKeyboardMarkup([keyboard]) if keyboard else None

    target = update.message or update.callback_query.message
    await target.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=reply_markup
    )


@sync_to_async
def update_user_repeat_threshold(user, value: int):
    user.repeat_threshold = value
    user.save()

@sync_to_async
def get_user_by_chat(chat_id):
    return TelegramUser.objects.get(chat_id=chat_id)

def _main_settings_text(user):
    repeat_text = f"Слово изучается после *{user.repeat_threshold}* правильных ответов"
    review_text = "включено" if user.enable_review_old_words else "выключено"
    reminder_text = "включены" if user.reminder_enabled else "отключены"

    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней")
    time_text = user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"

    text = (
        "⚙️ *Настройки обучения и напоминаний:*\n\n"
        f"🔁 {repeat_text}\n"
        f"📅 Повтор старых слов: *{review_text}*\n"
        f"⏰ Напоминания: *{reminder_text}*\n"
        f"📅 Интервал: *{interval_text}*\n"
        f"🕒 Время: *{time_text}*"
    )
    return text

def _main_settings_keyboard():
    return [
        [InlineKeyboardButton("🔁 Повтор", callback_data="settings_repeat")],
        [InlineKeyboardButton("📅 Повтор старых слов", callback_data="settings_review")],
        [InlineKeyboardButton("⏰ Напоминания", callback_data="settings_reminders")],
    ]

def _repeat_settings_keyboard():
    return [
        [
            InlineKeyboardButton("1", callback_data="set_repeat_1"),
            InlineKeyboardButton("2", callback_data="set_repeat_2"),
            InlineKeyboardButton("3", callback_data="set_repeat_3"),
            InlineKeyboardButton("4", callback_data="set_repeat_4"),
            InlineKeyboardButton("5", callback_data="set_repeat_5"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]

def _repeat_menu_text(user):
    return (
        "🔁 *Повтор слов*\n\n"
        f"Текущий порог: *{user.repeat_threshold}*\n"
        "Выберите значение:"
    )

def _review_settings_keyboard():
    return [
        [InlineKeyboardButton("🔁 Переключить", callback_data="toggle_review")],
        [
            InlineKeyboardButton("⏱ Неделя", callback_data="set_review_days_7"),
            InlineKeyboardButton("📆 Месяц", callback_data="set_review_days_30"),
            InlineKeyboardButton("🗓 3 месяца", callback_data="set_review_days_90"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]

def _review_menu_text(user):
    status = "включено" if user.enable_review_old_words else "выключено"
    return (
        "📅 *Повтор старых слов*\n\n"
        f"Сейчас: *{status}*\n"
        f"Интервал: {user.days_before_review} дней"
    )

def _reminder_settings_keyboard():
    return [
        [InlineKeyboardButton("🔔 Переключить", callback_data="toggle_reminder")],
        [
            InlineKeyboardButton("📅 Каждый день", callback_data="set_reminder_interval_1"),
            InlineKeyboardButton("📅 Через день", callback_data="set_reminder_interval_2"),
        ],
        [InlineKeyboardButton("🕒 Установить время", callback_data="set_reminder_time")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]

def _reminder_menu_text(user):
    reminder_text = "включены" if user.reminder_enabled else "отключены"
    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней")
    time_text = user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"
    return (
        "⏰ *Напоминания*\n\n"
        f"Сейчас: *{reminder_text}*\n"
        f"Интервал: *{interval_text}*\n"
        f"Время: *{time_text}*"
    )

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )

    text = _main_settings_text(user)
    await safe_reply(
        update,
        text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(_main_settings_keyboard()),
    )

async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    chat_id = update.effective_chat.id
    username = update.effective_chat.username
    user, _ = await get_or_create_user(chat_id, username)

    if data == "settings_repeat":
        await query.edit_message_text(
            _repeat_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_repeat_settings_keyboard()),
        )
        return

    if data == "settings_review":
        await query.edit_message_text(
            _review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_review_settings_keyboard()),
        )
        return

    if data == "settings_reminders":
        await query.edit_message_text(
            _reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard()),
        )
        return

    if data == "back_to_settings":
        await query.edit_message_text(
            _main_settings_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_main_settings_keyboard()),
        )
        return

    if data.startswith("set_repeat_"):
        value = int(data.split("_")[-1])
        await update_user_repeat_threshold(user, value)
        await query.edit_message_text(
            _repeat_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_repeat_settings_keyboard()),
        )
        
    elif data == "toggle_review":
        user.enable_review_old_words = not user.enable_review_old_words
        await save_user(user)
        await query.edit_message_text(
            _review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_review_settings_keyboard()),
        )

    elif data.startswith("set_review_days_"):
        days = int(data.split("_")[-1])
        user.days_before_review = days
        await save_user(user)
        await query.edit_message_text(
            _review_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_review_settings_keyboard()),
        )

    elif data == "toggle_reminder":
        user.reminder_enabled = not user.reminder_enabled
        await save_user(user)
        await query.edit_message_text(
            _reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard()),
        )

    elif data.startswith("set_reminder_interval_"):
        interval = int(data.split("_")[-1])
        user.reminder_interval_days = interval
        await save_user(user)
        await query.edit_message_text(
            _reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard()),
        )

    elif data == "set_reminder_time":
        await query.edit_message_text(
            "🕒 Введите время в формате `HH:MM`, например: `08:30` или `21:00`",
            parse_mode="Markdown"
        )
        return SET_REMINDER_TIME

@sync_to_async
def get_user_progress(user):
    total = VocabularyItem.objects.filter(user=user).count()
    learned = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    learning = total - learned
    start_date = VocabularyItem.objects.filter(user=user).aggregate(Min("created_at"))['created_at__min']

    user_stats = TelegramUser.objects.annotate(
        learned_count=Count("vocabularyitem", filter=Q(vocabularyitem__is_learned=True))
    ).order_by("-learned_count")

    total_users = user_stats.count()
    better_than = sum(1 for u in user_stats if u.learned_count < learned)
    rank_percent = round(100 * (1 - better_than / total_users)) if total_users else None

    return {
        "total": total,
        "learned": learned,
        "learning": learning,
        "start_date": start_date,
        "rank_percent": rank_percent,
        "irregular": user.irregular_correct,
    }

async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )
    stats = await get_user_progress(user)

    if stats["total"] == 0:
        await safe_reply(update, "📜 У тебя пока нет слов. Добавь их через /add")
        return

    started = stats["start_date"].strftime("%d.%m.%Y") if stats["start_date"] else "неизвестно"
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

    # 🎖 Добавим список ачивок
    earned = await get_user_achievements(user)
    if earned:
        message += "\n\n🎖 *Твои достижения:*\n" + "\n".join(f"• {a}" for a in earned)

    await safe_reply(update, message, parse_mode="Markdown")

@sync_to_async
def get_unlearned_words(user, count=10, part_of_speech=None):
    base_qs = VocabularyItem.objects.filter(user=user, is_learned=False)
    if part_of_speech:
        base_qs = base_qs.filter(part_of_speech=part_of_speech)
    base_ids = base_qs.values_list("id", flat=True)

    review_ids = []
    if user.enable_review_old_words:
        threshold = now() - timedelta(days=user.days_before_review)
        review_qs = VocabularyItem.objects.filter(
            user=user,
            is_learned=True,
            updated_at__lt=threshold
        )
        if part_of_speech:
            review_qs = review_qs.filter(part_of_speech=part_of_speech)
        review_ids = review_qs.values_list("id", flat=True)

    all_ids = list(base_ids) + list(review_ids)
    selected_ids = random.sample(all_ids, min(len(all_ids), count))

    return list(VocabularyItem.objects.filter(id__in=selected_ids))


@sync_to_async
def update_user_reminder_time(user, time_obj):
    user.reminder_time = time_obj
    user.save()

async def set_reminder_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)

    try:
        parsed_time = datetime.strptime(text, "%H:%M").time()
        await update_user_reminder_time(user, parsed_time)
        await update.message.reply_text(
            f"✅ Напоминания будут приходить в *{parsed_time.strftime('%H:%M')}*.",
            parse_mode="Markdown",
        )
        await settings(update, context)
    except ValueError:
        await update.message.reply_text("⚠️ Неверный формат. Попробуй ещё раз в формате `HH:MM`, например `09:00`", parse_mode="Markdown")
        return SET_REMINDER_TIME

    return ConversationHandler.END

@sync_to_async
def get_user_achievements(user):
    learned = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    today = now().date()

    days = user.consecutive_days or 0
    irregular = user.irregular_correct or 0

    achievements = []

    # По словам
    if learned >= 10:
        achievements.append("🎉 Выучено 10 слов — Первый шаг!")
    if learned >= 50:
        achievements.append("🏅 Выучено 50 слов — Начинающий!")
    if learned >= 100:
        achievements.append("🎯 Выучено 100 слов — Опытный!")
    if learned >= 200:
        achievements.append("🚀 Выучено 200+ слов — Гуру слов!")

    # Неправильные глаголы
    if irregular >= 10:
        achievements.append("🔤 10 неправильных глаголов освоено!")
    if irregular >= 30:
        achievements.append("🚀 30 неправильных глаголов — Продвинутый!")
    if irregular >= 60:
        achievements.append("🏆 60 неправильных глаголов — Мастер!")

    # По дням подряд
    if days >= 3:
        achievements.append("📆 3 дня подряд — Ты в ритме!")
    if days >= 7:
        achievements.append("📅 7 дней подряд — Неделя прогресса!")
    if days >= 30:
        achievements.append("🔥 30 дней подряд — Мастер привычки!")

    return achievements

@sync_to_async
def get_new_achievements(user):
    learned_words = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    days = user.consecutive_days or 0
    irregular = user.irregular_correct or 0

    word_achievements = [
        (10, "words_10", "🎉 Выучено 10 слов — Первый шаг!"),
        (50, "words_50", "🏅 Выучено 50 слов — Начинающий!"),
        (100, "words_100", "🎯 Выучено 100 слов — Опытный!"),
        (200, "words_200", "🚀 Выучено 200 слов — Гуру слов!"),
        (500, "words_500", "👑 500 слов — Мастер словарного запаса!"),
        (1000, "words_1000", "🧠 1000 слов — Легенда!"),
        (2000, "words_2000", "🌟 2000 слов — Полиглот уровня бог!"),
        (5000, "words_5000", "🏆 5000 слов — Энциклопедия на ногах!"),
    ]

    day_achievements = [
        (3, "days_3", "📆 3 дня подряд — Ты в ритме!"),
        (7, "days_7", "📅 7 дней подряд — Неделя прогресса!"),
        (14, "days_14", "🧭 14 дней подряд — Курс на успех!"),
        (30, "days_30", "🔥 30 дней подряд — Мастер привычки!"),
        (60, "days_60", "🕯️ 60 дней подряд — Упорство!"),
        (100, "days_100", "⚔️ 100 дней подряд — Воин знаний!"),
        (200, "days_200", "🛡️ 200 дней подряд — Гуру дисциплины!"),
        (365, "days_365", "🌈 365 дней подряд — Год знаний!"),
    ]

    irregular_achievements = [
        (10, "irr_10", "🔤 10 неправильных глаголов освоено!"),
        (30, "irr_30", "🚀 30 неправильных глаголов — Продвинутый!"),
        (60, "irr_60", "🏆 60 неправильных глаголов — Мастер!"),
    ]

    earned = Achievement.objects.filter(user=user).values_list("code", flat=True)
    new_achievements = []

    for threshold, code, text in word_achievements:
        if learned_words >= threshold and code not in earned:
            Achievement.objects.create(user=user, code=code)
            new_achievements.append(text)

    for threshold, code, text in day_achievements:
        if days >= threshold and code not in earned:
            Achievement.objects.create(user=user, code=code)
            new_achievements.append(text)

    for threshold, code, text in irregular_achievements:
        if irregular >= threshold and code not in earned:
            Achievement.objects.create(user=user, code=code)
            new_achievements.append(text)

    return new_achievements

@sync_to_async
def get_user_word_page(user, page: int):
    qs = VocabularyItem.objects.filter(user=user, is_learned=False).order_by("word")
    total = qs.count()
    start = page * WORDS_PER_PAGE
    end = start + WORDS_PER_PAGE
    words = list(qs[start:end].values_list("word", "transcription", "translation"))
    return words, total

async def handle_mywords_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_data = context.user_data

    page = user_data.get("mywords_page", 0)
    if query.data == "mywords_prev":
        page = max(0, page - 1)
    elif query.data == "mywords_next":
        page += 1

    user_data["mywords_page"] = page
    await mywords(update, context)

@sync_to_async
def get_available_parts(user):
    return list(
        VocabularyItem.objects
        .filter(user=user, is_learned=False)
        .values_list("part_of_speech", flat=True)
        .distinct()
    )

async def learn_reverse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("learning_stopped"):
        context.user_data["learning_stopped"] = False
        return

    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    lesson = user_lessons.get(f"rev_{update.effective_chat.id}")
    session_info = context.user_data.get("session_info")

    if not lesson:
        if session_info:
            correct = session_info.get("correct", 0)
            total = session_info.get("total", 0)
            praise = get_praise(correct, total)
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="start")]
            ])
            await safe_reply(
                update,
                f"📊 Результат: {correct} из {total} слов угадано.\n{praise}",
                reply_markup=keyboard,
            )
            context.user_data.pop("session_info", None)
            return

        word_list = await get_unlearned_words(user, count=MAX_WORDS_PER_SESSION)
        if not word_list:
            await safe_reply(update, "🎉 Все слова выучены! Добавь новые через /add.")
            return
        user_lessons[f"rev_{update.effective_chat.id}"] = word_list
        context.user_data["session_info"] = {"correct": 0, "total": len(word_list), "answered": 0}
        lesson = word_list

    word_obj = lesson.pop(0)

    fakes = await get_fake_words(user, exclude_word=word_obj.word, part_of_speech=word_obj.part_of_speech)
    all_options = fakes + [word_obj.word]
    random.shuffle(all_options)

    options_map = context.user_data.setdefault("rev_options", {})
    options_map[str(word_obj.id)] = all_options

    keyboard = [
        [InlineKeyboardButton(text=opt, callback_data=f"rev|{word_obj.id}|{i}")]
        for i, opt in enumerate(all_options)
    ]
    keyboard.append([InlineKeyboardButton("⏭ Пропустить", callback_data=f"revskip|{word_obj.id}")])

    msg = f"""💬 *{esc(word_obj.translation)}*

Выбери правильный английский эквивалент:"""
    await safe_reply(
        update,
        msg,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

@sync_to_async
def get_fake_words(user, exclude_word, part_of_speech=None, count=3):
    qs = VocabularyItem.objects.exclude(word__iexact=exclude_word)
    if part_of_speech:
        qs = qs.filter(part_of_speech=part_of_speech)

    words = list(
        qs.values_list("word", flat=True)
        .distinct()
        .order_by("?")[:count]
    )

    if len(words) < count:
        remaining = count - len(words)
        extra_qs = VocabularyItem.objects.exclude(word__iexact=exclude_word)
        extras = list(
            extra_qs.values_list("word", flat=True)
            .distinct()
            .order_by("?")[:remaining]
        )
        for w in extras:
            if w not in words:
                words.append(w)
                if len(words) == count:
                    break

    return words

# --- LISTENING ---
async def listening_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🇬🇧 Написать слово", callback_data="listening_word")],
        [InlineKeyboardButton("🇷🇺 Написать перевод", callback_data="listening_translate")],
    ]
    await safe_reply(update, "Выбери режим аудирования:", reply_markup=InlineKeyboardMarkup(keyboard))


async def listening_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["aud_mode"] = "word"
    await listening(update, context)


# --- IRREGULAR VERBS ---
async def irregular_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start training irregular verbs without choosing the form."""
    await irregular_train(update, context)


async def irregular_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Train user on irregular verbs using V2/V3 pairs."""
    chat_id = update.effective_chat.id
    key = f"irr_{chat_id}"
    info_key = f"irr_info_{chat_id}"

    lesson = user_lessons.get(key)
    session_info = context.user_data.get(info_key)

    if not lesson:
        if session_info:
            correct = session_info.get("correct", 0)
            total = session_info.get("total", 0)
            praise = get_praise(correct, total)
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="start")]
            ])
            await safe_reply(
                update,
                f"📊 Результат: {correct} из {total} слов угадано.\n{praise}",
                reply_markup=keyboard,
            )
            context.user_data.pop(info_key, None)
            return

        words = random.sample(
            IRREGULAR_VERBS,
            min(len(IRREGULAR_VERBS), MAX_IRREGULAR_PER_SESSION),
        )
        user_lessons[key] = words
        context.user_data[info_key] = {
            "correct": 0,
            "total": len(words),
            "answered": 0,
        }
        lesson = words

    word = lesson.pop(0)
    correct = f"{word['past']} {word['participle']}"
    options = [correct] + word["wrong_pairs"]
    unique_options = []
    for opt in options:
        if opt not in unique_options:
            unique_options.append(opt)

    while len(unique_options) < 4:
        extra = get_random_pairs(word, 1, unique_options)
        if not extra:
            break
        unique_options.extend(extra)

    options = unique_options[:4]
    random.shuffle(options)

    keyboard = [
        [InlineKeyboardButton(opt, callback_data=f"irrans|{word['base']}|{opt}")]
        for opt in options
    ]
    keyboard.append([
        InlineKeyboardButton(
            "⏭ Пропустить", callback_data=f"irrskip|{word['base']}"
        )
    ])

    await safe_reply(
        update,
        f"🔤 *{word['base']}* — выбери правильную пару V2/V3:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_irregular_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("irrskip"):
        _, base = data.split("|")
        word = next((w for w in IRREGULAR_VERBS if w["base"] == base), None)
        if not word:
            return
        correct = f"{word['past']} {word['participle']}"
        await query.edit_message_text(f"⏭ Пропущено: {word['base']} → {correct}")
        info_key = f"irr_info_{query.message.chat.id}"
        session = context.user_data.get(info_key)
        if session:
            session["answered"] += 1
            context.user_data[info_key] = session
        await irregular_train(update, context)
        return

    if not data.startswith("irrans"):
        return

    _, base, chosen = data.split("|")
    word = next((w for w in IRREGULAR_VERBS if w["base"] == base), None)
    if not word:
        return
    correct = f"{word['past']} {word['participle']}"
    is_correct = chosen == correct

    response = (
        f"✅ Верно! {word['base']} → {correct}" if is_correct else f"❌ Неверно. {word['base']} → {correct}"
    )
    await query.edit_message_text(response)

    info_key = f"irr_info_{query.message.chat.id}"
    session = context.user_data.get(info_key)
    if session:
        session["answered"] += 1
        if is_correct:
            session["correct"] += 1
        context.user_data[info_key] = session

    # update user's irregular stats
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    if is_correct:
        user.irregular_correct += 1
        await save_user(user)

    # check for achievements
    new_achievements = await get_new_achievements(user)
    for a in new_achievements:
        await safe_reply(update, f"🏆 {a}")

    await irregular_train(update, context)


async def listening_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["aud_mode"] = "translate"
    await listening(update, context)


async def listening(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("learning_stopped"):
        context.user_data["learning_stopped"] = False
        return

    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    lesson = user_lessons.get(f"aud_{update.effective_chat.id}")
    session_info = context.user_data.get("aud_session_info")

    if not lesson:
        if session_info:
            correct = session_info.get("correct", 0)
            total = session_info.get("total", 0)
            praise = get_praise(correct, total)
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏠 Главное меню", callback_data="start")]
            ])
            await safe_reply(
                update,
                f"📊 Результат: {correct} из {total} слов угадано.\n{praise}",
                reply_markup=keyboard,
            )
            context.user_data.pop("aud_session_info", None)
            return

        word_list = await get_unlearned_words(user, count=MAX_WORDS_PER_SESSION)
        if not word_list:
            await safe_reply(update, "🎉 Все слова выучены! Добавь новые через /add.")
            return
        user_lessons[f"aud_{update.effective_chat.id}"] = word_list
        context.user_data["aud_session_info"] = {"correct": 0, "total": len(word_list), "answered": 0}
        lesson = word_list

    word_obj = lesson.pop(0)
    audio_path = await generate_temp_audio(word_obj.word)
    with open(audio_path, "rb") as audio:
        await safe_reply(update, "🔊 Слушай внимательно:")
        if update.message:
            await update.message.reply_audio(audio)
        elif update.callback_query:
            await update.callback_query.message.reply_audio(audio)
    os.remove(audio_path)

    mode = context.user_data.get("aud_mode", "word")
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("⏭ Пропустить", callback_data="audskip")]]
    )
    if mode == "translate":
        await safe_reply(
            update,
            "Напиши перевод услышанного слова:",
            reply_markup=keyboard,
        )
    else:
        await safe_reply(
            update,
            "Напиши услышанное слово:",
            reply_markup=keyboard,
        )

    context.user_data["aud_current_word"] = word_obj.id


async def handle_listening_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "aud_current_word" not in context.user_data:
        return

    user_answer = update.message.text.strip().lower()
    item_id = context.user_data.pop("aud_current_word")
    item = await get_word_by_id(item_id)
    mode = context.user_data.get("aud_mode", "word")

    if mode == "translate":
        correct = item.translation.lower()
        is_correct = user_answer == correct
        await update_correct_count(item.id, correct=is_correct)
        response = (
            f"✅ Верно! *{esc(item.translation)}* — {esc(item.word)}"
            if is_correct else
            f"❌ Неверно. *{esc(item.translation)}* — {esc(item.word)}"
        )
    else:
        correct = item.word.lower()
        is_correct = user_answer == correct
        await update_correct_count(item.id, correct=is_correct)
        response = (
            f"✅ Верно! *{esc(item.word)}* — {esc(item.translation)}"
            if is_correct else
            f"❌ Неверно. *{esc(item.word)}* — {esc(item.translation)}"
        )

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN_V2)
    session = context.user_data.get("aud_session_info")
    if session:
        session["answered"] += 1
        if is_correct:
            session["correct"] += 1
        context.user_data["aud_session_info"] = session

    await listening(update, context)


async def handle_listening_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if "aud_current_word" not in context.user_data:
        return

    item_id = context.user_data.pop("aud_current_word")
    item = await get_word_by_id(item_id)

    await query.edit_message_text(
        f"⏭ Пропущено: *{esc(item.word)}* — {esc(item.translation)}",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    session = context.user_data.get("aud_session_info")
    if session:
        session["answered"] += 1
        context.user_data["aud_session_info"] = session

    await listening(update, context)
