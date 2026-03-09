import os
import random
import re
import html
import asyncio
import hashlib
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from .irregular_verbs import IRREGULAR_VERBS, get_random_pairs
import logging

from core.env import get_telegram_token, get_webapp_url

# Note: django.setup() is not called here because:
# 1. When imported from run.py, Django is already set up before the import
# 2. When used in Django management commands, Django is automatically set up
# Calling django.setup() here would cause it to be called twice, which is not idempotent

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, WebAppInfo
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut, TelegramError, Forbidden
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
from .models import TelegramUser, VocabularyItem, Achievement, IrregularVerbProgress
from .services import (
    bind_web_login_token,
    build_user_progress as build_user_progress_service,
    create_word as create_word_service,
    delete_all_words as delete_all_words_service,
    delete_word as delete_word_service,
    get_fake_translations as get_fake_translations_service,
    get_fake_words as get_fake_words_service,
    get_learned_words as get_learned_words_service,
    get_available_parts as get_available_parts_service,
    get_ordered_unlearned_words as get_ordered_unlearned_words_service,
    get_word_by_id as get_word_by_id_service,
    get_new_achievements as get_new_achievements_service,
    get_telegram_user_by_chat_id,
    get_user_achievements as get_user_achievements_service,
    get_user_word_list as get_user_word_list_service,
    get_user_word_page as get_user_word_page_service,
    get_unlearned_words as get_unlearned_words_service,
    reset_word_progress as reset_word_progress_service,
    resolve_shared_image_path,
    save_user as save_user_service,
    set_user_repeat_threshold as set_user_repeat_threshold_service,
    update_user_reminder_time as update_user_reminder_time_service,
    update_user_timezone as update_user_timezone_service,
    update_word_translation as update_word_translation_service,
    update_irregular_progress as update_irregular_progress_service,
    update_word_progress as update_word_progress_service,
    upsert_telegram_user,
    word_already_exists as word_already_exists_service,
)
from .openai_utils import generate_word_data, detect_language
from .utils import (
    clean_word,
    translate_to_ru,
    normalize_timezone_value,
    timezone_from_name,
    format_timezone_short,
)
from .tts import generate_tts_audio, generate_temp_audio
from django.db import IntegrityError
from django.db.models import Count, Q, Min
from django.utils.timezone import now
from datetime import timedelta, datetime
from types import SimpleNamespace

TELEGRAM_TOKEN = get_telegram_token()
WEBAPP_URL = get_webapp_url()
ADD_WORDS, WAIT_TRANSLATION, WAIT_PHOTO = range(3)
WORDS_PER_PAGE = 10
MAX_WORDS_PER_SESSION = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "media"
IMAGE_CACHE_DIR = MEDIA_ROOT / "card_images"
USER_IMAGE_DIR = MEDIA_ROOT / "user_images"


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)

# Память сессии (временно)
user_lessons = {}

SET_REMINDER_TIME = 1
SET_REMINDER_TZ = 2

MAX_IRREGULAR_PER_SESSION = 10
IRREGULARS_PER_PAGE = 20
IRREGULAR_MASTERY_THRESHOLD = 5

def _stable_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 1_000_000


def compute_image_cache_path(word_obj) -> Path:
    cache_key = f"{getattr(word_obj, 'id', '')}_{getattr(word_obj, 'word', '')}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", cache_key) or "word"
    return IMAGE_CACHE_DIR / f"{slug}.jpg"


def get_image_queries(word_obj) -> list[str]:
    word = getattr(word_obj, "word", "") or ""
    translation = getattr(word_obj, "translation", "") or ""
    example = getattr(word_obj, "example", "") or ""
    part = (getattr(word_obj, "part_of_speech", "") or "").lower()

    queries: list[str] = []
    seen = set()

    def add(q: str):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    add(word)
    if translation and translation != word:
        add(translation)
    if word and translation:
        add(f"{word} {translation}")

    if part.startswith("verb"):
        add(f"{word} verb action")
        add(f"{translation} глагол действие")
    elif part.startswith("adjective"):
        add(f"{word} adjective")
        add(f"{translation} прилагательное")

    if example:
        # берём первые несколько слов примера, чтобы уточнить контекст
        truncated = " ".join(example.split()[:6])
        add(truncated)

    return queries


def get_image_urls(word_obj, seed: int = 0) -> list[str]:
    """
    Возвращает список URL для подходящих фото (без API-ключа).
    Делаем несколько запросов (EN/ru/комбо) и разные провайдеры с детерминированным сидом, чтобы уменьшить одинаковые картинки.
    """
    urls: list[str] = []
    queries = get_image_queries(word_obj)
    if not queries:
        return urls

    for idx, query in enumerate(queries):
        sig = (seed + idx * 137) % 1_000_000
        q = quote_plus(query)
        urls.append(f"https://source.unsplash.com/random/?{q}&sig={sig}")
        urls.append(f"https://loremflickr.com/1280/720/{q}?lock={sig}")

    # запасной генератор без текста — чтобы хоть что-то отдать
    urls.append(f"https://picsum.photos/seed/{seed}/1280/720")
    return urls


async def fetch_image_bytes(word_obj) -> bytes | None:
    """
    Скачивает картинку из списка провайдеров и кладёт в кэш на диск, чтобы карточки открывались быстрее.
    """
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manual_path = getattr(word_obj, "image_path", "") or ""
    if manual_path:
        manual_file = Path(manual_path)
        if not manual_file.is_absolute():
            manual_file = PROJECT_ROOT / manual_file
        try:
            resolved = manual_file.resolve(strict=True)
        except FileNotFoundError:
            resolved = None
        except Exception:
            resolved = None

        if resolved is not None:
            allowed_roots = (IMAGE_CACHE_DIR.resolve(), USER_IMAGE_DIR.resolve())
            if any(resolved.is_relative_to(root) for root in allowed_roots):
                try:
                    return resolved.read_bytes()
                except Exception:
                    logging.warning("Failed to read manual image %s", resolved)
            else:
                logging.warning("Rejected unsafe image_path outside media dirs: %s", resolved)

    cache_key = f"{getattr(word_obj, 'id', '')}_{getattr(word_obj, 'word', '')}"
    seed = _stable_seed(cache_key or (word_obj.translation or ""))

    cache_path = compute_image_cache_path(word_obj)

    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except Exception:
            logging.warning("Failed to read image cache %s, will refetch", cache_path)

    urls = get_image_urls(word_obj, seed)

    def download(url: str) -> bytes | None:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=8) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "image" not in ctype:
                    raise ValueError(f"Unexpected content-type: {ctype}")
                return resp.read()
        except Exception as e:  # noqa: BLE001 — логируем и идём к следующему урлу
            logging.warning("Image fetch failed for %s: %s", url, e)
            return None

    for url in urls:
        data = await asyncio.to_thread(download, url)
        if data:
            try:
                cache_path.write_bytes(data)
            except Exception:
                logging.warning("Failed to write image cache %s", cache_path)
            return data
    return None

async def finalize_add_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    replies = context.user_data.pop("add_replies", [])
    context.user_data.pop("add_queue", None)
    context.user_data.pop("pending_data", None)
    context.user_data.pop("awaiting_manual_translation", None)
    logging.info("Add flow finished chat_id=%s replies=%s", update.effective_chat.id if update.effective_chat else "unknown", len(replies))

    if not replies:
        await safe_reply(update, "❌ Нечего сохранять. Попробуй снова через /add.")
        return ConversationHandler.END

    final_message = "\n\n".join(replies) + "\n\n🧠 Все слова добавлены. Чтобы начать изучение — напиши /learn"
    await safe_reply(update, final_message, parse_mode="Markdown")
    return ConversationHandler.END

async def save_and_store_reply(user, data: dict, context: ContextTypes.DEFAULT_TYPE):
    norm = clean_word(data.get("word", ""))
    try:
        await save_word(user, data["word"], data)
        append_reply(context, build_word_reply(data))
    except IntegrityError:
        logging.warning("Add flow: duplicate word for user=%s norm=%s", getattr(user, "id", "?"), norm)
        append_reply(context, f"⛔ Слово уже есть у тебя: *{norm}*")
    except Exception:
        logging.exception("Add flow: failed to save word %s for user=%s", norm, getattr(user, "id", "?"))
        append_reply(context, f"⚠️ Не удалось сохранить: *{norm or data.get('word', '?')}*")

async def process_next_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    queue = context.user_data.get("add_queue", [])
    if not queue:
        return await finalize_add_flow(update, context)

    entry = queue.pop(0)
    context.user_data["add_queue"] = queue

    user = context.user_data.get("current_user")
    if not user:
        user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
        context.user_data["current_user"] = user

    cleaned_word = entry["cleaned_word"]
    if not cleaned_word:
        append_reply(context, f"⚠️ Не удалось обработать: *{entry['original']}*")
        return await process_next_word(update, context)

    if await word_already_exists(user, cleaned_word):
        append_reply(context, f"⛔ Слово уже есть у тебя: *{cleaned_word}*")
        return await process_next_word(update, context)

    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(
                generate_word_data,
                cleaned_word,
                entry["part_of_speech_hint"],
                entry.get("manual_translation"),
            ),
            timeout=25,
        )
    except asyncio.TimeoutError:
        logging.warning("Add flow: generate_word_data timeout chat_id=%s word=%s", update.effective_chat.id, cleaned_word)
        data = None
    except Exception:
        logging.exception("Add flow: generate_word_data crashed chat_id=%s word=%s", update.effective_chat.id, cleaned_word)
        data = None
    if not data:
        append_reply(context, f"⚠️ Не удалось получить данные для: *{entry['original']}*")
        return await process_next_word(update, context)

    if entry.get("manual_translation"):
        data["translation"] = entry["manual_translation"]
        if entry.get("part_of_speech_hint"):
            data["part_of_speech"] = entry["part_of_speech_hint"]

    data["word"] = cleaned_word
    if entry["part_of_speech_hint"]:
        data["part_of_speech"] = entry["part_of_speech_hint"]

    context.user_data["pending_data"] = data

    if entry["manual_translation"]:
        return await prompt_photo_selection(update, context)

    if entry["language"] == "en":
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✍️ Ввести перевод", callback_data="translation_choice_manual"),
                    InlineKeyboardButton("🤖 Авто-перевод", callback_data="translation_choice_auto"),
                ]
            ]
        )
        await safe_reply(
            update,
            f"Как перевести *{cleaned_word}*? Выбери вариант:",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        return WAIT_TRANSLATION

    return await prompt_photo_selection(update, context)

def strip_article_or_infinitive(text: str):
    """
    Removes leading 'a', 'an', or 'to' and returns (cleaned_word, part_of_speech_hint).
    """
    cleaned = text.strip()
    lower = cleaned.lower()
    hint = None

    if lower.startswith("to "):
        cleaned = cleaned[3:].strip()
        hint = "verb"
    elif lower.startswith("a "):
        cleaned = cleaned[2:].strip()
        hint = "noun"
    elif lower.startswith("an "):
        cleaned = cleaned[3:].strip()
        hint = "noun"

    # Если после обрезки ничего не осталось — вернём исходное значение без подсказки
    if not cleaned:
        return text.strip(), None

    return cleaned, hint

def append_reply(context: ContextTypes.DEFAULT_TYPE, message: str):
    replies = context.user_data.get("add_replies", [])
    replies.append(message)
    context.user_data["add_replies"] = replies

def normalize_raw_word(text: str) -> str:
    """
    Trim, drop punctuation, collapse spaces, and limit length.
    """
    cleaned = re.sub(r"[^\w\s'-]", " ", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:255].strip()

def guess_part_of_speech_ru(translation: str | None) -> str | None:
    if not translation:
        return None
    t = translation.strip().lower()
    if t.endswith(("ть", "ться")):
        return "verb"
    if t.endswith(("ый", "ий", "ая", "ое", "ее", "ие", "ые", "ой", "ей", "ых", "их")):
        return "adjective"
    return "noun"

def build_word_reply(data: dict) -> str:
    norm = clean_word(data["word"])
    return (
        f"✅ *{norm}*\n"
        f"📖 {data['translation']}\n"
        f"🗣️ /{data['transcription']}/\n"
        f"✏️ _{data['example']}_"
    )


def esc(text: str) -> str:
    """Escape text for safe use in MarkdownV2 messages."""
    return escape_markdown(text, version=2) if text else ""

@sync_to_async
def get_or_create_user(chat_id, username):
    user = upsert_telegram_user(chat_id=chat_id, username=username)
    return user, False

@sync_to_async
def word_already_exists(user, word):
    return word_already_exists_service(user, word)

@sync_to_async
def save_word(user, _original_input, data):
    return create_word_service(user, data)

@sync_to_async
def get_fake_translations(user, exclude_word, part_of_speech=None, count=3):
    return get_fake_translations_service(user, exclude_word, part_of_speech=part_of_speech, count=count)

@sync_to_async
def update_correct_count(item_id, correct: bool):
    return update_word_progress_service(item_id, correct=correct)

@sync_to_async
def get_word_by_id(item_id):
    return get_word_by_id_service(item_id)

async def safe_reply(update: Update, text: str, **kwargs):
    target = None
    if update.message:
        target = update.message
    elif update.callback_query and update.callback_query.message:
        target = update.callback_query.message

    if not target:
        logging.warning("safe_reply: no message target (update=%s)", update)
        return None

    chat_id = update.effective_chat.id if update.effective_chat else None

    async def _send(send_kwargs: dict):
        return await target.reply_text(text, **send_kwargs)

    try:
        return await _safe_telegram_request(
            "reply_text",
            lambda: _send(kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        message = str(exc).lower()
        if "parse" in message and "parse_mode" in kwargs:
            logging.warning("safe_reply: parse error, retrying without parse_mode chat_id=%s err=%s", chat_id, exc)
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("parse_mode", None)
            return await _safe_telegram_request(
                "reply_text(no-parse)",
                lambda: _send(fallback_kwargs),
                chat_id=chat_id,
                attempts=2,
            )
        logging.warning("safe_reply: BadRequest chat_id=%s err=%s", chat_id, exc)
        return None


async def safe_answer(query):
    try:
        await query.answer()
    except BadRequest as exc:
        logging.warning("Callback answer failed (possibly stale): %s", exc)
    except (TimedOut, NetworkError, RetryAfter, Forbidden, TelegramError) as exc:
        logging.warning("Callback answer failed: %s", exc)
    except Exception:
        logging.exception("Callback answer crashed")


async def safe_photo_reply(update: Update, photo: bytes, **kwargs):
    target = None
    if update.message:
        target = update.message
    elif update.callback_query and update.callback_query.message:
        target = update.callback_query.message

    if not target:
        logging.warning("safe_photo_reply: no message target (update=%s)", update)
        return None

    chat_id = update.effective_chat.id if update.effective_chat else None

    async def _send(send_kwargs: dict):
        return await target.reply_photo(photo=photo, **send_kwargs)

    try:
        return await _safe_telegram_request(
            "reply_photo",
            lambda: _send(kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        logging.warning("safe_photo_reply: BadRequest chat_id=%s err=%s", chat_id, exc)
        return None


async def safe_edit_message_text(query, text: str, **kwargs):
    chat_id = query.message.chat_id if getattr(query, "message", None) else None
    try:
        return await _safe_telegram_request(
            "edit_message_text",
            lambda: query.edit_message_text(text, **kwargs),
            chat_id=chat_id,
            swallow_bad_request=False,
        )
    except BadRequest as exc:
        message = str(exc).lower()
        if "parse" in message and "parse_mode" in kwargs:
            logging.warning(
                "safe_edit_message_text: parse error, retrying without parse_mode chat_id=%s err=%s",
                chat_id,
                exc,
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("parse_mode", None)
            return await _safe_telegram_request(
                "edit_message_text(no-parse)",
                lambda: query.edit_message_text(text, **fallback_kwargs),
                chat_id=chat_id,
                attempts=2,
            )
        logging.warning("safe_edit_message_text: BadRequest chat_id=%s err=%s", chat_id, exc)
        return None


async def _safe_telegram_request(
    action: str,
    coro_factory,
    *,
    chat_id: int | None = None,
    attempts: int = 3,
    swallow_bad_request: bool = True,
):
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except RetryAfter as exc:
            delay = float(getattr(exc, "retry_after", 1.0)) + 0.2
            logging.warning("%s: RetryAfter %.2fs chat_id=%s", action, delay, chat_id)
            await asyncio.sleep(delay)
        except (TimedOut, NetworkError) as exc:
            if attempt >= attempts - 1:
                logging.exception("%s: network timeout exhausted chat_id=%s err=%s", action, chat_id, exc)
                return None
            delay = min(2 ** attempt, 8) + random.random() * 0.25
            logging.warning("%s: network error, retry in %.2fs chat_id=%s err=%s", action, delay, chat_id, exc)
            await asyncio.sleep(delay)
        except Forbidden as exc:
            logging.warning("%s: forbidden chat_id=%s err=%s", action, chat_id, exc)
            return None
        except BadRequest as exc:
            if swallow_bad_request:
                logging.warning("%s: bad request chat_id=%s err=%s", action, chat_id, exc)
                return None
            raise
        except TelegramError as exc:
            logging.exception("%s: TelegramError chat_id=%s err=%s", action, chat_id, exc)
            return None
        except Exception:
            logging.exception("%s: unexpected error chat_id=%s", action, chat_id)
            return None


async def on_telegram_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    err = getattr(context, "error", None)
    if isinstance(err, BaseException):
        logging.error(
            "Unhandled telegram error (update=%s)",
            update,
            exc_info=(type(err), err, err.__traceback__),
        )
    else:
        logging.error("Unhandled telegram error (update=%s)", update)

    if isinstance(update, Update) and update.effective_chat:
        await _safe_telegram_request(
            "error_notify",
            lambda: context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Что-то пошло не так. Попробуй ещё раз или начни заново через /start.",
            ),
            chat_id=update.effective_chat.id,
            attempts=1,
        )

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
    if update.callback_query:
        try:
            await update.callback_query.answer()
        except BadRequest as exc:
            logging.warning("Callback answer failed (possibly stale): %s", exc)

    if context.args:
        start_arg = context.args[0]
        if start_arg.startswith("login_"):
            token = start_arg.removeprefix("login_")
            user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
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

    keyboard = []
    if WEBAPP_URL:
        keyboard.append([InlineKeyboardButton("🚀 Открыть Lingua Voyage", web_app=WebAppInfo(url=WEBAPP_URL))])

    await safe_reply(
        update,
        "👋 Lingua Voyage теперь работает как Telegram Mini App.\n\n"
        "Открывай приложение, чтобы добавлять слова, проходить практику, смотреть прогресс и управлять словарём.\n"
        "Этот бот остаётся для входа и напоминаний о занятиях.",
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
    )

async def learn_cards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает карточки по одному слову без вариантов ответа."""
    if update.callback_query:
        await safe_answer(update.callback_query)

    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    session = context.user_data.get("cards_info")
    lesson = context.user_data.get("cards_queue")
    callback_data = update.callback_query.data if update.callback_query else None
    is_next_batch = callback_data == "cards_next_batch"
    is_fresh_start = update.message or callback_data == "start_learn_cards"
    is_start = (
        update.message
        or callback_data in ("start_learn_cards", "cards_next_batch")
    )
    is_repeat = callback_data == "cards_repeat"
    previous_batch_ids = context.user_data.get("cards_last_batch_ids") if is_next_batch else None
    seen_ids = context.user_data.get("cards_seen_ids", [])

    if is_fresh_start:
        seen_ids = []
        context.user_data["cards_seen_ids"] = seen_ids
        context.user_data.pop("cards_last_batch_ids", None)

    # Если нажали "Далее", но сессия потерялась — не перезапускаем автоматически
    if not lesson and not (is_start or is_repeat):
        await safe_reply(
            update,
            "Нет активной сессии карточек. Нажми «Учить», чтобы начать заново.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("📚 Учить", callback_data="start_learn_cards")],
                    [InlineKeyboardButton("🏠 Главное меню", callback_data="start")],
                ]
            ),
        )
        context.user_data.pop("cards_info", None)
        return

    if is_repeat:
        last_ids = context.user_data.get("cards_last_batch_ids")
        if not last_ids:
            await safe_reply(
                update,
                "Нет предыдущей подборки, чтобы повторить. Нажми «Учить», чтобы начать заново.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("📚 Учить", callback_data="start_learn_cards")],
                        [InlineKeyboardButton("🏠 Главное меню", callback_data="start")],
                    ]
                ),
            )
            return

        word_list = []
        for wid in last_ids:
            word = await get_word_by_id(wid)
            if word:
                word_list.append(word)

        if not word_list:
            await safe_reply(update, "Не удалось загрузить прошлые карточки. Попробуй начать заново через «Учить».")
            return

        lesson = list(word_list)
        context.user_data["cards_queue"] = lesson
        context.user_data["cards_info"] = {"total": len(word_list), "shown": 0}
        context.user_data["cards_last_batch_ids"] = list(last_ids)

    if not lesson and not is_repeat:
        word_list = await get_ordered_unlearned_words(
            user,
            count=MAX_WORDS_PER_SESSION,
            exclude_ids=seen_ids or None,
        )
        if not word_list:
            if seen_ids:
                await safe_reply(
                    update,
                    "Новых карточек пока нет — можно добавить слова через /add или повторить предыдущие.",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [InlineKeyboardButton("🔁 Повторить эти 10", callback_data="cards_repeat")],
                            [InlineKeyboardButton("🏠 Главное меню", callback_data="start")],
                        ]
                    ),
                )
            else:
                await safe_reply(update, "🎉 Все слова выучены! Добавь новые через /add.")
            return

        lesson = list(word_list)
        context.user_data["cards_queue"] = lesson
        context.user_data["cards_info"] = {"total": len(word_list), "shown": 0}
        context.user_data["cards_last_batch_ids"] = [w.id for w in word_list]
        context.user_data["cards_seen_ids"] = seen_ids + [w.id for w in word_list]

    word_obj = lesson.pop(0)
    context.user_data["cards_queue"] = lesson

    if not word_obj.example_translation:
        word_obj.example_translation = translate_to_ru(word_obj.example)
        await sync_to_async(word_obj.save)()

    session = context.user_data.get("cards_info", {"total": 0, "shown": 0})
    session["shown"] = session.get("shown", 0) + 1
    context.user_data["cards_info"] = session

    remaining = lesson and len(lesson) or 0
    buttons = [[InlineKeyboardButton("🏠 Главное меню", callback_data="start")]]
    if remaining:
        buttons.insert(0, [InlineKeyboardButton("➡️ Далее", callback_data="cards_next")])
    else:
        context.user_data.pop("cards_info", None)
        context.user_data.pop("cards_queue", None)
        buttons.insert(
            0,
            [
                InlineKeyboardButton("🔁 Повторить эти 10", callback_data="cards_repeat"),
            ],
        )
        buttons.insert(1, [InlineKeyboardButton("➡️ Следующие 10", callback_data="cards_next_batch")])

    transcription = word_obj.transcription or ""
    example_text = word_obj.example or ""
    example_translation = word_obj.example_translation or ""
    photo_sent = False
    image_bytes = await fetch_image_bytes(word_obj)
    if image_bytes:
        try:
            await update.effective_chat.send_photo(
                photo=InputFile(image_bytes, filename="card.jpg"),
                caption=(
                    f"📚 Карточка {session['shown']}/{session['total']}\n"
                    f"<b>{html.escape(word_obj.word)}</b> — {html.escape(word_obj.translation)}"
                ),
                parse_mode="HTML",
            )
            photo_sent = True
        except Exception:
            logging.exception("Failed to send downloaded image for %s", word_obj.word)

    if not photo_sent:
        logging.warning("No image providers worked for %s", word_obj.word)

    # отправляем аудио (не генерируем повторно, если уже есть)
    try:
        audio_path = await generate_tts_audio(
            word_obj.word, language_code=getattr(word_obj, "course_code", None)
        )
        with open(audio_path, "rb") as audio:
            if update.message:
                await update.message.reply_audio(audio)
            elif update.callback_query:
                await update.callback_query.message.reply_audio(audio)
    except Exception:
        logging.exception("Failed to send TTS for %s", word_obj.word)
        await safe_reply(update, "⚠️ Не удалось озвучить слово, попробуй позже.")

    msg = (
        f"📚 Карточка {session['shown']}/{session['total']}\n"
        f"<b>{html.escape(word_obj.word)} — {html.escape(word_obj.translation)}</b>\n"
        f"🗣️ /{html.escape(transcription)}/\n"
        f"✏️ <i>{html.escape(example_text)}</i>\n"
        f"<tg-spoiler>{html.escape(example_translation)}</tg-spoiler>"
    )
    await safe_reply(update, msg, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))


async def practice_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await safe_answer(update.callback_query)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🇬🇧 → 🇷🇺 Классический", callback_data="practice_classic")],
            [InlineKeyboardButton("🇷🇺 → 🇬🇧 Обратный", callback_data="practice_reverse")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="start")],
        ]
    )
    await safe_reply(
        update,
        "Выбери режим практики:\n"
        "• Классический — переводишь с английского на русский\n"
        "• Обратный — переводишь с русского на английский",
        reply_markup=keyboard,
    )


# --- ADD ---
async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info("Add flow started chat_id=%s", update.effective_chat.id if update.effective_chat else "unknown")
    await safe_reply(
        update,
        (
            "✍️ Введи слово или фразу. Можно несколько — каждое с новой строки.\n"
            "🔤 Можно указать артикль/частицу: `a`/`an` — существительное, `to` — глагол.\n"
            "🪄 Можно сразу указать перевод: `word - перевод` или `word: перевод`.\n"
            "✂️ Лишние знаки препинания уберём сами.\n\nКогда закончишь — просто отправь сообщение."
        ),
    )
    return ADD_WORDS

async def process_words(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )
    logging.info("Add flow: processing raw input chat_id=%s", update.effective_chat.id)
    context.user_data["current_user"] = user

    words = update.message.text.strip().split("\n")
    words = [w.strip() for w in words if w.strip()]

    entries = []
    for original_input in words:
        normalized_input = normalize_raw_word(original_input)
        cleaned_word, part_hint = strip_article_or_infinitive(normalized_input)
        if not cleaned_word:
            continue
        lang = detect_language(cleaned_word)

        # Попытка распознать формат "word - translation" или "word: translation"
        manual_translation = None
        if "-" in original_input or ":" in original_input:
            splitter = "-" if "-" in original_input else ":"
            left, right = original_input.split(splitter, 1)
            left = normalize_raw_word(left)
            right = right.strip()
            if left and right:
                manual_translation = right
                cleaned_word, part_hint = strip_article_or_infinitive(left)
                lang = detect_language(cleaned_word)
                if not part_hint:
                    part_hint = guess_part_of_speech_ru(manual_translation)

        entries.append(
            {
                "original": original_input,
                "cleaned_word": cleaned_word,
                "part_of_speech_hint": part_hint,
                "language": lang,
                "manual_translation": manual_translation,
            }
        )

    if not entries:
        await safe_reply(update, "❌ Не увидел слов. Отправь хотя бы одно слово или фразу.")
        return ConversationHandler.END

    context.user_data["add_queue"] = entries
    context.user_data["add_replies"] = []

    await safe_reply(update, "⏳ Обрабатываем слова, это может занять несколько секунд...")
    return await process_next_word(update, context)

async def handle_translation_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)

    data = context.user_data.get("pending_data")
    if not data:
        await safe_edit_message_text(query, "⚠️ Нет слова для обработки. Попробуй снова через /add.")
        return ConversationHandler.END

    choice = query.data.replace("translation_choice_", "")
    if choice == "auto":
        translation = data.get("translation", "")
        if not translation:
            context.user_data["awaiting_manual_translation"] = True
            await safe_edit_message_text(
                query,
                f"✍️ Не удалось получить перевод автоматически. Введи перевод для *{data['word']}*",
                parse_mode="Markdown",
            )
            return WAIT_TRANSLATION

        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Оставить перевод", callback_data="translation_choice_accept")],
                [InlineKeyboardButton("✍️ Ввести свой перевод", callback_data="translation_choice_manual")],
            ]
        )
        await safe_edit_message_text(
            query,
            f"🤖 Нашёл перевод для *{data['word']}*:\n*{translation}*\n\nОставить этот вариант?",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        return WAIT_TRANSLATION

    if choice == "accept":
        return await prompt_photo_selection(update, context)

    if choice == "manual":
        context.user_data["awaiting_manual_translation"] = True
        await safe_edit_message_text(query, f"✍️ Введи перевод для *{data['word']}*", parse_mode="Markdown")
        return WAIT_TRANSLATION

    await safe_edit_message_text(query, "⚠️ Неизвестный выбор. Попробуй снова через /add.")
    return ConversationHandler.END

async def handle_manual_translation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_manual_translation"):
        await safe_reply(update, "Сначала выбери, как переводить слово, используя кнопки выше.")
        return WAIT_TRANSLATION

    data = context.user_data.get("pending_data")
    if not data:
        await safe_reply(update, "⚠️ Нет слова для перевода. Запусти /add заново.")
        return ConversationHandler.END

    translation = update.message.text.strip()
    if not translation:
        await safe_reply(update, "Введите перевод текстом или выберите авто-перевод.")
        return WAIT_TRANSLATION

    data["translation"] = translation
    context.user_data["pending_data"] = data
    context.user_data["awaiting_manual_translation"] = False

    return await prompt_photo_selection(update, context)


async def prompt_photo_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Предлагает пользователю загрузить свою картинку для слова.
    """
    data = context.user_data.get("pending_data")
    if not data:
        await safe_reply(update, "⚠️ Нет слова для сохранения. Запусти /add заново.")
        return ConversationHandler.END

    keyboard_rows = [
        [InlineKeyboardButton("📤 Хочу своё фото", callback_data="photo_manual")],
        [InlineKeyboardButton("⏭️ Без картинки", callback_data="photo_skip")],
    ]
    preview_sent = False
    auto_path = None

    try:
        # Используем те же данные, что и при сохранении слова, чтобы показать предложенное фото
        word_obj = SimpleNamespace(**data)
        auto_path = compute_image_cache_path(word_obj)
        auto_bytes = await fetch_image_bytes(word_obj)
        if auto_bytes:
            context.user_data["auto_image_path"] = _to_project_relative(auto_path)
            keyboard_rows.insert(0, [InlineKeyboardButton("✅ Оставить это фото", callback_data="photo_accept")])
            await safe_reply(
                update,
                "Нашёл такое фото. Подходит?\n"
                "Если нет — просто отправь своё фото в чат одним сообщением.",
            )
            await safe_photo_reply(
                update,
                photo=auto_bytes,
                reply_markup=InlineKeyboardMarkup(keyboard_rows),
            )
            preview_sent = True
    except Exception:
        logging.exception("Add flow: failed to fetch preview image for %s", data.get("word"))

    if not preview_sent:
        await safe_reply(
            update,
            "📸 Пришли свою картинку одним сообщением, если хочешь добавить её к карточке.\n"
            "Или нажми «Без картинки» — тогда подберём позже.",
            reply_markup=InlineKeyboardMarkup(keyboard_rows),
        )

    return WAIT_PHOTO


async def finalize_pending_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = context.user_data.get("pending_data")
    if not data:
        await safe_reply(update, "⚠️ Нет слова для сохранения. Запусти /add заново.")
        return ConversationHandler.END

    user = context.user_data.get("current_user")
    if not user:
        user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
        context.user_data["current_user"] = user

    await save_and_store_reply(user, data, context)
    context.user_data.pop("pending_data", None)
    context.user_data.pop("auto_image_path", None)
    return await process_next_word(update, context)


async def handle_photo_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)

    if not context.user_data.get("pending_data"):
        await safe_edit_message_text(query, "⚠️ Нет слова для сохранения. Запусти /add заново.")
        return ConversationHandler.END

    action = query.data
    if action == "photo_manual":
        await safe_reply(update, "Пришли свою картинку одним сообщением.")
        return WAIT_PHOTO

    if action == "photo_accept":
        auto_path = context.user_data.pop("auto_image_path", None)
        if auto_path:
            data = context.user_data.get("pending_data", {})
            data["image_path"] = auto_path
            context.user_data["pending_data"] = data
        await safe_reply(update, "✅ Сохраняю с этим фото.")
        return await finalize_pending_word(update, context)

    await safe_reply(update, "✅ Продолжаем без картинки.")
    context.user_data.pop("auto_image_path", None)
    return await finalize_pending_word(update, context)


async def handle_photo_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = context.user_data.get("pending_data")
    if not data:
        await safe_reply(update, "⚠️ Нет слова для сохранения. Запусти /add заново.")
        return ConversationHandler.END

    photos = update.message.photo
    if not photos:
        await safe_reply(update, "Пришли картинку файлом или фото.")
        return WAIT_PHOTO

    largest = photos[-1]
    file = await context.bot.get_file(largest.file_id)

    USER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", data["word"]) or "word"
    filename = f"{update.effective_chat.id}_{slug}.jpg"
    dest = USER_IMAGE_DIR / filename

    await file.download_to_drive(custom_path=str(dest))

    data["image_path"] = _to_project_relative(dest)
    context.user_data["pending_data"] = data
    context.user_data.pop("auto_image_path", None)

    await safe_reply(update, "✅ Картинка сохранена. Продолжаем.")
    return await finalize_pending_word(update, context)


async def handle_photo_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "Пришли фото или нажми «Без картинки».")
    return WAIT_PHOTO

# --- LEARN ---
async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("learning_stopped"):
        context.user_data["learning_stopped"] = False
        return

    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username
    )

    logging.info("/learn invoked by %s", update.effective_chat.id)

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

        word_list = await get_unlearned_words(user, count=MAX_WORDS_PER_SESSION)

        if not word_list:
            await safe_reply(update, "🎉 Все слова выучены! Добавь новые через /add.")
            return

        user_lessons[update.effective_chat.id] = word_list
        context.user_data["session_info"] = {"correct": 0, "total": len(word_list), "answered": 0}
        lesson = word_list

    word_obj = lesson.pop(0)
    logging.info(
        "Question for %s: %s (%s)",
        update.effective_chat.id,
        word_obj.word,
        word_obj.id,
    )

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
    keyboard.append([InlineKeyboardButton("⏹ Завершить", callback_data="start")])

    transcription = word_obj.transcription or ""
    example_text = word_obj.example or ""

    try:
        audio_path = await generate_tts_audio(
            word_obj.word, language_code=getattr(word_obj, "course_code", None)
        )
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

# --- REVIEW OLD WORDS ---
async def review_old_words(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await safe_answer(update.callback_query)

    if context.user_data.get("learning_stopped"):
        context.user_data["learning_stopped"] = False
        return

    user, _ = await get_or_create_user(
        update.effective_chat.id,
        update.effective_chat.username,
    )
    lesson_key = f"review_{update.effective_chat.id}"
    lesson = user_lessons.get(lesson_key)
    session_info = context.user_data.get("review_session_info")

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
            context.user_data.pop("review_session_info", None)
            return

        word_list = await get_learned_words(user)
        if not word_list:
            await safe_reply(update, "📭 Пока нет выученных слов для повтора. Сначала изучи их через /learn.")
            return

        user_lessons[lesson_key] = word_list
        context.user_data["review_session_info"] = {
            "correct": 0,
            "total": len(word_list),
            "answered": 0,
        }
        lesson = word_list

    word_obj = lesson.pop(0)

    try:
        fakes = await get_fake_translations(
            user,
            exclude_word=word_obj.word,
            part_of_speech=word_obj.part_of_speech,
        )
        all_options = list(set(fakes + [word_obj.translation]))
        random.shuffle(all_options)
        if len(all_options) < 2:
            logging.error("Not enough options for review word %s", word_obj.word)
            await safe_reply(update, "⚠️ Не удалось подготовить варианты ответа. Попробуй позже.")
            return
    except Exception as e:
        logging.exception("Failed to prepare review options for %s: %s", word_obj.word, e)
        await safe_reply(update, "⚠️ Ошибка подготовки вопроса. Попробуй позже.")
        return

    options_map = context.user_data.setdefault("review_options", {})
    options_map[str(word_obj.id)] = all_options

    keyboard = [
        [InlineKeyboardButton(text=opt, callback_data=f"oldans|{word_obj.id}|{i}")]
        for i, opt in enumerate(all_options)
    ]
    keyboard.append([InlineKeyboardButton("⏭ Пропустить", callback_data=f"oldskip|{word_obj.id}")])
    keyboard.append([InlineKeyboardButton("⏹ Завершить", callback_data="start")])

    msg = (
        f"💬 *{esc(word_obj.word)}*\n\n"
        "Выбери правильный перевод на русском:"
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
    await safe_answer(query)

    logging.info("handle_answer from %s: %s", query.from_user.id, query.data)

    if query.data.startswith("oldskip|"):
        _, item_id = query.data.split("|")
        item = await get_word_by_id(item_id)
        logging.info("review skip word %s for user %s", item.word, query.from_user.id)
        await query.edit_message_text(
            f"⏭ Пропущено: *{esc(item.word)}* — {esc(item.translation)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        session = context.user_data.get("review_session_info")
        if session:
            session["answered"] += 1
            context.user_data["review_session_info"] = session
        await review_old_words(update, context)
        return

    if query.data.startswith("oldans|"):
        _, item_id, idx = query.data.split("|")
        options = context.user_data.get("review_options", {}).get(item_id, [])
        chosen = options[int(idx)] if int(idx) < len(options) else ""
        context.user_data.get("review_options", {}).pop(item_id, None)
        item = await get_word_by_id(item_id)
        logging.info(
            "review answer from %s: chosen=%s correct=%s",
            query.from_user.id,
            chosen,
            item.translation,
        )
        is_correct = chosen == item.translation
        if is_correct:
            await update_correct_count(item.id, correct=True)
            response = f"✅ Верно\\! *{esc(item.word)}* \\= {esc(item.translation)}"
        else:
            await mark_word_unlearned(item.id)
            response = (
                f"❌ Неверно\\. *{esc(item.word)}* \\= {esc(item.translation)}\n"
                "Слово вернулось в список изучения\\."
            )

        await query.edit_message_text(response, parse_mode=ParseMode.MARKDOWN_V2)
        session = context.user_data.get("review_session_info")
        if session:
            session["answered"] += 1
            if is_correct:
                session["correct"] += 1
            context.user_data["review_session_info"] = session

        await review_old_words(update, context)
        return

    if query.data.startswith("skip|"):
        _, item_id = query.data.split("|")
        item = await get_word_by_id(item_id)
        logging.info("skip word %s for user %s", item.word, query.from_user.id)
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
        logging.info("reverse skip word %s for user %s", item.word, query.from_user.id)
        await query.edit_message_text(
            f"⏭ Пропущено: *{esc(item.translation)}* — {esc(item.word)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

        session = context.user_data.get("session_info")
        if session:
            session["answered"] += 1
            context.user_data["session_info"] = session

        # 🗣️ Озвучка пропущенного
        try:
            audio_path = await generate_tts_audio(
                item.word, language_code=getattr(item, "course_code", None)
            )
            with open(audio_path, "rb") as audio:
                await query.message.reply_audio(audio)
        except Exception as e:
            logging.exception(
                "Failed to send reverse skip audio for %s (user %s): %s",
                item.word,
                query.from_user.id,
                e,
            )

        await learn_reverse(update, context)
        return

    if query.data.startswith("rev|"):
        _, item_id, idx = query.data.split("|")
        options = context.user_data.get("rev_options", {}).get(item_id, [])
        chosen = options[int(idx)] if int(idx) < len(options) else ""
        context.user_data.get("rev_options", {}).pop(item_id, None)
        item = await get_word_by_id(item_id)
        logging.info(
            "rev answer from %s: chosen=%s correct=%s",
            query.from_user.id,
            chosen,
            item.word,
        )
        is_correct = chosen == item.word
        await update_correct_count(item.id, correct=is_correct)

        response = (
            f"✅ Верно\\! *{esc(item.translation)}* \\= {esc(item.word)}"
            if is_correct else
            f"❌ Неверно\\. *{esc(item.translation)}* \\= {esc(item.word)}"
        )

        await query.edit_message_text(response, parse_mode=ParseMode.MARKDOWN_V2)
        session = context.user_data.get("session_info")
        if session:
            session["answered"] += 1
            if is_correct:
                session["correct"] += 1
            context.user_data["session_info"] = session

        # 🗣️ Озвучка после ответа
        try:
            audio_path = await generate_tts_audio(
                item.word, language_code=getattr(item, "course_code", None)
            )
            with open(audio_path, "rb") as audio:
                await query.message.reply_audio(audio)
        except Exception as e:
            logging.exception(
                "Failed to send reverse answer audio for %s (user %s): %s",
                item.word,
                query.from_user.id,
                e,
            )

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
    logging.info(
        "answer from %s: chosen=%s correct=%s",
        query.from_user.id,
        chosen,
        item.translation,
    )
    is_correct = chosen == item.translation

    await update_correct_count(item.id, correct=is_correct)

    if is_correct:
        response = f"✅ Верно\\! *{esc(item.word)}* \\= {esc(item.translation)}"
    else:
        response = f"❌ Неверно\\. *{esc(item.word)}* \\= {esc(item.translation)}"

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
    user_lessons.pop(f"cards_{update.effective_chat.id}", None)
    user_lessons.pop(f"review_{update.effective_chat.id}", None)
    context.user_data.pop("session_info", None)
    context.user_data.pop("aud_session_info", None)
    context.user_data.pop("aud_current_word", None)
    context.user_data.pop(f"irr_info_{update.effective_chat.id}", None)
    context.user_data.pop("cards_info", None)
    context.user_data.pop("review_session_info", None)
    context.user_data.pop("review_options", None)
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
    app.add_error_handler(on_telegram_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(start, pattern="^start$"))
    logging.info("Telegram bot is running...")
    # When running inside a background thread (see run.py) the default
    # signal handlers used by run_polling() can't be registered. Setting
    # ``stop_signals=None`` prevents the library from trying to register
    # them and avoids "set_wakeup_fd" errors.
    app.run_polling(stop_signals=None)

@sync_to_async
def save_user(user):
    return save_user_service(user)

@sync_to_async
def get_user_word_list(user):
    return get_user_word_list_service(user)

@sync_to_async
def update_irregular_progress(user, base: str, correct: bool):
    return update_irregular_progress_service(user, base, correct)

async def mywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    page = context.user_data.get("mywords_page", 0)

    words, total = await get_user_word_page(user, page)
    if not words:
        await safe_reply(update, "📭 У тебя пока нет слов для изучения. Добавь их через /add")
        return

    lines = []
    for word_id, word, tr, trans in words:
        tr_part = f" /{tr}/" if tr else ""
        lines.append(f"📘 *{word}*{tr_part} — {trans}")

    keyboard = []
    if page > 0:
        keyboard.append(InlineKeyboardButton("◀️ Назад", callback_data="mywords_prev"))
    if (page + 1) * WORDS_PER_PAGE < total:
        keyboard.append(InlineKeyboardButton("Вперёд ▶️", callback_data="mywords_next"))

    edit_row = [InlineKeyboardButton("✏️ Изменить перевод", callback_data="mywords_edit")]
    delete_row = [
        InlineKeyboardButton("🗑 Удалить все", callback_data="mywords_delete_all_confirm"),
        InlineKeyboardButton("❌ Удалить одно", callback_data="mywords_delete_one"),
    ]

    reply_keyboard = []
    if keyboard:
        reply_keyboard.append(keyboard)
    reply_keyboard.append(edit_row)
    reply_keyboard.append(delete_row)

    reply_markup = InlineKeyboardMarkup(reply_keyboard)

    target = update.message or update.callback_query.message
    await target.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=reply_markup
    )


@sync_to_async
def update_user_repeat_threshold(user, value: int):
    return set_user_repeat_threshold_service(user, value)

@sync_to_async
def get_user_by_chat(chat_id):
    return get_telegram_user_by_chat_id(chat_id)

def _main_settings_text(user):
    repeat_text = f"Слово изучается после *{user.repeat_threshold}* правильных ответов"
    review_text = "включено" if user.enable_review_old_words else "выключено"
    reminder_text = "включены" if user.reminder_enabled else "отключены"
    tz_text = format_timezone_short(user.reminder_timezone or "UTC")

    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней")
    time_text = user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"

    text = (
        "⚙️ *Настройки обучения и напоминаний:*\n\n"
        f"🔁 {repeat_text}\n"
        f"📅 Повтор старых слов: *{review_text}*\n"
        f"⏰ Напоминания: *{reminder_text}*\n"
        f"🌍 Часовой пояс: *{tz_text}*\n"
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

def _review_settings_keyboard(user: TelegramUser):
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

def _review_menu_text(user):
    status = "включено" if user.enable_review_old_words else "выключено"
    return (
        "📅 *Повтор старых слов*\n\n"
        f"Сейчас: *{status}*\n"
        f"Интервал: {user.days_before_review} дней"
    )

def _reminder_settings_keyboard(user: TelegramUser):
    toggle_label = "🔔 Выключить" if user.reminder_enabled else "🔔 Включить"
    return [
        [InlineKeyboardButton(toggle_label, callback_data="toggle_reminder")],
        [
            InlineKeyboardButton("📅 Каждый день", callback_data="set_reminder_interval_1"),
            InlineKeyboardButton("📅 Через день", callback_data="set_reminder_interval_2"),
        ],
        [InlineKeyboardButton("🌍 Часовой пояс", callback_data="set_reminder_tz")],
        [InlineKeyboardButton("🕒 Установить время", callback_data="set_reminder_time")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_settings")],
    ]

def _reminder_menu_text(user):
    reminder_text = "включены" if user.reminder_enabled else "отключены"
    interval_map = {1: "каждый день", 2: "через день"}
    interval_text = interval_map.get(user.reminder_interval_days, f"каждые {user.reminder_interval_days} дней")
    time_text = user.reminder_time.strftime("%H:%M") if user.reminder_time else "не задано"
    tz_text = format_timezone_short(user.reminder_timezone or "UTC")
    return (
        "⏰ *Напоминания*\n\n"
        f"Сейчас: *{reminder_text}*\n"
        f"Интервал: *{interval_text}*\n"
        f"Время: *{time_text}*\n"
        f"Часовой пояс: *{tz_text}*"
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
    await safe_answer(query)
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
            reply_markup=InlineKeyboardMarkup(_review_settings_keyboard(user)),
        )
        return

    if data == "settings_reminders":
        await query.edit_message_text(
            _reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard(user)),
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
            reply_markup=InlineKeyboardMarkup(_review_settings_keyboard(user)),
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
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard(user)),
        )

    elif data.startswith("set_reminder_interval_"):
        interval = int(data.split("_")[-1])
        user.reminder_interval_days = interval
        await save_user(user)
        await query.edit_message_text(
            _reminder_menu_text(user),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(_reminder_settings_keyboard(user)),
        )

    elif data == "set_reminder_time":
        await query.edit_message_text(
            "🕒 Введите время в формате `HH:MM`, например: `08:30` или `21:00`",
            parse_mode="Markdown"
        )
        return SET_REMINDER_TIME

    elif data == "set_reminder_tz":
        await query.edit_message_text(
            "🌍 Введите часовой пояс. Примеры: `Europe/Moscow`, `UTC+03`, `-5`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TZ

@sync_to_async
def get_user_progress(user):
    return build_user_progress_service(user)

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
    return get_unlearned_words_service(user, count=count, part_of_speech=part_of_speech)

@sync_to_async
def get_learned_words(user):
    return get_learned_words_service(user)

@sync_to_async
def mark_word_unlearned(item_id):
    return reset_word_progress_service(item_id)


@sync_to_async
def update_user_reminder_time(user, time_obj):
    return update_user_reminder_time_service(user, time_obj)

@sync_to_async
def update_user_timezone(user, tz_value: str):
    return update_user_timezone_service(user, tz_value)

async def set_reminder_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)

    def _parse_time(value: str):
        clean = value.replace(" ", "")
        clean = clean.replace(".", ":").replace("-", ":").replace("—", ":").replace("–", ":")
        if ":" not in clean and len(clean) == 4 and clean.isdigit():
            clean = f"{clean[:2]}:{clean[2:]}"
        parts = clean.split(":")
        if len(parts) != 2:
            raise ValueError("Wrong format")
        hours, minutes = parts
        if not (hours.isdigit() and minutes.isdigit()):
            raise ValueError("Not digits")
        parsed = datetime.strptime(f"{int(hours):02d}:{int(minutes):02d}", "%H:%M").time()
        return parsed

    try:
        parsed_time = _parse_time(text)
        await update_user_reminder_time(user, parsed_time)
        await safe_reply(
            update,
            f"✅ Напоминания будут приходить в *{parsed_time.strftime('%H:%M')}*.",
            parse_mode="Markdown",
        )
        await settings(update, context)
        return ConversationHandler.END
    except Exception as exc:  # noqa: BLE001 broad to keep UX smooth
        logging.exception("Failed to parse reminder time: %s", exc)
        await safe_reply(
            update,
            "⚠️ Неверный формат. Попробуй ещё раз в формате `HH:MM`, например `09:00`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TIME


async def set_reminder_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)

    try:
        normalized = normalize_timezone_value(text)
        await update_user_timezone(user, normalized)

        tzinfo = timezone_from_name(normalized)
        offset = (datetime.now(tzinfo).utcoffset() or timedelta(0)) if tzinfo else timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        hours, minutes = divmod(total_minutes, 60)
        offset_text = f"UTC{sign}{hours:02d}:{minutes:02d}"

        await safe_reply(
            update,
            f"✅ Часовой пояс сохранён: *{normalized}* ({offset_text}).",
            parse_mode="Markdown",
        )
        await settings(update, context)
        return ConversationHandler.END
    except Exception as exc:  # noqa: BLE001 broad catch to prompt retry
        logging.exception("Failed to parse timezone: %s", exc)
        await safe_reply(
            update,
            "⚠️ Не удалось распознать часовой пояс. Примеры: `Europe/Moscow`, `UTC+03`, `-5`",
            parse_mode="Markdown",
        )
        return SET_REMINDER_TZ

@sync_to_async
def get_user_achievements(user):
    return get_user_achievements_service(user)

@sync_to_async
def get_new_achievements(user):
    return get_new_achievements_service(user)

@sync_to_async
def get_user_word_page(user, page: int):
    return get_user_word_page_service(user, page, WORDS_PER_PAGE)


async def _show_delete_one_menu(query, user, page: int):
    items, total = await get_user_word_page(user, page)
    if not items:
        await query.edit_message_text("📭 Нет слов для удаления.")
        return

    keyboard = [
        [
            InlineKeyboardButton(
                f"❌ {w[1]}",
                callback_data=f"mywords_delete_one_confirm|{w[0]}|{page}",
            )
        ]
        for w in items
    ]

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Назад", callback_data=f"mywords_delete_one_page|{page-1}"))
    if (page + 1) * WORDS_PER_PAGE < total:
        nav_row.append(InlineKeyboardButton("Вперёд ▶️", callback_data=f"mywords_delete_one_page|{page+1}"))
    if nav_row:
        keyboard.append(nav_row)

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
                f"✏️ {w[1]}",
                callback_data=f"mywords_edit_choose|{w[0]}|{page}",
            )
        ]
        for w in items
    ]

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀️ Назад", callback_data=f"mywords_edit_page|{page-1}"))
    if (page + 1) * WORDS_PER_PAGE < total:
        nav_row.append(InlineKeyboardButton("Вперёд ▶️", callback_data=f"mywords_edit_page|{page+1}"))
    if nav_row:
        keyboard.append(nav_row)

    keyboard.append([InlineKeyboardButton("⬅️ Отмена", callback_data="start_mywords")])

    await query.edit_message_text(
        "Выбери слово для изменения перевода:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def handle_mywords_pagination(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    user_data = context.user_data

    page = user_data.get("mywords_page", 0)
    if query.data == "mywords_prev":
        page = max(0, page - 1)
    elif query.data == "mywords_next":
        page += 1

    user_data["mywords_page"] = page
    await mywords(update, context)

async def handle_mywords_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    data = query.data

    if data == "mywords_delete_all_confirm":
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Да, удалить все", callback_data="mywords_delete_all"),
                    InlineKeyboardButton("❌ Отмена", callback_data="start_mywords"),
                ]
            ]
        )
        await query.edit_message_text(
            "Удалить ВСЕ твои слова? Это действие необратимо.",
            reply_markup=keyboard,
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
        word_obj = await get_word_by_id(word_id)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Да, удалить", callback_data=f"mywords_delete_one_do|{word_id}|{page}"),
                    InlineKeyboardButton("❌ Отмена", callback_data="start_mywords"),
                ]
            ]
        )
        await query.edit_message_text(
            f"Удалить *{word_obj.word}*?",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        return

    if data.startswith("mywords_delete_one_do|"):
        _, word_id, page = data.split("|", 2)
        await delete_single_word(user, word_id)
        await query.edit_message_text("🗑 Слово удалено.")
        context.user_data["mywords_page"] = 0
        await _show_delete_one_menu(query, user, int(page))
        return

async def handle_mywords_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
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
        word_obj = await get_word_by_id(word_id)
        context.user_data["edit_translation_word_id"] = word_id
        context.user_data["edit_translation_page"] = int(page)
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("⬅️ Отмена", callback_data="mywords_edit_cancel")]]
        )
        await query.edit_message_text(
            f"Введи новый перевод для *{word_obj.word}*.\n"
            f"Текущий: *{word_obj.translation}*",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
        return

    if data == "mywords_edit_cancel":
        context.user_data.pop("edit_translation_word_id", None)
        context.user_data.pop("edit_translation_page", None)
        await mywords(update, context)
        return

@sync_to_async
def delete_single_word(user, word_id):
    return delete_word_service(user, word_id)

@sync_to_async
def delete_all_words(user):
    return delete_all_words_service(user)

@sync_to_async
def update_word_translation(user, word_id, translation):
    return update_word_translation_service(user, word_id, translation)

@sync_to_async
def get_available_parts(user):
    return get_available_parts_service(user)

@sync_to_async
def get_ordered_unlearned_words(user, count=10, exclude_ids=None):
    return get_ordered_unlearned_words_service(user, count=count, exclude_ids=exclude_ids)

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
    keyboard.append([InlineKeyboardButton("⏹ Завершить", callback_data="start")])

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
    return get_fake_words_service(exclude_word, part_of_speech=part_of_speech, count=count)

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
    """Show menu for irregular verbs."""
    keyboard = [
        [InlineKeyboardButton("🔁 Повторять", callback_data="irregular_repeat")],
        [InlineKeyboardButton("🔥 Тренироваться", callback_data="irregular_train")],
    ]
    await safe_reply(
        update,
        "Выбери режим:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def irregular_repeat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show list of all irregular verbs with pagination."""
    context.user_data["irrlist_page"] = 0
    await _show_irregular_page(update, context)


async def handle_irregular_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)
    page = context.user_data.get("irrlist_page", 0)
    if query.data == "irrlist_prev":
        page = max(0, page - 1)
    elif query.data == "irrlist_next":
        page += 1
    context.user_data["irrlist_page"] = page
    await _show_irregular_page(update, context)


async def _show_irregular_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    page = context.user_data.get("irrlist_page", 0)
    start = page * IRREGULARS_PER_PAGE
    end = start + IRREGULARS_PER_PAGE
    verbs = IRREGULAR_VERBS[start:end]

    lines = [
        f"🔹 *{v['base']}* — {v['past']} — {v['participle']} — {v['translation']}"
        for v in verbs
    ]

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️ Назад", callback_data="irrlist_prev"))
    if end < len(IRREGULAR_VERBS):
        nav.append(InlineKeyboardButton("Вперёд ▶️", callback_data="irrlist_next"))

    keyboard = []
    if nav:
        keyboard.append(nav)
    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="start_irregular")])

    target = update.message or update.callback_query.message
    await target.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


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
    keyboard.append([InlineKeyboardButton("⏹ Завершить", callback_data="start")])

    await safe_reply(
        update,
        f"🔤 *{word['base']}* — выбери правильную пару V2/V3:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_irregular_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_answer(query)

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

    # update user's irregular verb progress
    user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
    if is_correct:
        await update_irregular_progress(user, base, True)

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
    logging.info("/listening invoked by %s", update.effective_chat.id)
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
    logging.info(
        "Listening question for %s: %s (%s)",
        update.effective_chat.id,
        word_obj.word,
        word_obj.id,
    )
    audio_path = await generate_temp_audio(
        word_obj.word, language_code=getattr(word_obj, "course_code", None)
    )
    with open(audio_path, "rb") as audio:
        await safe_reply(update, "🔊 Слушай внимательно:")
        if update.message:
            await update.message.reply_audio(audio)
        elif update.callback_query:
            await update.callback_query.message.reply_audio(audio)
    os.remove(audio_path)

    mode = context.user_data.get("aud_mode", "word")
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("⏭ Пропустить", callback_data="audskip")],
            [InlineKeyboardButton("⏹ Завершить", callback_data="start")],
        ]
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
    if "edit_translation_word_id" in context.user_data:
        new_translation = update.message.text.strip()
        if not new_translation:
            await update.message.reply_text("Введите перевод текстом.")
            return

        user, _ = await get_or_create_user(update.effective_chat.id, update.effective_chat.username)
        word_id = context.user_data.pop("edit_translation_word_id")
        context.user_data.pop("edit_translation_page", None)
        item = await update_word_translation(user, word_id, new_translation)
        await update.message.reply_text(
            f"✅ Перевод обновлён: *{esc(item.word)}* \\= {esc(item.translation)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        await mywords(update, context)
        return

    if "aud_current_word" not in context.user_data:
        return

    user_answer = update.message.text.strip().lower()
    item_id = context.user_data.pop("aud_current_word")
    item = await get_word_by_id(item_id)
    mode = context.user_data.get("aud_mode", "word")

    logging.info(
        "listening answer from %s: %s (mode=%s, correct=%s/%s)",
        update.effective_chat.id,
        user_answer,
        mode,
        item.word,
        item.translation,
    )

    if mode == "translate":
        correct = item.translation.lower()
        is_correct = user_answer == correct
        await update_correct_count(item.id, correct=is_correct)
        response = (
            f"✅ Верно\\! *{esc(item.translation)}* — {esc(item.word)}"
            if is_correct else
            f"❌ Неверно\\. *{esc(item.translation)}* — {esc(item.word)}"
        )
    else:
        correct = item.word.lower()
        is_correct = user_answer == correct
        await update_correct_count(item.id, correct=is_correct)
        response = (
            f"✅ Верно\\! *{esc(item.word)}* — {esc(item.translation)}"
            if is_correct else
            f"❌ Неверно\\. *{esc(item.word)}* — {esc(item.translation)}"
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
    await safe_answer(query)

    logging.info("listening skip from %s", query.from_user.id)

    if "aud_current_word" not in context.user_data:
        return

    item_id = context.user_data.pop("aud_current_word")
    item = await get_word_by_id(item_id)
    logging.info("skipped in listening by %s: %s", query.from_user.id, item.word)

    await query.edit_message_text(
        f"⏭ Пропущено: *{esc(item.word)}* — {esc(item.translation)}",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    session = context.user_data.get("aud_session_info")
    if session:
        session["answered"] += 1
        context.user_data["aud_session_info"] = session

    await listening(update, context)
