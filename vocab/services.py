from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock, Thread
from typing import Iterable
import logging
import random
import re
import time

from django.contrib.auth.hashers import check_password, make_password
from django.db.models import Count, Min, Q
from django.utils import timezone
from django.utils.timezone import now

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow may be absent in some envs
    Image = None

from .irregular_verbs import IRREGULAR_VERBS, get_random_pairs
from .models import (
    AddWordDraft,
    Achievement,
    DEFAULT_STUDIED_LANGUAGE,
    IrregularVerbProgress,
    PackPreparedWord,
    STUDIED_LANGUAGE_CHOICES,
    TelegramUser,
    UserCourseProgress,
    VocabularyItem,
    WebLoginToken,
)
from .openai_utils import (
    build_visual_prompt,
    generate_card_image,
    generate_word_data,
    generate_word_data_batch,
)
from .utils import clean_word, normalize_timezone_value, translate_to_ru
from .word_packs import get_pack_definitions, get_pack_level

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "media"
IMAGE_CACHE_DIR = MEDIA_ROOT / "card_images"
USER_IMAGE_DIR = MEDIA_ROOT / "user_images"
DRAFT_IMAGE_DIR = MEDIA_ROOT / "draft_images"
_IMAGE_OPTIMIZATION_LOCK = Lock()
_IMAGE_OPTIMIZATION_IN_FLIGHT: set[str] = set()
_PACK_PREPARATION_LOCK = Lock()
_PACK_PREPARATION_IN_FLIGHT: set[str] = set()
logger = logging.getLogger(__name__)
EXERCISE_TYPE_LABELS = {
    "practice_en_ru": "Тест EN -> RU",
    "practice_ru_en": "Тест RU -> EN",
    "listening_word": "Аудирование: слово",
    "listening_translate": "Аудирование: перевод",
    "speaking": "Говорение",
}
EXERCISE_PRIORITY = [
    "practice_en_ru",
    "listening_word",
    "practice_ru_en",
    "speaking",
    "listening_translate",
]
ACHIEVEMENT_DEFINITIONS = [
    {"kind": "words", "threshold": 10, "text": "🎉 Выучено 10 слов — Первый шаг!"},
    {"kind": "words", "threshold": 50, "text": "🌿 Выучено 50 слов — Хороший темп!"},
    {"kind": "words", "threshold": 100, "text": "🎯 Выучено 100 слов — Опытный!"},
    {"kind": "words", "threshold": 200, "text": "🚀 Выучено 200+ слов — Гуру слов!"},
    {"kind": "practice", "threshold": 10, "text": "🎲 10 тестов — Ты вошёл в ритм!"},
    {"kind": "practice", "threshold": 50, "text": "🧠 50 тестов — Отличная реакция!"},
    {
        "kind": "listening",
        "threshold": 10,
        "text": "🎧 10 аудио-ответов — Уже слышишь лучше!",
    },
    {
        "kind": "listening",
        "threshold": 50,
        "text": "📻 50 аудио-ответов — Слух прокачан!",
    },
    {"kind": "speaking", "threshold": 10, "text": "🎙️ 10 произношений — Голос в деле!"},
    {
        "kind": "speaking",
        "threshold": 50,
        "text": "🗣️ 50 произношений — Звучишь увереннее!",
    },
    {
        "kind": "review",
        "threshold": 10,
        "text": "🔁 10 повторов — Память закрепляется!",
    },
    {
        "kind": "review",
        "threshold": 50,
        "text": "🪄 50 повторов — Старые слова держатся!",
    },
    {
        "kind": "irregular",
        "threshold": 10,
        "text": "🔤 10 неправильных глаголов — База собрана!",
    },
    {
        "kind": "irregular",
        "threshold": 30,
        "text": "🧩 30 неправильных глаголов — Уже уверенно!",
    },
    {
        "kind": "irregular",
        "threshold": 60,
        "text": "🏆 60 неправильных глаголов — Мастер форм!",
    },
    {"kind": "days", "threshold": 3, "text": "📆 3 дня подряд — Ты в ритме!"},
    {"kind": "days", "threshold": 7, "text": "📅 7 дней подряд — Неделя прогресса!"},
    {"kind": "days", "threshold": 14, "text": "🧭 14 дней подряд — Курс на успех!"},
    {"kind": "days", "threshold": 30, "text": "🔥 30 дней подряд — Мастер привычки!"},
    {"kind": "days", "threshold": 60, "text": "🕯️ 60 дней подряд — Упорство без пауз!"},
    {"kind": "days", "threshold": 100, "text": "⚔️ 100 дней подряд — Воин знаний!"},
    {"kind": "days", "threshold": 200, "text": "🛡️ 200 дней подряд — Гуру дисциплины!"},
    {"kind": "days", "threshold": 365, "text": "🌈 365 дней подряд — Год знаний!"},
]
AVAILABLE_STUDIED_LANGUAGES = [
    {"code": code, "label": label} for code, label in STUDIED_LANGUAGE_CHOICES
]


@dataclass
class ParsedWordEntry:
    word: str
    translation_hint: str | None = None


def normalize_course_code(course_code: str | None) -> str:
    value = (course_code or DEFAULT_STUDIED_LANGUAGE).strip().lower()
    supported = {code for code, _ in STUDIED_LANGUAGE_CHOICES}
    return value if value in supported else DEFAULT_STUDIED_LANGUAGE


def get_active_course_code(user: TelegramUser) -> str:
    return normalize_course_code(getattr(user, "active_studied_language", None))


def get_or_create_user_course_progress(
    user: TelegramUser, course_code: str | None = None
) -> UserCourseProgress:
    normalized = normalize_course_code(course_code or get_active_course_code(user))
    progress, _ = UserCourseProgress.objects.get_or_create(
        user=user, course_code=normalized
    )
    return progress


def get_course_pack_definitions(course_code: str | None = None) -> list[dict]:
    active_course = normalize_course_code(course_code)
    return get_pack_definitions() if active_course == "en" else []


def get_course_progress_stats(user: TelegramUser, course_code: str | None = None) -> dict:
    active_course = normalize_course_code(course_code or get_active_course_code(user))
    progress = get_or_create_user_course_progress(user, active_course)
    learned = VocabularyItem.objects.filter(
        user=user, course_code=active_course, is_learned=True
    ).count()
    irregular = IrregularVerbProgress.objects.filter(
        user=user, course_code=active_course, is_learned=True
    ).count()
    return {
        "words": learned,
        "days": progress.consecutive_days or 0,
        "irregular": irregular,
        "practice": progress.practice_correct or 0,
        "listening": progress.listening_correct or 0,
        "speaking": progress.speaking_correct or 0,
        "review": progress.review_correct or 0,
    }


def normalize_translation_value(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower().replace("ё", "е"))
    return normalized.strip(" \t\r\n.,;:!?\"'")


def split_translation_variants(translation: str) -> list[str]:
    source = (translation or "").strip()
    if not source:
        return []

    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in source:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1

        if depth == 0 and char in {",", "/"}:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    normalized_variants: list[str] = []
    for candidate in [source, *parts]:
        normalized = normalize_translation_value(candidate)
        if normalized and normalized not in normalized_variants:
            normalized_variants.append(normalized)
    return normalized_variants


def is_translation_answer_correct(answer: str, translation: str) -> bool:
    normalized_answer = normalize_translation_value(answer)
    if not normalized_answer:
        return False
    return normalized_answer in split_translation_variants(translation)


def is_single_typo_match(answer: str, expected: str) -> bool:
    normalized_answer = clean_word(answer)
    normalized_expected = clean_word(expected)
    if not normalized_answer or not normalized_expected:
        return False
    if normalized_answer == normalized_expected:
        return False

    len_answer = len(normalized_answer)
    len_expected = len(normalized_expected)
    if abs(len_answer - len_expected) > 1:
        return False

    if len_answer == len_expected:
        mismatches = sum(
            1
            for left, right in zip(normalized_answer, normalized_expected)
            if left != right
        )
        return mismatches <= 1

    if len_answer > len_expected:
        longer, shorter = normalized_answer, normalized_expected
    else:
        longer, shorter = normalized_expected, normalized_answer

    index_longer = 0
    index_shorter = 0
    edits = 0
    while index_longer < len(longer) and index_shorter < len(shorter):
        if longer[index_longer] == shorter[index_shorter]:
            index_longer += 1
            index_shorter += 1
            continue
        edits += 1
        if edits > 1:
            return False
        index_longer += 1

    return True


def upsert_telegram_user(chat_id: int, username: str | None = None) -> TelegramUser:
    user, created = TelegramUser.objects.get_or_create(
        chat_id=chat_id,
        defaults={"username": username},
    )
    if not created and username and user.username != username:
        user.username = username
        user.save(update_fields=["username"])
    return user


def get_telegram_user_by_id(user_id: int) -> TelegramUser | None:
    return TelegramUser.objects.filter(id=user_id).first()


def get_telegram_user_by_chat_id(chat_id: int) -> TelegramUser | None:
    return TelegramUser.objects.filter(chat_id=chat_id).first()


def _next_web_chat_id() -> int:
    last_web_user = (
        TelegramUser.objects.filter(chat_id__lt=0).order_by("chat_id").first()
    )
    if not last_web_user:
        return -1
    return last_web_user.chat_id - 1


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def create_web_user(email: str, password: str) -> TelegramUser:
    normalized_email = normalize_email(email)
    if not normalized_email:
        raise ValueError("Email is required.")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    if TelegramUser.objects.filter(email=normalized_email).exists():
        raise ValueError("A user with this email already exists.")

    username = normalized_email.split("@", 1)[0][:255] or "webuser"
    return TelegramUser.objects.create(
        chat_id=_next_web_chat_id(),
        username=username,
        email=normalized_email,
        password_hash=make_password(password),
        auth_provider="web",
    )


def authenticate_web_user(email: str, password: str) -> TelegramUser | None:
    normalized_email = normalize_email(email)
    if not normalized_email or not password:
        return None

    user = TelegramUser.objects.filter(
        email=normalized_email, auth_provider="web"
    ).first()
    if not user or not user.password_hash:
        return None
    if not check_password(password, user.password_hash):
        return None
    return user


def get_achievement_stats(
    user: TelegramUser, course_code: str | None = None
) -> dict:
    return get_course_progress_stats(user, course_code=course_code)


def get_user_achievements(
    user: TelegramUser, course_code: str | None = None
) -> list[str]:
    stats = get_achievement_stats(user, course_code=course_code)
    return [
        item["text"]
        for item in ACHIEVEMENT_DEFINITIONS
        if stats[item["kind"]] >= item["threshold"]
    ]


def get_new_achievements(
    user: TelegramUser, course_code: str | None = None
) -> list[str]:
    active_course = normalize_course_code(course_code or get_active_course_code(user))
    stats = get_achievement_stats(user, course_code=active_course)
    earned = set(
        Achievement.objects.filter(user=user, course_code=active_course).values_list(
            "code", flat=True
        )
    )
    new_achievements: list[str] = []

    for item in ACHIEVEMENT_DEFINITIONS:
        code = f"{item['kind']}_{item['threshold']}"
        if stats[item["kind"]] >= item["threshold"] and code not in earned:
            Achievement.objects.create(user=user, course_code=active_course, code=code)
            new_achievements.append(item["text"])
    return new_achievements


def get_pending_achievements(
    user: TelegramUser, course_code: str | None = None
) -> list[dict]:
    stats = get_achievement_stats(user, course_code=course_code)

    pending: list[dict] = []
    for item in ACHIEVEMENT_DEFINITIONS:
        current = stats[item["kind"]]
        if current >= item["threshold"]:
            continue
        pending.append(
            {
                "kind": item["kind"],
                "text": item["text"],
                "current": current,
                "target": item["threshold"],
            }
        )
    return pending[:12]


def get_pending_achievement_highlights(
    user: TelegramUser, course_code: str | None = None
) -> list[dict]:
    pending = get_pending_achievements(user, course_code=course_code)
    highlights: list[dict] = []
    seen_kinds: set[str] = set()
    for item in pending:
        if item["kind"] in seen_kinds:
            continue
        seen_kinds.add(item["kind"])
        highlights.append(item)
    return highlights


def build_user_progress(user: TelegramUser) -> dict:
    active_course = get_active_course_code(user)
    course_progress = get_or_create_user_course_progress(user, active_course)
    total = VocabularyItem.objects.filter(user=user, course_code=active_course).count()
    learned = VocabularyItem.objects.filter(
        user=user, course_code=active_course, is_learned=True
    ).count()
    learning = total - learned
    irregular_learned = IrregularVerbProgress.objects.filter(
        user=user, course_code=active_course, is_learned=True
    ).count()
    start_date = VocabularyItem.objects.filter(
        user=user, course_code=active_course
    ).aggregate(Min("created_at"))["created_at__min"]
    today = now().date()
    learned_today = VocabularyItem.objects.filter(
        user=user, course_code=active_course, learned_at__date=today
    ).count()
    current_moment = timezone.now()
    week_window_start = current_moment - timedelta(days=7)
    month_window_start = current_moment - timedelta(days=30)
    learned_week = VocabularyItem.objects.filter(
        user=user, course_code=active_course, learned_at__gte=week_window_start
    ).count()
    learned_month = VocabularyItem.objects.filter(
        user=user, course_code=active_course, learned_at__gte=month_window_start
    ).count()

    user_stats = TelegramUser.objects.annotate(
        learned_count=Count(
            "vocabularyitem",
            filter=Q(
                vocabularyitem__course_code=active_course,
                vocabularyitem__is_learned=True,
            ),
        )
    ).order_by("-learned_count")

    total_users = user_stats.count()
    better_than = sum(
        1 for candidate in user_stats if candidate.learned_count < learned
    )
    rank_percent = round(100 * (1 - better_than / total_users)) if total_users else None

    return {
        "total": total,
        "learned": learned,
        "learning": learning,
        "irregular": irregular_learned,
        "start_date": start_date.isoformat() if start_date else None,
        "rank_percent": rank_percent,
        "achievements": get_user_achievements(user, course_code=active_course),
        "pending_achievements": get_pending_achievements(
            user, course_code=active_course
        ),
        "pending_achievement_highlights": get_pending_achievement_highlights(
            user, course_code=active_course
        ),
        "streak_days": course_progress.consecutive_days,
        "study_days": course_progress.total_study_days,
        "studied_today": course_progress.last_study_date == today,
        "learned_today": learned_today,
        "learned_week": learned_week,
        "learned_month": learned_month,
        "practice_correct": course_progress.practice_correct,
        "listening_correct": course_progress.listening_correct,
        "speaking_correct": course_progress.speaking_correct,
        "review_correct": course_progress.review_correct,
        "course_code": active_course,
    }


def _webp_variant_path(source: Path) -> Path:
    return source.with_suffix(".webp")


def _optimize_image_to_webp(source: Path) -> Path | None:
    if Image is None or source.suffix.lower() == ".webp":
        return source if source.exists() else None

    target = _webp_variant_path(source)
    try:
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
    except OSError:
        pass

    try:
        with Image.open(source) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            img.save(target, format="WEBP", quality=82, method=6)
        return target
    except Exception:
        return None


def _schedule_image_optimization(source: Path) -> None:
    if Image is None or source.suffix.lower() == ".webp" or not source.exists():
        return

    key = str(source)
    with _IMAGE_OPTIMIZATION_LOCK:
        if key in _IMAGE_OPTIMIZATION_IN_FLIGHT:
            return
        _IMAGE_OPTIMIZATION_IN_FLIGHT.add(key)

    def _run() -> None:
        try:
            _optimize_image_to_webp(source)
        finally:
            with _IMAGE_OPTIMIZATION_LOCK:
                _IMAGE_OPTIMIZATION_IN_FLIGHT.discard(key)

    Thread(target=_run, daemon=True).start()


def _preferred_served_image(source: Path) -> Path:
    if source.suffix.lower() == ".webp":
        return source
    webp = _webp_variant_path(source)
    if webp.exists():
        return webp
    _schedule_image_optimization(source)
    return source


def get_word_image_file(item: VocabularyItem) -> Path | None:
    candidates: list[Path] = []

    if item.image_path:
        raw_path = Path(item.image_path)
        candidates.append(
            raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        )

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", item.word or "").strip("_") or "word"
    candidates.extend(
        [
            IMAGE_CACHE_DIR / f"{item.id}_{slug}.jpg",
            IMAGE_CACHE_DIR / f"_{slug}.jpg",
        ]
    )

    allowed_roots = (
        IMAGE_CACHE_DIR.resolve(),
        USER_IMAGE_DIR.resolve(),
        DRAFT_IMAGE_DIR.resolve(),
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if any(resolved.is_relative_to(root) for root in allowed_roots):
            return _preferred_served_image(resolved)
    return None


def serialize_user(user: TelegramUser) -> dict:
    return {
        "id": user.id,
        "chat_id": user.chat_id,
        "username": user.username,
        "email": user.email,
        "auth_provider": user.auth_provider,
        "active_studied_language": get_active_course_code(user),
        "available_studied_languages": AVAILABLE_STUDIED_LANGUAGES,
        "display_name": user.username or user.email or f"user{user.chat_id}",
        "joined_at": user.joined_at.isoformat() if user.joined_at else None,
    }


def serialize_word(item: VocabularyItem) -> dict:
    image_file = get_word_image_file(item)
    return {
        "id": item.id,
        "word": item.word,
        "translation": item.translation,
        "transcription": item.transcription,
        "example": item.example,
        "example_translation": item.example_translation,
        "part_of_speech": item.part_of_speech,
        "course_code": item.course_code,
        "correct_count": item.correct_count,
        "completed_exercise_types": list(item.completed_exercise_types or []),
        "is_learned": item.is_learned,
        "image_regeneration_count": item.image_regeneration_count,
        "image_generation_in_progress": item.image_generation_in_progress,
        "image_path": item.image_path,
        "has_image": image_file is not None,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def list_words(
    user: TelegramUser, search: str = "", status: str = "all", limit: int = 100
) -> list[VocabularyItem]:
    qs = VocabularyItem.objects.filter(
        user=user, course_code=get_active_course_code(user)
    ).order_by("-updated_at", "-id")

    if search:
        qs = qs.filter(
            Q(word__icontains=search)
            | Q(translation__icontains=search)
            | Q(example__icontains=search)
        )

    if status == "learning":
        qs = qs.filter(is_learned=False)
    elif status == "learned":
        qs = qs.filter(is_learned=True)

    return list(qs[:limit])


def get_user_word_page(
    user: TelegramUser, page: int, page_size: int
) -> tuple[list[tuple[int, str, str, str]], int]:
    qs = VocabularyItem.objects.filter(
        user=user, course_code=get_active_course_code(user), is_learned=False
    ).order_by("word")
    total = qs.count()
    start = page * page_size
    end = start + page_size
    words = list(
        qs[start:end].values_list("id", "word", "transcription", "translation")
    )
    return words, total


def get_user_word_list(user: TelegramUser) -> list[tuple[str, str, str]]:
    return list(
        VocabularyItem.objects.filter(
            user=user, course_code=get_active_course_code(user), is_learned=False
        )
        .values_list("word", "transcription", "translation")
        .order_by("word")
    )


def get_user_word(user: TelegramUser, word_id: int) -> VocabularyItem | None:
    return VocabularyItem.objects.filter(
        id=word_id, user=user, course_code=get_active_course_code(user)
    ).first()


def get_word_by_id(word_id: int) -> VocabularyItem | None:
    return VocabularyItem.objects.filter(id=word_id).first()


def get_user_draft(user: TelegramUser, draft_id: int) -> AddWordDraft | None:
    return AddWordDraft.objects.filter(
        id=draft_id, user=user, course_code=get_active_course_code(user)
    ).first()


def delete_user_draft(user: TelegramUser, draft_id: int) -> bool:
    deleted, _ = AddWordDraft.objects.filter(
        id=draft_id, user=user, course_code=get_active_course_code(user)
    ).delete()
    return deleted > 0


def word_already_exists(user: TelegramUser, word: str) -> bool:
    return VocabularyItem.objects.filter(
        user=user,
        course_code=get_active_course_code(user),
        normalized_word=clean_word(word),
    ).exists()


def resolve_shared_image_path(
    word: str,
    translation: str,
    preferred_path: str = "",
    course_code: str | None = None,
) -> str:
    if preferred_path:
        return preferred_path

    normalized_word = clean_word(word)
    normalized_translation = (translation or "").strip()
    if not normalized_word or not normalized_translation:
        return ""

    active_course = normalize_course_code(course_code)
    existing = (
        VocabularyItem.objects.filter(
            course_code=active_course,
            normalized_word=normalized_word,
            translation__iexact=normalized_translation,
        )
        .exclude(image_path="")
        .order_by("-updated_at", "-id")
        .first()
    )
    if existing:
        return existing.image_path

    prepared = (
        PackPreparedWord.objects.filter(
            course_code=active_course,
            normalized_word=normalized_word,
            translation__iexact=normalized_translation,
        )
        .exclude(image_path="")
        .order_by("-prepared_at", "-id")
        .first()
    )
    return prepared.image_path if prepared else ""


def create_word(user: TelegramUser, data: dict) -> VocabularyItem:
    course_code = normalize_course_code(data.get("course_code") or get_active_course_code(user))
    word = clean_word(data["word"])
    transcription = data.get("transcription", "") or ""
    if any(char in transcription for char in "абвгдеёжзийклмнопрстуфхцчшщыэюя"):
        transcription = ""

    example_translation = data.get("example_translation") or translate_to_ru(
        data.get("example", "")
    )
    image_path = resolve_shared_image_path(
        word,
        data["translation"],
        data.get("image_path", ""),
        course_code=course_code,
    )
    return VocabularyItem.objects.create(
        user=user,
        course_code=course_code,
        word=word,
        normalized_word=word,
        translation=data["translation"],
        transcription=transcription,
        example=data["example"],
        example_translation=example_translation,
        part_of_speech=data.get("part_of_speech", "unknown"),
        image_regeneration_count=data.get("image_regeneration_count", 0) or 0,
        image_path=image_path,
    )


def add_words_from_text(
    user: TelegramUser, text: str, max_batch_words: int = 10
) -> dict:
    entries = parse_word_batch(text)
    if not entries:
        raise ValueError("Add at least one word.")
    if len(entries) > max_batch_words:
        raise ValueError(f"You can add at most {max_batch_words} words at once.")

    created_items = []
    skipped = []
    failed = []

    for entry in entries:
        if word_already_exists(user, entry.word):
            skipped.append({"word": entry.word, "reason": "duplicate"})
            continue

        word_data = generate_word_data(
            entry.word, translation_hint=entry.translation_hint
        )
        if not word_data:
            failed.append({"word": entry.word, "reason": "generation_failed"})
            continue

        try:
            item = create_word(user, word_data)
            created_items.append(item)
        except Exception:
            logger.exception("Failed to save word %s for user %s", entry.word, user.id)
            failed.append({"word": entry.word, "reason": "save_failed"})

    return {"created": created_items, "skipped": skipped, "failed": failed}


def create_word_drafts_from_text(
    user: TelegramUser, text: str, max_batch_words: int = 10
) -> dict:
    entries = parse_word_batch(text)
    if not entries:
        raise ValueError("Add one word or phrase.")
    if len(entries) > max_batch_words:
        raise ValueError(
            f"За один раз можно добавить максимум {max_batch_words} слов или фраз."
        )

    if len(entries) > 1:
        missing_translation = [
            entry.word for entry in entries if not entry.translation_hint
        ]
        if missing_translation:
            raise ValueError(
                "For multiple lines, provide a translation on every line. Add these one by one: "
                + ", ".join(missing_translation[:5])
            )

        drafts: list[AddWordDraft] = []
        skipped = []
        failed = []

        batch_generated = generate_word_data_batch(
            [
                {"word": entry.word, "translation_hint": entry.translation_hint}
                for entry in entries
            ]
        )

        for entry, generated in zip(entries, batch_generated, strict=False):
            if word_already_exists(user, entry.word):
                skipped.append({"word": entry.word, "reason": "duplicate"})
                continue

            if not generated:
                failed.append({"word": entry.word, "reason": "generation_failed"})
                continue

            try:
                draft = create_word_draft(
                    user, entry.word, generated, translation_hint=entry.translation_hint
                )
                shared_image_path = resolve_shared_image_path(
                    draft.word, draft.translation, "", course_code=draft.course_code
                )
                if shared_image_path:
                    draft.image_path = shared_image_path
                    draft.save(update_fields=["image_path", "updated_at"])
                else:
                    draft = request_draft_image_generation(draft)
                drafts.append(draft)
            except Exception:
                logger.exception("Batch draft save failed for %s", entry.word)
                failed.append({"word": entry.word, "reason": "save_failed"})

        return {
            "mode": "batch_review",
            "drafts": drafts,
            "skipped": skipped,
            "failed": failed,
        }

    entry = entries[0]
    if word_already_exists(user, entry.word):
        raise ValueError("This word already exists.")

    generated = generate_word_data(entry.word, translation_hint=entry.translation_hint)
    if not generated:
        raise ValueError("Could not prepare the word data.")

    shared_image_path = resolve_shared_image_path(
        generated["word"],
        generated.get("translation", ""),
        "",
        course_code=get_active_course_code(user),
    )
    if generated.get("translation") and shared_image_path:
        item = create_word(user, {**generated, "image_path": shared_image_path})
        return {"mode": "auto_saved", "item": item}

    draft = create_word_draft(
        user, entry.word, generated, translation_hint=entry.translation_hint
    )
    step = "confirm_translation"
    if draft.translation_confirmed:
        draft = request_draft_image_generation(draft)
        step = "confirm_image"
    return {"mode": "draft", "draft": draft, "step": step}


def update_word_translation(
    user: TelegramUser, word_id: int, translation: str
) -> VocabularyItem:
    item = VocabularyItem.objects.get(id=word_id, user=user)
    item.translation = translation.strip()
    item.save(update_fields=["translation", "updated_at"])
    return item


def delete_word(user: TelegramUser, word_id: int) -> bool:
    deleted, _ = VocabularyItem.objects.filter(id=word_id, user=user).delete()
    return deleted > 0


def delete_all_words(user: TelegramUser) -> int:
    deleted, _ = VocabularyItem.objects.filter(
        user=user, course_code=get_active_course_code(user)
    ).delete()
    return deleted


def get_available_parts(user: TelegramUser) -> list[str]:
    return list(
        VocabularyItem.objects.filter(
            user=user, course_code=get_active_course_code(user), is_learned=False
        )
        .values_list("part_of_speech", flat=True)
        .distinct()
    )


def get_fake_words(
    exclude_word: str,
    part_of_speech: str | None = None,
    count: int = 3,
    course_code: str | None = None,
) -> list[str]:
    qs = VocabularyItem.objects.filter(
        course_code=normalize_course_code(course_code)
    ).exclude(word__iexact=exclude_word)
    if part_of_speech:
        qs = qs.filter(part_of_speech=part_of_speech)

    words = list(qs.values_list("word", flat=True).distinct().order_by("?")[:count])
    if len(words) < count:
        extras = list(
            VocabularyItem.objects.exclude(word__iexact=exclude_word)
            .values_list("word", flat=True)
            .distinct()
            .order_by("?")[: count - len(words)]
        )
        for candidate in extras:
            if candidate not in words:
                words.append(candidate)
            if len(words) == count:
                break
    return words


def get_user_settings_payload(user: TelegramUser) -> dict:
    return {
        "exercise_goal": get_exercise_goal(user),
        "session_question_limit": get_session_question_limit(user),
        "enable_review_old_words": user.enable_review_old_words,
        "days_before_review": user.days_before_review,
        "reminder_enabled": user.reminder_enabled,
        "reminder_time": user.reminder_time.strftime("%H:%M"),
        "reminder_interval_days": user.reminder_interval_days,
        "reminder_timezone": user.reminder_timezone,
        "active_studied_language": get_active_course_code(user),
        "available_studied_languages": AVAILABLE_STUDIED_LANGUAGES,
    }


def apply_user_settings(user: TelegramUser, payload: dict) -> TelegramUser:
    previous_goal = user.repeat_threshold
    previous_course = get_active_course_code(user)
    exercise_goal = payload.get(
        "exercise_goal", payload.get("repeat_threshold", user.repeat_threshold)
    )
    user.repeat_threshold = max(2, min(int(exercise_goal), 5))
    user.session_question_limit = max(
        1,
        min(
            int(payload.get("session_question_limit", user.session_question_limit)), 50
        ),
    )
    user.enable_review_old_words = bool(
        payload.get("enable_review_old_words", user.enable_review_old_words)
    )
    user.days_before_review = max(
        1, min(int(payload.get("days_before_review", user.days_before_review)), 365)
    )
    user.reminder_enabled = bool(payload.get("reminder_enabled", user.reminder_enabled))
    user.reminder_interval_days = max(
        1,
        min(
            int(payload.get("reminder_interval_days", user.reminder_interval_days)), 30
        ),
    )
    reminder_time = payload.get("reminder_time", user.reminder_time.strftime("%H:%M"))
    user.reminder_time = datetime.strptime(reminder_time, "%H:%M").time()
    user.reminder_timezone = normalize_timezone_value(
        payload.get("reminder_timezone", user.reminder_timezone)
    )
    user.active_studied_language = normalize_course_code(
        payload.get("active_studied_language", user.active_studied_language)
    )
    user.save()
    get_or_create_user_course_progress(user, user.active_studied_language)
    if user.repeat_threshold != previous_goal:
        recalculate_user_word_progress(user)
    elif user.active_studied_language != previous_course:
        get_or_create_user_course_progress(user, user.active_studied_language)
    return user


def set_user_repeat_threshold(user: TelegramUser, value: int) -> TelegramUser:
    user.repeat_threshold = max(2, min(int(value), 5))
    user.save(update_fields=["repeat_threshold"])
    recalculate_user_word_progress(user)
    return user


def save_user(user: TelegramUser) -> TelegramUser:
    user.save()
    return user


def update_user_reminder_time(user: TelegramUser, time_obj) -> TelegramUser:
    user.reminder_time = time_obj
    user.save(update_fields=["reminder_time"])
    return user


def update_user_timezone(user: TelegramUser, tz_value: str) -> TelegramUser:
    user.reminder_timezone = tz_value
    user.save(update_fields=["reminder_timezone"])
    return user


def get_exercise_goal(user: TelegramUser) -> int:
    return max(2, min(int(user.repeat_threshold or 4), 5))


def get_session_question_limit(user: TelegramUser) -> int:
    return max(1, min(int(user.session_question_limit or 12), 50))


def get_required_exercise_types(user: TelegramUser) -> list[str]:
    return EXERCISE_PRIORITY[: get_exercise_goal(user)]


def get_completed_exercise_types(item: VocabularyItem) -> list[str]:
    raw = item.completed_exercise_types or []
    cleaned: list[str] = []
    for exercise_type in raw:
        if exercise_type in EXERCISE_TYPE_LABELS and exercise_type not in cleaned:
            cleaned.append(exercise_type)
    return cleaned


def sync_word_learning_state(item: VocabularyItem) -> VocabularyItem:
    completed = get_completed_exercise_types(item)
    goal = get_exercise_goal(item.user)
    item.completed_exercise_types = completed
    item.correct_count = len(completed)
    item.is_learned = len(completed) >= goal
    if item.is_learned and not item.learned_at:
        item.learned_at = now()
    if not item.is_learned:
        item.learned_at = None
    return item


def get_pending_exercise_types(item: VocabularyItem) -> list[str]:
    completed = set(get_completed_exercise_types(item))
    return [
        exercise_type
        for exercise_type in get_required_exercise_types(item.user)
        if exercise_type not in completed
    ]


def recalculate_user_word_progress(user: TelegramUser) -> None:
    for item in VocabularyItem.objects.filter(
        user=user, course_code=get_active_course_code(user)
    ).iterator():
        sync_word_learning_state(item)
        item.save(
            update_fields=[
                "completed_exercise_types",
                "correct_count",
                "is_learned",
                "learned_at",
                "updated_at",
            ]
        )


def _serialize_pack_level(
    pack_id: str,
    level: dict,
    prepared_map: dict[str, PackPreparedWord],
    existing_user_words: set[str] | None = None,
) -> dict:
    items = []
    for entry in level["items"]:
        normalized = clean_word(entry["word"])
        prepared = prepared_map.get(normalized)
        items.append(
            {
                "word": entry["word"],
                "translation": entry["translation"],
                "normalized_word": normalized,
                "has_prepared_image": bool(prepared and prepared.image_path),
                "prepared": bool(
                    prepared and prepared.example and prepared.transcription
                ),
                "image_generation_in_progress": bool(
                    prepared and prepared.image_generation_in_progress
                ),
                "already_added": normalized in (existing_user_words or set()),
            }
        )
    prepared_count = sum(1 for item in items if item["prepared"])
    return {
        "id": level["id"],
        "title": level["title"],
        "description": level["description"],
        "size": len(level["items"]),
        "prepared_count": prepared_count,
        "items": items,
    }


def ensure_pack_placeholders(
    pack_id: str, level_id: str, course_code: str | None = None
) -> None:
    active_course = normalize_course_code(course_code)
    level = get_pack_level(pack_id, level_id)
    if not level:
        return
    for entry in level["items"]:
        normalized = clean_word(entry["word"])
        PackPreparedWord.objects.get_or_create(
            course_code=active_course,
            pack_id=pack_id,
            level_id=level_id,
            normalized_word=normalized,
            defaults={
                "word": entry["word"],
                "translation": entry["translation"],
            },
        )


def list_word_packs(
    user: TelegramUser | None = None, course_code: str | None = None
) -> list[dict]:
    active_course = normalize_course_code(
        course_code or (get_active_course_code(user) if user else None)
    )
    definitions = get_course_pack_definitions(active_course)
    for pack in definitions:
        for level in pack["levels"]:
            ensure_pack_placeholders(pack["id"], level["id"], course_code=active_course)
    prepared_items = PackPreparedWord.objects.filter(
        course_code=active_course,
        pack_id__in=[pack["id"] for pack in definitions]
    )
    prepared_map: dict[tuple[str, str], dict[str, PackPreparedWord]] = {}
    for prepared in prepared_items:
        prepared_map.setdefault((prepared.pack_id, prepared.level_id), {})[
            prepared.normalized_word
        ] = prepared

    existing_user_words = (
        set(
            VocabularyItem.objects.filter(user=user).values_list(
                "normalized_word", flat=True
            )
        )
        if user
        else set()
    )
    if user:
        existing_user_words = set(
            VocabularyItem.objects.filter(
                user=user, course_code=active_course
            ).values_list("normalized_word", flat=True)
        )
    packs = []
    for pack in definitions:
        levels = [
            _serialize_pack_level(
                pack["id"],
                level,
                prepared_map.get((pack["id"], level["id"]), {}),
                existing_user_words=existing_user_words,
            )
            for level in pack["levels"]
        ]
        packs.append(
            {
                "id": pack["id"],
                "title": pack["title"],
                "emoji": pack["emoji"],
                "description": pack["description"],
                "levels": levels,
            }
        )
    return packs


def prepare_all_word_packs() -> None:
    for pack in get_course_pack_definitions("en"):
        for level in pack["levels"]:
            _prepare_pack_level_sync(pack["id"], level["id"], course_code="en")


def prepare_next_pack_word() -> PackPreparedWord | None:
    for pack in get_course_pack_definitions("en"):
        for level in pack["levels"]:
            ensure_pack_placeholders(pack["id"], level["id"], course_code="en")

    candidate = (
        PackPreparedWord.objects.filter(
            course_code="en", image_generation_in_progress=False
        )
        .filter(Q(example="") | Q(transcription="") | Q(image_path=""))
        .order_by("pack_id", "level_id", "prepared_at", "id")
        .first()
    )
    if not candidate:
        return None

    candidate.image_generation_in_progress = True
    candidate.save(update_fields=["image_generation_in_progress", "prepared_at"])

    try:
        if not candidate.example or not candidate.transcription:
            generated = generate_word_data(
                candidate.word,
                translation_hint=candidate.translation,
            )
            if generated:
                candidate.word = generated["word"]
                candidate.translation = (
                    generated.get("translation") or candidate.translation
                )
                candidate.transcription = generated.get("transcription", "") or ""
                candidate.example = generated.get("example", "") or ""
                candidate.example_translation = generated.get(
                    "example_translation"
                ) or translate_to_ru(candidate.example)
                candidate.part_of_speech = (
                    generated.get("part_of_speech", "unknown") or "unknown"
                )

        if candidate.example and candidate.translation and not candidate.image_path:
            built = _build_item_image(
                candidate.word,
                candidate.translation,
                candidate.part_of_speech,
                candidate.example,
                f"pack_{candidate.pack_id}_{candidate.level_id}_{candidate.normalized_word}_{int(time.time())}",
            )
            if built:
                _, image_path = built
                candidate.image_path = image_path
        candidate.image_generation_in_progress = False
        candidate.save(
            update_fields=[
                "word",
                "translation",
                "transcription",
                "example",
                "example_translation",
                "part_of_speech",
                "image_path",
                "image_generation_in_progress",
                "prepared_at",
            ]
        )
        return candidate
    except Exception:
        candidate.image_generation_in_progress = False
        candidate.save(update_fields=["image_generation_in_progress", "prepared_at"])
        raise


def _prepare_pack_level_sync(
    pack_id: str, level_id: str, course_code: str | None = None
) -> None:
    active_course = normalize_course_code(course_code)
    level = get_pack_level(pack_id, level_id)
    if not level:
        return

    existing = {
        item.normalized_word: item
        for item in PackPreparedWord.objects.filter(
            course_code=active_course, pack_id=pack_id, level_id=level_id
        )
    }
    pending_entries = []
    for entry in level["items"]:
        normalized = clean_word(entry["word"])
        cached = existing.get(normalized)
        if cached and cached.example and cached.transcription and cached.image_path:
            continue
        pending_entries.append(
            {
                "word": entry["word"],
                "translation_hint": entry["translation"],
            }
        )

    if not pending_entries:
        return

    generated_items = generate_word_data_batch(pending_entries)
    for entry, generated in zip(pending_entries, generated_items, strict=False):
        normalized = clean_word(entry["word"])
        cached = existing.get(normalized)
        if cached is None:
            cached, _ = PackPreparedWord.objects.get_or_create(
                course_code=active_course,
                pack_id=pack_id,
                level_id=level_id,
                normalized_word=normalized,
                defaults={
                    "word": entry["word"],
                    "translation": entry["translation_hint"],
                },
            )
            existing[normalized] = cached

        if generated:
            cached.word = generated["word"]
            cached.translation = (
                generated.get("translation") or entry["translation_hint"]
            )
            cached.transcription = generated.get("transcription", "") or ""
            cached.example = generated.get("example", "") or ""
            cached.example_translation = generated.get(
                "example_translation"
            ) or translate_to_ru(cached.example)
            cached.part_of_speech = (
                generated.get("part_of_speech", "unknown") or "unknown"
            )

        cached.image_generation_in_progress = True
        cached.save(
            update_fields=[
                "word",
                "translation",
                "transcription",
                "example",
                "example_translation",
                "part_of_speech",
                "image_generation_in_progress",
                "prepared_at",
            ]
        )
        built = None
        if cached.example and cached.translation:
            built = _build_item_image(
                cached.word,
                cached.translation,
                cached.part_of_speech,
                cached.example,
                f"pack_{pack_id}_{level_id}_{normalized}_{int(time.time())}",
            )
        if built:
            _, image_path = built
            cached.image_path = image_path
        cached.image_generation_in_progress = False
        cached.save(
            update_fields=["image_path", "image_generation_in_progress", "prepared_at"]
        )


def ensure_pack_preparation(
    pack_id: str, level_id: str, course_code: str | None = None
) -> None:
    active_course = normalize_course_code(course_code)
    key = f"{active_course}:{pack_id}:{level_id}"
    with _PACK_PREPARATION_LOCK:
        if key in _PACK_PREPARATION_IN_FLIGHT:
            return
        _PACK_PREPARATION_IN_FLIGHT.add(key)

    def _run() -> None:
        try:
            _prepare_pack_level_sync(pack_id, level_id, course_code=active_course)
        except Exception:
            logger.exception("Pack preparation failed for %s", key)
        finally:
            with _PACK_PREPARATION_LOCK:
                _PACK_PREPARATION_IN_FLIGHT.discard(key)

    Thread(target=_run, daemon=True).start()


def add_pack_words_to_user(
    user: TelegramUser, pack_id: str, level_id: str, selected_words: list[str]
) -> dict:
    active_course = get_active_course_code(user)
    level = get_pack_level(pack_id, level_id)
    if not level:
        raise ValueError("Pack level not found.")

    allowed = {clean_word(item["word"]): item for item in level["items"]}
    normalized_selected = [
        clean_word(word) for word in selected_words if clean_word(word) in allowed
    ]
    if not normalized_selected:
        raise ValueError("Choose at least one word from the pack.")

    prepared_map = {
        item.normalized_word: item
        for item in PackPreparedWord.objects.filter(
            course_code=active_course,
            pack_id=pack_id,
            level_id=level_id,
            normalized_word__in=normalized_selected,
        )
    }

    created = []
    skipped = []
    fallback_entries = []

    for normalized in normalized_selected:
        entry = allowed[normalized]
        if word_already_exists(user, entry["word"]):
            skipped.append({"word": entry["word"], "reason": "duplicate"})
            continue
        prepared = prepared_map.get(normalized)
        if prepared and prepared.example and prepared.transcription:
            item = create_word(
                user,
                {
                    "word": prepared.word,
                    "translation": prepared.translation,
                    "transcription": prepared.transcription,
                    "example": prepared.example,
                    "example_translation": prepared.example_translation,
                    "part_of_speech": prepared.part_of_speech,
                    "image_path": prepared.image_path,
                },
            )
            if not item.image_path:
                request_word_image_generation(item)
            created.append(serialize_word(item))
        else:
            fallback_entries.append(
                {"word": entry["word"], "translation_hint": entry["translation"]}
            )

    if fallback_entries:
        generated_batch = generate_word_data_batch(fallback_entries)
        for entry, generated in zip(fallback_entries, generated_batch, strict=False):
            if not generated:
                generated = {
                    "word": entry["word"],
                    "translation": entry["translation_hint"],
                    "transcription": "",
                    "example": "",
                    "example_translation": "",
                    "part_of_speech": "unknown",
                }
            item = create_word(
                user,
                {
                    "word": generated["word"],
                    "translation": generated.get("translation")
                    or entry["translation_hint"],
                    "transcription": generated.get("transcription", "") or "",
                    "example": generated.get("example", "") or "",
                    "example_translation": generated.get("example_translation", "")
                    or "",
                    "part_of_speech": generated.get("part_of_speech", "unknown")
                    or "unknown",
                    "image_path": resolve_shared_image_path(
                        entry["word"],
                        generated.get("translation") or entry["translation_hint"],
                        "",
                        course_code=active_course,
                    ),
                },
            )
            if not item.image_path:
                request_word_image_generation(item)
            created.append(serialize_word(item))

    ensure_pack_preparation(pack_id, level_id)
    return {"created": created, "skipped": skipped}


def serialize_draft(draft: AddWordDraft) -> dict:
    image_file = get_draft_image_file(draft)
    return {
        "id": draft.id,
        "source_text": draft.source_text,
        "word": draft.word,
        "translation": draft.translation,
        "translation_confirmed": draft.translation_confirmed,
        "transcription": draft.transcription,
        "example": draft.example,
        "example_translation": draft.example_translation,
        "part_of_speech": draft.part_of_speech,
        "course_code": draft.course_code,
        "image_regeneration_count": draft.image_regeneration_count,
        "image_generation_in_progress": draft.image_generation_in_progress,
        "image_prompt": draft.image_prompt,
        "image_path": draft.image_path,
        "has_image": image_file is not None,
        "created_at": draft.created_at.isoformat(),
        "updated_at": draft.updated_at.isoformat(),
    }


def get_draft_image_file(draft: AddWordDraft) -> Path | None:
    if not draft.image_path:
        return None
    raw_path = Path(draft.image_path)
    candidate = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    draft_root = (PROJECT_ROOT / "media" / "draft_images").resolve()
    allowed_roots = (
        draft_root,
        IMAGE_CACHE_DIR.resolve(),
        USER_IMAGE_DIR.resolve(),
    )
    if any(resolved.is_relative_to(root) for root in allowed_roots):
        return _preferred_served_image(resolved)
    return None


def create_word_draft(
    user: TelegramUser,
    source_text: str,
    generated: dict,
    translation_hint: str | None = None,
) -> AddWordDraft:
    translation = (translation_hint or generated.get("translation") or "").strip()
    return AddWordDraft.objects.create(
        user=user,
        course_code=normalize_course_code(
            generated.get("course_code") or get_active_course_code(user)
        ),
        source_text=source_text,
        word=generated["word"],
        normalized_word=clean_word(generated["word"]),
        translation=translation,
        translation_confirmed=bool(translation_hint),
        transcription=generated.get("transcription", "") or "",
        example=generated.get("example", "") or "",
        example_translation=generated.get("example_translation")
        or translate_to_ru(generated.get("example", "")),
        part_of_speech=generated.get("part_of_speech", "unknown"),
    )


def refresh_draft_language_data(draft: AddWordDraft, translation: str) -> AddWordDraft:
    generated = generate_word_data(
        draft.word, part_hint=draft.part_of_speech, translation_hint=translation
    )
    if generated:
        draft.translation = translation
        draft.translation_confirmed = True
        draft.transcription = generated.get("transcription", "") or draft.transcription
        draft.example = generated.get("example", "") or draft.example
        draft.example_translation = generated.get(
            "example_translation"
        ) or translate_to_ru(draft.example)
        draft.part_of_speech = (
            generated.get("part_of_speech", draft.part_of_speech)
            or draft.part_of_speech
        )
    else:
        draft.translation = translation
        draft.translation_confirmed = True
    draft.save(
        update_fields=[
            "translation",
            "translation_confirmed",
            "transcription",
            "example",
            "example_translation",
            "part_of_speech",
            "image_path",
            "image_prompt",
            "image_generation_version",
            "image_generation_in_progress",
            "updated_at",
        ]
    )
    return draft


def _build_item_image(
    word: str, translation: str, part_of_speech: str, example: str, slug: str
) -> tuple[str, str] | None:
    visual_prompt = build_visual_prompt(word, translation, part_of_speech, example)
    if not visual_prompt:
        return None
    image_path = generate_card_image(visual_prompt, slug)
    return visual_prompt, image_path


def _run_draft_image_generation(draft_id: int, version: int) -> None:
    try:
        draft = AddWordDraft.objects.get(id=draft_id)
    except AddWordDraft.DoesNotExist:
        return

    try:
        built = _build_item_image(
            draft.word,
            draft.translation,
            draft.part_of_speech,
            draft.example,
            f"{draft.user_id}_{draft.normalized_word}_{int(time.time())}",
        )
        try:
            latest = AddWordDraft.objects.get(id=draft_id)
        except AddWordDraft.DoesNotExist:
            return
        if latest.image_generation_version != version:
            return
        if not built:
            latest.image_generation_in_progress = False
            latest.save(update_fields=["image_generation_in_progress", "updated_at"])
            return
        visual_prompt, image_path = built
        latest.image_prompt = visual_prompt
        latest.image_path = image_path
        latest.image_generation_in_progress = False
        latest.save(
            update_fields=[
                "image_prompt",
                "image_path",
                "image_generation_in_progress",
                "updated_at",
            ]
        )
    except Exception:
        logger.exception("Draft image generation failed for draft %s", draft_id)
        AddWordDraft.objects.filter(
            id=draft_id, image_generation_version=version
        ).update(image_generation_in_progress=False)


def request_draft_image_generation(
    draft: AddWordDraft, force_regenerate: bool = False
) -> AddWordDraft:
    reused_path = (
        ""
        if force_regenerate
        else resolve_shared_image_path(draft.word, draft.translation, "")
    )
    if reused_path:
        draft.image_path = reused_path
        draft.image_prompt = ""
        draft.image_generation_in_progress = False
        draft.save(
            update_fields=[
                "image_path",
                "image_prompt",
                "image_generation_in_progress",
                "updated_at",
            ]
        )
        return draft

    draft.image_path = ""
    draft.image_prompt = ""
    draft.image_generation_version += 1
    draft.image_generation_in_progress = True
    if force_regenerate:
        draft.image_regeneration_count += 1
    draft.save(
        update_fields=[
            "image_path",
            "image_prompt",
            "image_generation_version",
            "image_generation_in_progress",
            "image_regeneration_count",
            "updated_at",
        ]
    )
    Thread(
        target=_run_draft_image_generation,
        args=(draft.id, draft.image_generation_version),
        daemon=True,
    ).start()
    return draft


def _run_word_image_generation(item_id: int, version: int) -> None:
    try:
        item = VocabularyItem.objects.get(id=item_id)
    except VocabularyItem.DoesNotExist:
        return

    try:
        built = _build_item_image(
            item.word,
            item.translation,
            item.part_of_speech,
            item.example,
            f"word_{item.user_id}_{item.normalized_word}_{int(time.time())}",
        )
        try:
            latest = VocabularyItem.objects.get(id=item_id)
        except VocabularyItem.DoesNotExist:
            return
        if latest.image_generation_version != version:
            return
        if not built:
            latest.image_generation_in_progress = False
            latest.save(update_fields=["image_generation_in_progress", "updated_at"])
            return
        _, image_path = built
        latest.image_path = image_path
        latest.image_generation_in_progress = False
        latest.save(
            update_fields=["image_path", "image_generation_in_progress", "updated_at"]
        )
    except Exception:
        logger.exception("Word image generation failed for item %s", item_id)
        VocabularyItem.objects.filter(
            id=item_id, image_generation_version=version
        ).update(image_generation_in_progress=False)


def request_word_image_generation(
    item: VocabularyItem, force_regenerate: bool = False
) -> VocabularyItem:
    reused_path = (
        ""
        if force_regenerate
        else resolve_shared_image_path(item.word, item.translation, "")
    )
    if reused_path:
        item.image_path = reused_path
        item.image_generation_in_progress = False
        item.save(
            update_fields=["image_path", "image_generation_in_progress", "updated_at"]
        )
        return item

    item.image_generation_version += 1
    item.image_generation_in_progress = True
    if force_regenerate:
        item.image_regeneration_count += 1
    item.save(
        update_fields=[
            "image_generation_version",
            "image_generation_in_progress",
            "image_regeneration_count",
            "updated_at",
        ]
    )
    Thread(
        target=_run_word_image_generation,
        args=(item.id, item.image_generation_version),
        daemon=True,
    ).start()
    return item


def ensure_draft_image(
    draft: AddWordDraft, force_regenerate: bool = False
) -> AddWordDraft:
    if not force_regenerate:
        reused_path = resolve_shared_image_path(draft.word, draft.translation, "")
        if reused_path:
            draft.image_path = reused_path
            draft.image_prompt = ""
            draft.save(update_fields=["image_path", "image_prompt", "updated_at"])
            return draft
        if draft.image_path and get_draft_image_file(draft):
            return draft

    visual_prompt = build_visual_prompt(
        draft.word,
        draft.translation,
        draft.part_of_speech,
        draft.example,
    )
    if not visual_prompt:
        return draft

    slug = f"{draft.user_id}_{draft.normalized_word}_{int(time.time())}"
    image_path = generate_card_image(visual_prompt, slug)
    draft.image_prompt = visual_prompt
    draft.image_path = image_path
    if force_regenerate:
        draft.image_regeneration_count += 1
    draft.save(
        update_fields=[
            "image_prompt",
            "image_path",
            "image_regeneration_count",
            "updated_at",
        ]
    )
    return draft


def finalize_word_draft(draft: AddWordDraft, use_image: bool = True) -> VocabularyItem:
    payload = {
        "course_code": draft.course_code,
        "word": draft.word,
        "translation": draft.translation,
        "transcription": draft.transcription,
        "example": draft.example,
        "example_translation": draft.example_translation,
        "part_of_speech": draft.part_of_speech,
        "image_regeneration_count": draft.image_regeneration_count,
        "image_path": draft.image_path if use_image else "",
    }
    item = create_word(draft.user, payload)
    if use_image and not item.image_path:
        item = request_word_image_generation(item)
    draft.delete()
    return item


def regenerate_word_image(item: VocabularyItem) -> VocabularyItem:
    item.image_regeneration_count += 1
    visual_prompt = build_visual_prompt(
        item.word,
        item.translation,
        item.part_of_speech,
        item.example,
    )
    if not visual_prompt:
        return item

    slug = f"word_{item.user_id}_{item.normalized_word}_{int(time.time())}"
    item.image_path = generate_card_image(visual_prompt, slug)
    item.save(update_fields=["image_path", "image_regeneration_count", "updated_at"])
    return item


def get_ordered_unlearned_words(
    user: TelegramUser,
    count: int = 10,
    exclude_ids: Iterable[int] | None = None,
) -> list[VocabularyItem]:
    exclude_ids = list(exclude_ids or [])
    return list(
        VocabularyItem.objects.filter(
            user=user, course_code=get_active_course_code(user), is_learned=False
        )
        .exclude(id__in=exclude_ids)
        .order_by("created_at", "id")[:count]
    )


def get_unlearned_words(
    user: TelegramUser, count: int = 10, part_of_speech: str | None = None
) -> list[VocabularyItem]:
    active_course = get_active_course_code(user)
    base_qs = VocabularyItem.objects.filter(
        user=user, course_code=active_course, is_learned=False
    )
    if part_of_speech:
        base_qs = base_qs.filter(part_of_speech=part_of_speech)
    base_ids = list(base_qs.values_list("id", flat=True))

    review_ids: list[int] = []
    if user.enable_review_old_words:
        threshold = now() - timedelta(days=user.days_before_review)
        review_qs = VocabularyItem.objects.filter(
            user=user,
            course_code=active_course,
            is_learned=True,
            updated_at__lt=threshold,
        )
        if part_of_speech:
            review_qs = review_qs.filter(part_of_speech=part_of_speech)
        review_ids = list(review_qs.values_list("id", flat=True))

    all_ids = base_ids + review_ids
    if not all_ids:
        return []
    selected_ids = random.sample(all_ids, min(len(all_ids), count))
    return list(VocabularyItem.objects.filter(id__in=selected_ids))


def get_learned_words(user: TelegramUser) -> list[VocabularyItem]:
    return list(
        VocabularyItem.objects.filter(
            user=user, course_code=get_active_course_code(user), is_learned=True
        ).order_by(
            "updated_at", "id"
        )
    )


def update_word_progress(
    item_id: int, correct: bool, exercise_type: str | None = None
) -> VocabularyItem:
    item = VocabularyItem.objects.select_related("user").get(id=item_id)
    completed = get_completed_exercise_types(item)
    if (
        correct
        and exercise_type
        and exercise_type in EXERCISE_TYPE_LABELS
        and exercise_type not in completed
    ):
        completed.append(exercise_type)
        item.completed_exercise_types = completed
    sync_word_learning_state(item)
    item.save(
        update_fields=[
            "completed_exercise_types",
            "correct_count",
            "is_learned",
            "learned_at",
            "updated_at",
        ]
    )
    return item


def increment_user_metric(user: TelegramUser, field_name: str) -> TelegramUser:
    progress = get_or_create_user_course_progress(user)
    current_value = getattr(progress, field_name, 0) or 0
    setattr(progress, field_name, current_value + 1)
    progress.save(update_fields=[field_name])
    return user


def reset_word_progress(item_id: int) -> VocabularyItem:
    item = VocabularyItem.objects.get(id=item_id)
    item.is_learned = False
    item.learned_at = None
    item.correct_count = 0
    item.completed_exercise_types = []
    item.save(
        update_fields=[
            "is_learned",
            "learned_at",
            "correct_count",
            "completed_exercise_types",
            "updated_at",
        ]
    )
    return item


def get_fake_translations(
    user: TelegramUser,
    exclude_word: str,
    part_of_speech: str | None = None,
    count: int = 3,
) -> list[str]:
    active_course = get_active_course_code(user)
    qs = VocabularyItem.objects.filter(course_code=active_course).exclude(
        word__iexact=exclude_word
    )
    if part_of_speech:
        qs = qs.filter(part_of_speech=part_of_speech)

    translations = list(
        qs.values_list("translation", flat=True).distinct().order_by("?")[:count]
    )
    if len(translations) < count:
        extras = list(
            VocabularyItem.objects.exclude(word__iexact=exclude_word)
            .filter(course_code=active_course)
            .values_list("translation", flat=True)
            .distinct()
            .order_by("?")[: count - len(translations)]
        )
        for candidate in extras:
            if candidate not in translations:
                translations.append(candidate)
            if len(translations) == count:
                break
    return translations


def get_learning_candidates(
    user: TelegramUser, exclude_ids: Iterable[int] | None = None
) -> list[VocabularyItem]:
    qs = VocabularyItem.objects.filter(
        user=user, course_code=get_active_course_code(user), is_learned=False
    ).exclude(id__in=list(exclude_ids or []))
    items = list(qs)
    random.shuffle(items)
    return [item for item in items if get_pending_exercise_types(item)]


def _build_choice_options(
    item: VocabularyItem, answer_mode: str
) -> tuple[str, list[str]]:
    if answer_mode == "practice_ru_en":
        correct_answer = item.word
        fake_options = list(
            VocabularyItem.objects.filter(course_code=item.course_code).exclude(id=item.id)
            .values_list("word", flat=True)
            .distinct()
            .order_by("?")[:3]
        )
    else:
        correct_answer = item.translation
        fake_options = get_fake_translations(
            item.user,
            exclude_word=item.word,
            part_of_speech=item.part_of_speech,
            count=3,
        )
    options = list(dict.fromkeys(fake_options + [correct_answer]))
    random.shuffle(options)
    return correct_answer, options


def build_learning_question(
    user: TelegramUser, exclude_ids: Iterable[int] | None = None
) -> dict | None:
    candidates = get_learning_candidates(user, exclude_ids=exclude_ids)
    if not candidates:
        return None

    item = candidates[0]
    pending_types = get_pending_exercise_types(item)
    if not pending_types:
        return None

    exercise_type = random.choice(pending_types)
    payload = {
        "exercise_type": exercise_type,
        "exercise_label": EXERCISE_TYPE_LABELS[exercise_type],
        "item": serialize_word(item),
    }

    if exercise_type in {"practice_en_ru", "practice_ru_en"}:
        correct_answer, options = _build_choice_options(item, exercise_type)
        payload.update(
            {
                "kind": "choice",
                "prompt": (
                    "Выбери правильный перевод"
                    if exercise_type == "practice_en_ru"
                    else "Выбери правильное английское слово"
                ),
                "answer_mode": exercise_type,
                "options": options,
                "correct_answer": correct_answer,
            }
        )
        return payload

    if exercise_type in {"listening_word", "listening_translate"}:
        payload.update(
            {
                "kind": "listening",
                "prompt": (
                    "Напиши услышанное слово"
                    if exercise_type == "listening_word"
                    else "Напиши перевод услышанного слова"
                ),
                "answer_mode": exercise_type,
            }
        )
        return payload

    payload.update({"kind": "speaking", "prompt": "Прослушай пример и произнеси слово"})
    return payload


def build_choice_question(user: TelegramUser, mode: str) -> dict | None:
    if mode == "review":
        candidates = get_learned_words(user)
        if not candidates:
            return None
        item = candidates[0]
        correct_answer = item.translation
        prompt = "Выбери правильный перевод старого слова"
    elif mode == "reverse":
        candidates = get_unlearned_words(user, count=10)
        if not candidates:
            return None
        item = candidates[0]
        correct_answer = item.word
        prompt = "Выбери правильное английское слово"
    else:
        candidates = get_unlearned_words(user, count=10)
        if not candidates:
            return None
        item = candidates[0]
        correct_answer = item.translation
        prompt = "Выбери правильный перевод"

    if mode == "reverse":
        fakes = list(
            VocabularyItem.objects.filter(course_code=item.course_code).exclude(id=item.id)
            .values_list("word", flat=True)
            .distinct()
            .order_by("?")[:3]
        )
    else:
        fakes = get_fake_translations(
            user, exclude_word=item.word, part_of_speech=item.part_of_speech, count=3
        )

    options = list(dict.fromkeys(fakes + [correct_answer]))
    random.shuffle(options)
    return {
        "item": serialize_word(item),
        "options": options,
        "mode": mode,
        "prompt": prompt,
    }


def submit_choice_answer(
    user: TelegramUser, item_id: int, answer: str, mode: str
) -> dict:
    item = VocabularyItem.objects.get(
        id=item_id, user=user, course_code=get_active_course_code(user)
    )
    normalized = answer.strip().lower()

    if mode == "reverse":
        correct = normalized == item.word.lower()
        updated = update_word_progress(
            item.id, correct=correct, exercise_type="practice_ru_en"
        )
        correct_answer = item.word
        if correct:
            increment_user_metric(user, "practice_correct")
    elif mode == "review":
        correct = is_translation_answer_correct(answer, item.translation)
        updated = (
            update_word_progress(item.id, correct=True)
            if correct
            else reset_word_progress(item.id)
        )
        correct_answer = item.translation
        if correct:
            increment_user_metric(user, "review_correct")
    else:
        correct = is_translation_answer_correct(answer, item.translation)
        updated = update_word_progress(
            item.id, correct=correct, exercise_type="practice_en_ru"
        )
        correct_answer = item.translation
        if correct:
            increment_user_metric(user, "practice_correct")

    update_learning_streak(user)
    return {
        "correct": correct,
        "item": serialize_word(updated),
        "correct_answer": correct_answer,
        "progress": build_user_progress(user),
    }


def build_listening_question(user: TelegramUser, mode: str) -> dict | None:
    candidates = get_unlearned_words(user, count=10)
    if not candidates:
        return None
    item = candidates[0]
    prompt = (
        "Напиши услышанное слово"
        if mode == "word"
        else "Напиши перевод услышанного слова"
    )
    return {"item": serialize_word(item), "mode": mode, "prompt": prompt}


def submit_listening_answer(
    user: TelegramUser, item_id: int, answer: str, mode: str
) -> dict:
    item = VocabularyItem.objects.get(
        id=item_id, user=user, course_code=get_active_course_code(user)
    )
    expected = item.word if mode == "word" else item.translation
    accepted_with_typo = False
    if mode == "word":
        correct = answer.strip().lower() == expected.lower()
        if not correct and is_single_typo_match(answer, expected):
            correct = True
            accepted_with_typo = True
    else:
        correct = is_translation_answer_correct(answer, expected)
    updated = update_word_progress(
        item.id,
        correct=correct,
        exercise_type="listening_word" if mode == "word" else "listening_translate",
    )
    if correct:
        increment_user_metric(user, "listening_correct")
    update_learning_streak(user)
    return {
        "correct": correct,
        "item": serialize_word(updated),
        "correct_answer": expected,
        "accepted_with_typo": accepted_with_typo,
        "progress": build_user_progress(user),
    }


def build_speaking_question(user: TelegramUser) -> dict | None:
    candidates = get_unlearned_words(user, count=10)
    if not candidates:
        return None
    item = candidates[0]
    return {"item": serialize_word(item), "prompt": "Прослушай и произнеси слово"}


def evaluate_speaking_answer(user: TelegramUser, item_id: int, transcript: str) -> dict:
    item = VocabularyItem.objects.get(
        id=item_id, user=user, course_code=get_active_course_code(user)
    )
    expected = clean_word(item.word)
    spoken = clean_word(transcript)
    similarity = SequenceMatcher(None, spoken, expected).ratio() if spoken else 0.0

    if spoken == expected:
        updated = update_word_progress(item.id, correct=True, exercise_type="speaking")
        increment_user_metric(user, "speaking_correct")
        update_learning_streak(user)
        return {
            "status": "correct",
            "message": "Отлично! Произношение принято.",
            "similarity": 1.0,
            "transcript": transcript,
            "item": serialize_word(updated),
            "correct_answer": item.word,
            "progress": build_user_progress(user),
        }

    if similarity >= 0.6:
        return {
            "status": "close",
            "message": "Неплохо, но попробуй ещё раз.",
            "similarity": round(similarity, 2),
            "transcript": transcript,
            "item": serialize_word(item),
            "correct_answer": item.word,
            "progress": build_user_progress(user),
        }

    return {
        "status": "wrong",
        "message": "Пока не совпало. Попробуй ещё раз.",
        "similarity": round(similarity, 2),
        "transcript": transcript,
        "item": serialize_word(item),
        "correct_answer": item.word,
        "progress": build_user_progress(user),
    }


def submit_learning_text_answer(
    user: TelegramUser, item_id: int, answer: str, exercise_type: str
) -> dict:
    item = VocabularyItem.objects.get(
        id=item_id, user=user, course_code=get_active_course_code(user)
    )
    normalized = answer.strip().lower()
    accepted_with_typo = False
    if exercise_type == "practice_en_ru":
        expected = item.translation
        correct = is_translation_answer_correct(answer, item.translation)
        updated = update_word_progress(
            item.id, correct=correct, exercise_type=exercise_type
        )
        if correct:
            increment_user_metric(user, "practice_correct")
    elif exercise_type == "practice_ru_en":
        expected = item.word
        correct = normalized == item.word.lower()
        if not correct and is_single_typo_match(answer, item.word):
            correct = True
            accepted_with_typo = True
        updated = update_word_progress(
            item.id, correct=correct, exercise_type=exercise_type
        )
        if correct:
            increment_user_metric(user, "practice_correct")
    elif exercise_type == "listening_word":
        expected = item.word
        correct = normalized == item.word.lower()
        if not correct and is_single_typo_match(answer, item.word):
            correct = True
            accepted_with_typo = True
        updated = update_word_progress(
            item.id, correct=correct, exercise_type=exercise_type
        )
        if correct:
            increment_user_metric(user, "listening_correct")
    elif exercise_type == "listening_translate":
        expected = item.translation
        correct = is_translation_answer_correct(answer, item.translation)
        updated = update_word_progress(
            item.id, correct=correct, exercise_type=exercise_type
        )
        if correct:
            increment_user_metric(user, "listening_correct")
    else:
        raise ValueError("Unsupported exercise type.")

    if correct:
        update_learning_streak(user)

    return {
        "correct": correct,
        "item": serialize_word(updated),
        "correct_answer": expected,
        "accepted_with_typo": accepted_with_typo,
        "progress": build_user_progress(user),
        "exercise_type": exercise_type,
    }


def list_irregular_page(page: int, per_page: int = 20) -> dict:
    start = page * per_page
    end = start + per_page
    return {
        "items": IRREGULAR_VERBS[start:end],
        "page": page,
        "has_prev": page > 0,
        "has_next": end < len(IRREGULAR_VERBS),
        "total": len(IRREGULAR_VERBS),
    }


def build_irregular_question() -> dict:
    verb = random.choice(IRREGULAR_VERBS)
    correct_pair = f"{verb['past']} {verb['participle']}"
    options = []
    for candidate in [correct_pair] + verb["wrong_pairs"] + get_random_pairs(verb, 2):
        if candidate not in options:
            options.append(candidate)
    random.shuffle(options)
    return {"verb": verb, "correct_pair": correct_pair, "options": options[:4]}


def update_irregular_progress(
    user: TelegramUser, base: str, correct: bool
) -> IrregularVerbProgress:
    active_course = get_active_course_code(user)
    progress, _ = IrregularVerbProgress.objects.get_or_create(
        user=user, course_code=active_course, verb_base=base
    )
    if correct:
        progress.correct_count += 1
        if not progress.is_learned and progress.correct_count >= 5:
            progress.is_learned = True
    progress.save()
    if correct:
        course_progress = get_or_create_user_course_progress(user, active_course)
        course_progress.irregular_correct += 1
        course_progress.save(update_fields=["irregular_correct"])
    return progress


def create_web_login_token() -> WebLoginToken:
    return WebLoginToken.objects.create(
        expires_at=timezone.now() + timedelta(minutes=15)
    )


def bind_web_login_token(token: str, user: TelegramUser) -> WebLoginToken | None:
    try:
        login_token = WebLoginToken.objects.get(
            token=token, expires_at__gt=timezone.now(), consumed_at__isnull=True
        )
    except WebLoginToken.DoesNotExist:
        return None
    login_token.user = user
    login_token.save(update_fields=["user"])
    return login_token


def consume_web_login_token(token: str) -> TelegramUser | None:
    try:
        login_token = WebLoginToken.objects.select_related("user").get(
            token=token,
            expires_at__gt=timezone.now(),
            consumed_at__isnull=True,
        )
    except WebLoginToken.DoesNotExist:
        return None

    if login_token.user is None:
        return None

    login_token.consumed_at = timezone.now()
    login_token.save(update_fields=["consumed_at"])
    return login_token.user


def parse_word_batch(text: str) -> list[ParsedWordEntry]:
    entries: list[ParsedWordEntry] = []

    def normalize_word_part(raw: str) -> str:
        value = (raw or "").strip()
        value = re.sub(r"^[•*·\-]+\s*", "", value)
        value = re.sub(r"\s*:\s*$", "", value)
        value = re.sub(
            r"\s*\((?:v|adj|adv|n|noun|verb|adjective|adverb|phrase)\)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\s+(?:v|adj|adv|n)\s*$", "", value, flags=re.IGNORECASE)
        return value.strip()

    def split_word_and_translation(line: str) -> tuple[str, str | None]:
        for separator in (" - ", " — ", " – "):
            if separator in line:
                word_part, translation_hint = line.split(separator, 1)
                return normalize_word_part(word_part), translation_hint.strip() or None

        if ":" in line:
            word_part, translation_hint = line.split(":", 1)
            normalized = normalize_word_part(word_part)
            if normalized and translation_hint.strip():
                return normalized, translation_hint.strip()

        cyrillic_match = re.search(r"[А-Яа-яЁё]", line)
        if cyrillic_match:
            word_part = normalize_word_part(line[: cyrillic_match.start()])
            translation_hint = line[cyrillic_match.start() :].strip()
            if word_part and translation_hint:
                return word_part, translation_hint

        return normalize_word_part(line), None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        word_part, translation_hint = split_word_and_translation(line)
        cleaned = clean_word(word_part)
        if not cleaned:
            continue
        entries.append(
            ParsedWordEntry(word=cleaned, translation_hint=translation_hint or None)
        )
    return entries


def update_learning_streak(user: TelegramUser) -> TelegramUser:
    progress = get_or_create_user_course_progress(user)
    today = now().date()
    if progress.last_study_date == today:
        return user

    if progress.last_study_date and (today - progress.last_study_date).days == 1:
        progress.consecutive_days += 1
    else:
        progress.consecutive_days = 1

    progress.total_study_days += 1
    progress.last_study_date = today
    progress.save(
        update_fields=["consecutive_days", "total_study_days", "last_study_date"]
    )
    return user
