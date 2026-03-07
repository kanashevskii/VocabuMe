from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable
import random
import re
import time

from django.db.models import Count, Min, Q
from django.utils import timezone
from django.utils.timezone import now

from .irregular_verbs import IRREGULAR_VERBS, get_random_pairs
from .models import AddWordDraft, IrregularVerbProgress, TelegramUser, VocabularyItem, WebLoginToken
from .openai_utils import build_visual_prompt, generate_card_image, generate_word_data
from .utils import clean_word, translate_to_ru

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "media"
IMAGE_CACHE_DIR = MEDIA_ROOT / "card_images"
USER_IMAGE_DIR = MEDIA_ROOT / "user_images"
DRAFT_IMAGE_DIR = MEDIA_ROOT / "draft_images"
ACHIEVEMENT_DEFINITIONS = [
    {"kind": "words", "threshold": 10, "text": "🎉 Выучено 10 слов — Первый шаг!"},
    {"kind": "words", "threshold": 50, "text": "🌿 Выучено 50 слов — Хороший темп!"},
    {"kind": "words", "threshold": 100, "text": "🎯 Выучено 100 слов — Опытный!"},
    {"kind": "words", "threshold": 200, "text": "🚀 Выучено 200+ слов — Гуру слов!"},
    {"kind": "irregular", "threshold": 10, "text": "🔤 10 неправильных глаголов — База собрана!"},
    {"kind": "irregular", "threshold": 30, "text": "🧩 30 неправильных глаголов — Уже уверенно!"},
    {"kind": "irregular", "threshold": 60, "text": "🏆 60 неправильных глаголов — Мастер форм!"},
    {"kind": "days", "threshold": 3, "text": "📆 3 дня подряд — Ты в ритме!"},
    {"kind": "days", "threshold": 7, "text": "📅 7 дней подряд — Неделя прогресса!"},
    {"kind": "days", "threshold": 14, "text": "🧭 14 дней подряд — Курс на успех!"},
    {"kind": "days", "threshold": 30, "text": "🔥 30 дней подряд — Мастер привычки!"},
    {"kind": "days", "threshold": 60, "text": "🕯️ 60 дней подряд — Упорство без пауз!"},
    {"kind": "days", "threshold": 100, "text": "⚔️ 100 дней подряд — Воин знаний!"},
    {"kind": "days", "threshold": 200, "text": "🛡️ 200 дней подряд — Гуру дисциплины!"},
    {"kind": "days", "threshold": 365, "text": "🌈 365 дней подряд — Год знаний!"},
]


@dataclass
class ParsedWordEntry:
    word: str
    translation_hint: str | None = None


def upsert_telegram_user(chat_id: int, username: str | None = None) -> TelegramUser:
    user, created = TelegramUser.objects.get_or_create(
        chat_id=chat_id,
        defaults={"username": username},
    )
    if not created and username and user.username != username:
        user.username = username
        user.save(update_fields=["username"])
    return user


def get_user_achievements(user: TelegramUser) -> list[str]:
    learned = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    days = user.consecutive_days or 0
    irregular = IrregularVerbProgress.objects.filter(user=user, is_learned=True).count()
    stats = {"words": learned, "days": days, "irregular": irregular}
    return [item["text"] for item in ACHIEVEMENT_DEFINITIONS if stats[item["kind"]] >= item["threshold"]]


def get_pending_achievements(user: TelegramUser) -> list[dict]:
    learned = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    days = user.consecutive_days or 0
    irregular = IrregularVerbProgress.objects.filter(user=user, is_learned=True).count()
    stats = {"words": learned, "days": days, "irregular": irregular}

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
    return pending[:8]


def build_user_progress(user: TelegramUser) -> dict:
    total = VocabularyItem.objects.filter(user=user).count()
    learned = VocabularyItem.objects.filter(user=user, is_learned=True).count()
    learning = total - learned
    irregular_learned = IrregularVerbProgress.objects.filter(user=user, is_learned=True).count()
    start_date = VocabularyItem.objects.filter(user=user).aggregate(Min("created_at"))["created_at__min"]
    today = now().date()

    user_stats = TelegramUser.objects.annotate(
        learned_count=Count("vocabularyitem", filter=Q(vocabularyitem__is_learned=True))
    ).order_by("-learned_count")

    total_users = user_stats.count()
    better_than = sum(1 for candidate in user_stats if candidate.learned_count < learned)
    rank_percent = round(100 * (1 - better_than / total_users)) if total_users else None

    return {
        "total": total,
        "learned": learned,
        "learning": learning,
        "irregular": irregular_learned,
        "start_date": start_date.isoformat() if start_date else None,
        "rank_percent": rank_percent,
        "achievements": get_user_achievements(user),
        "pending_achievements": get_pending_achievements(user),
        "streak_days": user.consecutive_days,
        "study_days": user.total_study_days,
        "studied_today": user.last_study_date == today,
    }


def get_word_image_file(item: VocabularyItem) -> Path | None:
    candidates: list[Path] = []

    if item.image_path:
        raw_path = Path(item.image_path)
        candidates.append(raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path)

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
            return resolved
    return None


def serialize_user(user: TelegramUser) -> dict:
    return {
        "id": user.id,
        "chat_id": user.chat_id,
        "username": user.username,
        "display_name": user.username or f"user{user.chat_id}",
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
        "correct_count": item.correct_count,
        "is_learned": item.is_learned,
        "image_path": item.image_path,
        "has_image": image_file is not None,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def list_words(user: TelegramUser, search: str = "", status: str = "all", limit: int = 100) -> list[VocabularyItem]:
    qs = VocabularyItem.objects.filter(user=user).order_by("-updated_at", "-id")

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


def word_already_exists(user: TelegramUser, word: str) -> bool:
    return VocabularyItem.objects.filter(user=user, normalized_word=clean_word(word)).exists()


def resolve_shared_image_path(word: str, translation: str, preferred_path: str = "") -> str:
    if preferred_path:
        return preferred_path

    normalized_word = clean_word(word)
    normalized_translation = (translation or "").strip()
    if not normalized_word or not normalized_translation:
        return ""

    existing = (
        VocabularyItem.objects.filter(
            normalized_word=normalized_word,
            translation__iexact=normalized_translation,
        )
        .exclude(image_path="")
        .order_by("-updated_at", "-id")
        .first()
    )
    return existing.image_path if existing else ""


def create_word(user: TelegramUser, data: dict) -> VocabularyItem:
    word = clean_word(data["word"])
    transcription = data.get("transcription", "") or ""
    if any(char in transcription for char in "абвгдеёжзийклмнопрстуфхцчшщыэюя"):
        transcription = ""

    example_translation = data.get("example_translation") or translate_to_ru(data.get("example", ""))
    image_path = resolve_shared_image_path(word, data["translation"], data.get("image_path", ""))
    return VocabularyItem.objects.create(
        user=user,
        word=word,
        normalized_word=word,
        translation=data["translation"],
        transcription=transcription,
        example=data["example"],
        example_translation=example_translation,
        part_of_speech=data.get("part_of_speech", "unknown"),
        image_path=image_path,
    )


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
        return resolved
    return None


def create_word_draft(user: TelegramUser, source_text: str, generated: dict, translation_hint: str | None = None) -> AddWordDraft:
    translation = (translation_hint or generated.get("translation") or "").strip()
    return AddWordDraft.objects.create(
        user=user,
        source_text=source_text,
        word=generated["word"],
        normalized_word=clean_word(generated["word"]),
        translation=translation,
        translation_confirmed=bool(translation_hint),
        transcription=generated.get("transcription", "") or "",
        example=generated.get("example", "") or "",
        example_translation=generated.get("example_translation") or translate_to_ru(generated.get("example", "")),
        part_of_speech=generated.get("part_of_speech", "unknown"),
    )


def refresh_draft_language_data(draft: AddWordDraft, translation: str) -> AddWordDraft:
    generated = generate_word_data(draft.word, part_hint=draft.part_of_speech, translation_hint=translation)
    if generated:
        draft.translation = translation
        draft.translation_confirmed = True
        draft.transcription = generated.get("transcription", "") or draft.transcription
        draft.example = generated.get("example", "") or draft.example
        draft.example_translation = generated.get("example_translation") or translate_to_ru(draft.example)
        draft.part_of_speech = generated.get("part_of_speech", draft.part_of_speech) or draft.part_of_speech
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
            "updated_at",
        ]
    )
    return draft


def ensure_draft_image(draft: AddWordDraft, force_regenerate: bool = False) -> AddWordDraft:
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
    draft.save(update_fields=["image_prompt", "image_path", "updated_at"])
    return draft


def finalize_word_draft(draft: AddWordDraft, use_image: bool = True) -> VocabularyItem:
    payload = {
        "word": draft.word,
        "translation": draft.translation,
        "transcription": draft.transcription,
        "example": draft.example,
        "example_translation": draft.example_translation,
        "part_of_speech": draft.part_of_speech,
        "image_path": draft.image_path if use_image else "",
    }
    item = create_word(draft.user, payload)
    draft.delete()
    return item


def get_ordered_unlearned_words(
    user: TelegramUser,
    count: int = 10,
    exclude_ids: Iterable[int] | None = None,
) -> list[VocabularyItem]:
    exclude_ids = list(exclude_ids or [])
    return list(
        VocabularyItem.objects.filter(user=user, is_learned=False)
        .exclude(id__in=exclude_ids)
        .order_by("created_at", "id")[:count]
    )


def get_unlearned_words(user: TelegramUser, count: int = 10, part_of_speech: str | None = None) -> list[VocabularyItem]:
    base_qs = VocabularyItem.objects.filter(user=user, is_learned=False)
    if part_of_speech:
        base_qs = base_qs.filter(part_of_speech=part_of_speech)
    base_ids = list(base_qs.values_list("id", flat=True))

    review_ids: list[int] = []
    if user.enable_review_old_words:
        threshold = now() - timedelta(days=user.days_before_review)
        review_qs = VocabularyItem.objects.filter(user=user, is_learned=True, updated_at__lt=threshold)
        if part_of_speech:
            review_qs = review_qs.filter(part_of_speech=part_of_speech)
        review_ids = list(review_qs.values_list("id", flat=True))

    all_ids = base_ids + review_ids
    if not all_ids:
        return []
    selected_ids = random.sample(all_ids, min(len(all_ids), count))
    return list(VocabularyItem.objects.filter(id__in=selected_ids))


def get_learned_words(user: TelegramUser) -> list[VocabularyItem]:
    return list(VocabularyItem.objects.filter(user=user, is_learned=True).order_by("updated_at", "id"))


def update_word_progress(item_id: int, correct: bool) -> VocabularyItem:
    item = VocabularyItem.objects.select_related("user").get(id=item_id)
    if correct:
        item.correct_count += 1
        threshold = item.user.repeat_threshold if item.user.repeat_threshold else 3
        if item.correct_count >= threshold:
            item.is_learned = True
    item.save(update_fields=["correct_count", "is_learned", "updated_at"])
    return item


def reset_word_progress(item_id: int) -> VocabularyItem:
    item = VocabularyItem.objects.get(id=item_id)
    item.is_learned = False
    item.correct_count = 0
    item.save(update_fields=["is_learned", "correct_count", "updated_at"])
    return item


def get_fake_translations(user: TelegramUser, exclude_word: str, part_of_speech: str | None = None, count: int = 3) -> list[str]:
    qs = VocabularyItem.objects.exclude(word__iexact=exclude_word)
    if part_of_speech:
        qs = qs.filter(part_of_speech=part_of_speech)

    translations = list(qs.values_list("translation", flat=True).distinct().order_by("?")[:count])
    if len(translations) < count:
        extras = list(
            VocabularyItem.objects.exclude(word__iexact=exclude_word)
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
            VocabularyItem.objects.exclude(id=item.id)
            .values_list("word", flat=True)
            .distinct()
            .order_by("?")[:3]
        )
    else:
        fakes = get_fake_translations(user, exclude_word=item.word, part_of_speech=item.part_of_speech, count=3)

    options = list(dict.fromkeys(fakes + [correct_answer]))
    random.shuffle(options)
    return {"item": serialize_word(item), "options": options, "mode": mode, "prompt": prompt}


def submit_choice_answer(user: TelegramUser, item_id: int, answer: str, mode: str) -> dict:
    item = VocabularyItem.objects.get(id=item_id, user=user)
    normalized = answer.strip().lower()

    if mode == "reverse":
        correct = normalized == item.word.lower()
        updated = update_word_progress(item.id, correct=correct)
        correct_answer = item.word
    elif mode == "review":
        correct = normalized == item.translation.lower()
        updated = update_word_progress(item.id, correct=True) if correct else reset_word_progress(item.id)
        correct_answer = item.translation
    else:
        correct = normalized == item.translation.lower()
        updated = update_word_progress(item.id, correct=correct)
        correct_answer = item.translation

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
    prompt = "Напиши услышанное слово" if mode == "word" else "Напиши перевод услышанного слова"
    return {"item": serialize_word(item), "mode": mode, "prompt": prompt}


def submit_listening_answer(user: TelegramUser, item_id: int, answer: str, mode: str) -> dict:
    item = VocabularyItem.objects.get(id=item_id, user=user)
    expected = item.word if mode == "word" else item.translation
    correct = answer.strip().lower() == expected.lower()
    updated = update_word_progress(item.id, correct=correct)
    update_learning_streak(user)
    return {
        "correct": correct,
        "item": serialize_word(updated),
        "correct_answer": expected,
        "progress": build_user_progress(user),
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


def update_irregular_progress(user: TelegramUser, base: str, correct: bool) -> IrregularVerbProgress:
    progress, _ = IrregularVerbProgress.objects.get_or_create(user=user, verb_base=base)
    if correct:
        progress.correct_count += 1
        if not progress.is_learned and progress.correct_count >= 5:
            progress.is_learned = True
    progress.save()
    return progress


def create_web_login_token() -> WebLoginToken:
    return WebLoginToken.objects.create(expires_at=timezone.now() + timedelta(minutes=15))


def bind_web_login_token(token: str, user: TelegramUser) -> WebLoginToken | None:
    try:
        login_token = WebLoginToken.objects.get(token=token, expires_at__gt=timezone.now(), consumed_at__isnull=True)
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
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if " - " in line:
            word_part, translation_hint = line.split(" - ", 1)
        elif " — " in line:
            word_part, translation_hint = line.split(" — ", 1)
        else:
            word_part, translation_hint = line, None

        cleaned = clean_word(word_part)
        if not cleaned:
            continue
        entries.append(ParsedWordEntry(word=cleaned, translation_hint=translation_hint or None))
    return entries


def update_learning_streak(user: TelegramUser) -> TelegramUser:
    today = now().date()
    if user.last_study_date == today:
        return user

    if user.last_study_date and (today - user.last_study_date).days == 1:
        user.consecutive_days += 1
    else:
        user.consecutive_days = 1

    user.total_study_days += 1
    user.last_study_date = today
    user.save(update_fields=["consecutive_days", "total_study_days", "last_study_date"])
    return user
