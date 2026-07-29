"""Course-scoped progress and achievement read models.

The module deliberately receives resolved course and calendar context from the
legacy services facade. That keeps application code independent from that
facade and prevents import cycles while callers keep their public API.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import cast

from django.db.models import Min
from django.utils import timezone

from vocab.application.streaks import (
    active_streak_days,
    is_study_day_qualified,
    qualified_study_days_count,
)
from vocab.models import (
    Achievement,
    IrregularVerbProgress,
    TelegramUser,
    UserCourseProgress,
    VocabularyItem,
)
from vocab.selectors.progress import get_rank_percent

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


def get_course_progress_stats(
    user: TelegramUser,
    *,
    course_code: str,
    course_progress: UserCourseProgress,
    today: date,
) -> dict:
    learned = VocabularyItem.objects.filter(
        user=user, course_code=course_code, is_learned=True
    ).count()
    irregular = IrregularVerbProgress.objects.filter(
        user=user, course_code=course_code, is_learned=True
    ).count()
    return {
        "words": learned,
        "days": active_streak_days(course_progress, today),
        "irregular": irregular,
        "practice": course_progress.practice_correct or 0,
        "listening": course_progress.listening_correct or 0,
        "speaking": course_progress.speaking_correct or 0,
        "review": course_progress.review_correct or 0,
        "points": course_progress.total_points or 0,
    }


def get_user_achievements(stats: dict) -> list[str]:
    return [
        str(item["text"])
        for item in ACHIEVEMENT_DEFINITIONS
        if stats[str(item["kind"])] >= cast(int, item["threshold"])
    ]


def get_new_achievements(
    user: TelegramUser, *, course_code: str, stats: dict
) -> list[str]:
    earned = set(
        Achievement.objects.filter(user=user, course_code=course_code).values_list(
            "code", flat=True
        )
    )
    new_achievements: list[str] = []

    for item in ACHIEVEMENT_DEFINITIONS:
        code = f"{item['kind']}_{item['threshold']}"
        if (
            stats[str(item["kind"])] >= cast(int, item["threshold"])
            and code not in earned
        ):
            Achievement.objects.create(user=user, course_code=course_code, code=code)
            new_achievements.append(str(item["text"]))
    return new_achievements


def get_pending_achievements(stats: dict) -> list[dict]:
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


def get_pending_achievement_highlights(stats: dict) -> list[dict]:
    highlights: list[dict] = []
    seen_kinds: set[str] = set()
    for item in get_pending_achievements(stats):
        if item["kind"] in seen_kinds:
            continue
        seen_kinds.add(item["kind"])
        highlights.append(item)
    return highlights


def build_user_progress(
    user: TelegramUser,
    *,
    course_code: str,
    course_progress: UserCourseProgress,
    today: date,
    day_start: datetime,
    day_end: datetime,
) -> dict:
    items = VocabularyItem.objects.filter(user=user, course_code=course_code)
    total = items.count()
    learned = items.filter(is_learned=True).count()
    stats = get_course_progress_stats(
        user,
        course_code=course_code,
        course_progress=course_progress,
        today=today,
    )
    irregular_learned = IrregularVerbProgress.objects.filter(
        user=user, course_code=course_code, is_learned=True
    ).count()
    start_date = items.aggregate(Min("created_at"))["created_at__min"]
    learned_today = items.filter(
        learned_at__gte=day_start, learned_at__lt=day_end
    ).count()
    current_moment = timezone.now()
    learned_week = items.filter(
        learned_at__gte=current_moment - timedelta(days=7)
    ).count()
    learned_month = items.filter(
        learned_at__gte=current_moment - timedelta(days=30)
    ).count()

    return {
        "total": total,
        "learned": learned,
        "learning": total - learned,
        "irregular": irregular_learned,
        "start_date": start_date.isoformat() if start_date else None,
        "rank_percent": get_rank_percent(
            course_code=course_code, learned_count=learned
        ),
        "achievements": get_user_achievements(stats),
        "pending_achievements": get_pending_achievements(stats),
        "pending_achievement_highlights": get_pending_achievement_highlights(stats),
        "streak_days": active_streak_days(course_progress, today),
        "study_days": qualified_study_days_count(course_progress),
        "studied_today": is_study_day_qualified(course_progress, today),
        "learned_today": learned_today,
        "learned_week": learned_week,
        "learned_month": learned_month,
        "practice_correct": course_progress.practice_correct,
        "listening_correct": course_progress.listening_correct,
        "speaking_correct": course_progress.speaking_correct,
        "review_correct": course_progress.review_correct,
        "total_points": course_progress.total_points,
        "course_code": course_code,
    }
