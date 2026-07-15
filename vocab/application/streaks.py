"""Auditable learning-streak use cases."""

from __future__ import annotations

from datetime import date, timedelta

from django.db import transaction
from django.utils import timezone

from vocab.models import TelegramUser, UserCourseProgress, UserStudyDay

STREAK_QUALIFYING_CORRECT_ANSWERS = 5


def _latest_qualified_study_day(progress: UserCourseProgress) -> UserStudyDay | None:
    return (
        UserStudyDay.objects.filter(
            user=progress.user,
            course_code=progress.course_code,
            streak_qualified_at__isnull=False,
        )
        .order_by("-study_date")
        .first()
    )


def active_streak_days(progress: UserCourseProgress, today: date) -> int:
    """Return a streak only when its latest day has auditable qualified activity."""
    latest_day = _latest_qualified_study_day(progress)
    if latest_day is None or latest_day.study_date != progress.last_study_date:
        return 0
    if latest_day.study_date not in {today, today - timedelta(days=1)}:
        return 0
    return progress.consecutive_days


def is_study_day_qualified(progress: UserCourseProgress, day: date) -> bool:
    return UserStudyDay.objects.filter(
        user=progress.user,
        course_code=progress.course_code,
        study_date=day,
        streak_qualified_at__isnull=False,
    ).exists()


def qualified_study_days_count(progress: UserCourseProgress) -> int:
    return UserStudyDay.objects.filter(
        user=progress.user,
        course_code=progress.course_code,
        streak_qualified_at__isnull=False,
    ).count()


def record_correct_answer(
    user: TelegramUser,
    *,
    course_code: str,
    study_date: date,
    get_or_create_progress,
) -> TelegramUser:
    """Record one correct answer and atomically qualify a meaningful study day."""
    with transaction.atomic():
        study_day, _ = UserStudyDay.objects.get_or_create(
            user=user,
            course_code=course_code,
            study_date=study_date,
        )
        study_day = UserStudyDay.objects.select_for_update().get(pk=study_day.pk)
        study_day.correct_answers += 1
        study_day.save(update_fields=["correct_answers", "updated_at"])
        if (
            study_day.correct_answers < STREAK_QUALIFYING_CORRECT_ANSWERS
            or study_day.streak_qualified_at is not None
        ):
            return user

        progress = get_or_create_progress(user, course_code)
        progress = UserCourseProgress.objects.select_for_update().get(pk=progress.pk)
        if (
            progress.last_study_date
            and (study_date - progress.last_study_date).days == 1
        ):
            progress.consecutive_days += 1
        else:
            progress.consecutive_days = 1
        progress.total_study_days += 1
        progress.last_study_date = study_date
        progress.save(
            update_fields=["consecutive_days", "total_study_days", "last_study_date"]
        )
        study_day.streak_qualified_at = timezone.now()
        study_day.save(update_fields=["streak_qualified_at", "updated_at"])
    return user
