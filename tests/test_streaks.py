from datetime import timedelta

import pytest
from django.utils import timezone

from vocab.models import TelegramUser, UserStudyDay, VocabularyItem
from vocab.services import (
    STREAK_QUALIFYING_CORRECT_ANSWERS,
    build_user_progress,
    get_or_create_user_course_progress,
    submit_choice_answer,
    update_learning_streak,
)


@pytest.mark.django_db
def test_progress_hides_expired_streak():
    user = TelegramUser.objects.create(chat_id=20_001, username="streak-user")
    progress = get_or_create_user_course_progress(user)
    progress.consecutive_days = 2
    progress.last_study_date = timezone.now().date() - timedelta(days=2)
    progress.save(update_fields=["consecutive_days", "last_study_date"])

    payload = build_user_progress(user)

    assert payload["streak_days"] == 0
    assert payload["studied_today"] is False


@pytest.mark.django_db
def test_progress_hides_legacy_streak_without_a_qualified_study_day():
    user = TelegramUser.objects.create(chat_id=20_005, username="legacy-streak")
    progress = get_or_create_user_course_progress(user)
    progress.consecutive_days = 2
    progress.last_study_date = timezone.now().date() - timedelta(days=1)
    progress.save(update_fields=["consecutive_days", "last_study_date"])

    payload = build_user_progress(user)

    assert payload["streak_days"] == 0
    assert payload["studied_today"] is False


@pytest.mark.django_db
def test_streak_restarts_after_a_missed_day():
    user = TelegramUser.objects.create(chat_id=20_002, username="streak-user")
    progress = get_or_create_user_course_progress(user)
    progress.consecutive_days = 7
    progress.total_study_days = 7
    progress.last_study_date = timezone.now().date() - timedelta(days=2)
    progress.save(
        update_fields=["consecutive_days", "total_study_days", "last_study_date"]
    )

    for _ in range(STREAK_QUALIFYING_CORRECT_ANSWERS):
        update_learning_streak(user)
    progress.refresh_from_db()

    assert progress.consecutive_days == 1
    assert progress.total_study_days == 8
    assert progress.last_study_date == timezone.now().date()


@pytest.mark.django_db
def test_streak_requires_a_qualified_block_of_correct_answers():
    user = TelegramUser.objects.create(chat_id=20_004, username="streak-threshold")
    progress = get_or_create_user_course_progress(user)

    for _ in range(STREAK_QUALIFYING_CORRECT_ANSWERS - 1):
        update_learning_streak(user)

    progress.refresh_from_db()
    study_day = UserStudyDay.objects.get(user=user, course_code="en")
    assert study_day.correct_answers == STREAK_QUALIFYING_CORRECT_ANSWERS - 1
    assert study_day.streak_qualified_at is None
    assert build_user_progress(user)["streak_days"] == 0
    assert build_user_progress(user)["studied_today"] is False

    update_learning_streak(user)

    progress.refresh_from_db()
    study_day.refresh_from_db()
    assert progress.consecutive_days == 1
    assert study_day.streak_qualified_at is not None
    assert build_user_progress(user)["streak_days"] == 1
    assert build_user_progress(user)["studied_today"] is True


@pytest.mark.django_db
def test_wrong_answer_does_not_start_streak():
    user = TelegramUser.objects.create(chat_id=20_003, username="streak-user")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = submit_choice_answer(user, item.id, "неверно", "classic")
    progress = get_or_create_user_course_progress(user)

    assert result["correct"] is False
    assert progress.last_study_date is None
    assert progress.consecutive_days == 0
