from datetime import timedelta

import pytest
from django.utils import timezone

from vocab.models import TelegramUser, VocabularyItem
from vocab.services import (
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
def test_streak_restarts_after_a_missed_day():
    user = TelegramUser.objects.create(chat_id=20_002, username="streak-user")
    progress = get_or_create_user_course_progress(user)
    progress.consecutive_days = 7
    progress.total_study_days = 7
    progress.last_study_date = timezone.now().date() - timedelta(days=2)
    progress.save(
        update_fields=["consecutive_days", "total_study_days", "last_study_date"]
    )

    update_learning_streak(user)
    progress.refresh_from_db()

    assert progress.consecutive_days == 1
    assert progress.total_study_days == 8
    assert progress.last_study_date == timezone.now().date()


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
