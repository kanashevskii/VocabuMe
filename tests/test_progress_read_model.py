import pytest

from vocab.models import TelegramUser, UserCourseProgress
from vocab.services import build_user_progress


@pytest.mark.django_db
def test_build_user_progress_does_not_create_course_progress_on_read():
    user = TelegramUser.objects.create(chat_id=30_001, username="read-only")

    payload = build_user_progress(user)

    assert payload["streak_days"] == 0
    assert UserCourseProgress.objects.filter(user=user).exists() is False
