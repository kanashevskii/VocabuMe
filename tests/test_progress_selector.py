import pytest

from vocab.models import TelegramUser, VocabularyItem
from vocab.selectors.progress import get_rank_percent


@pytest.mark.django_db
def test_rank_percent_is_calculated_in_the_database():
    lower = TelegramUser.objects.create(chat_id=91_001, username="lower")
    middle = TelegramUser.objects.create(chat_id=91_002, username="middle")
    higher = TelegramUser.objects.create(chat_id=91_003, username="higher")

    for user, count in ((lower, 1), (middle, 2), (higher, 3)):
        for position in range(count):
            VocabularyItem.objects.create(
                user=user,
                word=f"word-{user.id}-{position}",
                normalized_word=f"word-{user.id}-{position}",
                translation="слово",
                transcription="",
                example="Example.",
                example_translation="Пример.",
                is_learned=True,
            )

    assert get_rank_percent(course_code="en", learned_count=2) == 67
