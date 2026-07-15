import pytest

from vocab.models import TelegramUser, VocabularyItem
from vocab.services import _sample_distinct_values


@pytest.mark.django_db
def test_distinct_sampling_uses_a_bounded_offset(monkeypatch):
    user = TelegramUser.objects.create(chat_id=40_002, username="sampling")
    for word in ("alpha", "bravo", "charlie", "delta"):
        VocabularyItem.objects.create(
            user=user,
            word=word,
            normalized_word=word,
            translation=word,
            transcription="",
            example=f"{word} example.",
        )
    monkeypatch.setattr("vocab.services.random.randrange", lambda total: 2)

    sampled = _sample_distinct_values(
        VocabularyItem.objects.filter(user=user), "word", count=3
    )

    assert sampled == ["charlie", "delta", "alpha"]
