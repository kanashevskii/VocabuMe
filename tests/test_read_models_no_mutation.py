from datetime import timedelta

import pytest
from django.utils import timezone

from vocab.models import AddWordDraft, PackPreparedWord, TelegramUser, VocabularyItem
from vocab.services import (
    _preferred_served_image,
    list_word_packs,
    list_words,
    serialize_draft,
    serialize_word,
)


@pytest.mark.django_db
def test_word_read_models_do_not_clear_stale_generation_flags():
    user = TelegramUser.objects.create(chat_id=60_001, username="read-model")
    stale_at = timezone.now() - timedelta(hours=2)
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="An apple a day.",
        image_generation_in_progress=True,
    )
    draft = AddWordDraft.objects.create(
        user=user,
        source_text="pear",
        word="pear",
        normalized_word="pear",
        translation="груша",
        image_generation_in_progress=True,
    )
    VocabularyItem.objects.filter(pk=item.pk).update(updated_at=stale_at)
    AddWordDraft.objects.filter(pk=draft.pk).update(updated_at=stale_at)
    item.refresh_from_db()
    draft.refresh_from_db()

    assert serialize_word(item)["image_generation_in_progress"] is False
    assert serialize_draft(draft)["image_generation_in_progress"] is False
    assert list_words(user) == [item]

    item.refresh_from_db()
    draft.refresh_from_db()
    assert item.image_generation_in_progress is True
    assert item.updated_at == stale_at
    assert draft.image_generation_in_progress is True
    assert draft.updated_at == stale_at


@pytest.mark.django_db
def test_pack_read_model_does_not_create_placeholder_rows():
    user = TelegramUser.objects.create(chat_id=60_002, username="pack-read-model")

    assert PackPreparedWord.objects.count() == 0
    packs = list_word_packs(user)

    assert packs
    assert PackPreparedWord.objects.count() == 0


def test_image_read_model_does_not_enqueue_an_optimization_job(tmp_path, monkeypatch):
    source = tmp_path / "card.jpg"
    source.write_bytes(b"not-an-image")
    monkeypatch.setattr(
        "vocab.services._schedule_image_optimization",
        lambda _source: pytest.fail("a read path must not enqueue work"),
    )

    assert _preferred_served_image(source) == source
