from datetime import timedelta

import pytest
from django.utils import timezone

from vocab.models import TelegramUser, VocabularyItem
from vocab.tasks import clear_stale_image_generation_flags


@pytest.mark.django_db
def test_recovery_task_clears_stale_image_generation_flag():
    user = TelegramUser.objects.create(chat_id=70_001, username="recovery")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="An apple a day.",
        image_generation_in_progress=True,
    )
    VocabularyItem.objects.filter(pk=item.pk).update(
        updated_at=timezone.now() - timedelta(hours=2)
    )

    result = clear_stale_image_generation_flags.run()

    item.refresh_from_db()
    assert result["words"] == 1
    assert item.image_generation_in_progress is False
