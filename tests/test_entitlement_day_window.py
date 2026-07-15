from datetime import datetime, time, timedelta

import pytest
from django.utils import timezone

from vocab.models import TelegramUser, VocabularyItem
from vocab.services import get_daily_entitlement_usage


@pytest.mark.django_db
def test_daily_entitlement_usage_uses_a_timezone_aware_range():
    user = TelegramUser.objects.create(chat_id=40_001, username="entitlement-window")
    usage_date = timezone.localdate()
    today_start = timezone.make_aware(datetime.combine(usage_date, time.min))
    earlier_item = VocabularyItem.objects.create(
        user=user,
        word="earlier",
        normalized_word="earlier",
        translation="раньше",
        transcription="",
        example="Earlier example.",
    )
    today_item = VocabularyItem.objects.create(
        user=user,
        word="today",
        normalized_word="today",
        translation="сегодня",
        transcription="",
        example="Today example.",
    )
    VocabularyItem.objects.filter(id=earlier_item.id).update(
        created_at=today_start - timedelta(microseconds=1)
    )
    VocabularyItem.objects.filter(id=today_item.id).update(created_at=today_start)

    usage = get_daily_entitlement_usage(user, target_date=usage_date)

    assert usage.new_items_added == 1
