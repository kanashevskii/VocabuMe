from __future__ import annotations

import json
from datetime import timedelta

import pytest
from django.utils import timezone

from vocab.analytics import (
    build_funnel_report,
    purge_expired_product_events,
    record_product_event,
)
from vocab.models import ProductEvent, TelegramUser


@pytest.mark.django_db
def test_client_analytics_event_requires_telegram_identity(client):
    response = client.post(
        "/api/analytics/events",
        data=json.dumps({"name": "paywall_opened"}),
        content_type="application/json",
    )

    assert response.status_code == 401
    assert ProductEvent.objects.count() == 0


@pytest.mark.django_db
def test_client_analytics_event_records_only_whitelisted_sanitized_data(client):
    user = TelegramUser.objects.create(chat_id=91_001, username="analytics")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/analytics/events",
        data=json.dumps(
            {
                "name": "paywall_opened",
                "properties": {
                    "source": "pack_gate",
                    "attempt": 2,
                    "nested": {"must": "not persist"},
                    "token": "not-special-but-bounded",
                },
            }
        ),
        content_type="application/json",
    )

    assert response.status_code == 202
    event = ProductEvent.objects.get()
    assert event.name == "paywall_opened"
    assert event.properties == {
        "source": "pack_gate",
        "attempt": 2,
    }


@pytest.mark.django_db
def test_client_analytics_event_rejects_unknown_names(client):
    user = TelegramUser.objects.create(chat_id=91_002, username="analytics")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/analytics/events",
        data=json.dumps({"name": "arbitrary_event"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert ProductEvent.objects.count() == 0


@pytest.mark.django_db
def test_funnel_report_aggregates_unique_users_and_purges_expired_events():
    first = TelegramUser.objects.create(chat_id=91_003, username="first")
    second = TelegramUser.objects.create(chat_id=91_004, username="second")
    record_product_event(first, "checkout_started", properties={"source": "settings"})
    record_product_event(first, "checkout_started")
    record_product_event(second, "checkout_started")
    expired = record_product_event(first, "paywall_opened")
    ProductEvent.objects.filter(pk=expired.pk).update(
        occurred_at=timezone.now() - timedelta(days=181)
    )

    report = build_funnel_report(days=30)

    assert report["events"]["checkout_started"] == {"events": 3, "users": 2}
    assert report["events"]["paywall_opened"] == {"events": 0, "users": 0}
    assert purge_expired_product_events() == 1
    assert not ProductEvent.objects.filter(pk=expired.pk).exists()
