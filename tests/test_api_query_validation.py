from __future__ import annotations

import pytest

from vocab.models import TelegramUser


def _authenticate(client, user: TelegramUser) -> None:
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("path", "expected_message"),
    [
        ("/api/irregular/list?page=not-a-number", "Query parameter 'page'"),
        ("/api/alphabet/list?page=not-a-number", "Query parameter 'page'"),
        ("/api/study/cards?count=not-a-number", "Query parameter 'count'"),
    ],
)
def test_malformed_integer_query_parameters_return_a_client_error(
    client, path: str, expected_message: str
):
    user = TelegramUser.objects.create(chat_id=70_001, username="query-validation")
    _authenticate(client, user)

    response = client.get(path)

    assert response.status_code == 400
    assert expected_message in response.json()["error"]


@pytest.mark.django_db
def test_study_cards_clamps_the_requested_count(client, monkeypatch):
    user = TelegramUser.objects.create(chat_id=70_002, username="query-bounds")
    _authenticate(client, user)
    captured: dict[str, int] = {}

    def capture_count(_user, *, count: int):
        captured["count"] = count
        return []

    monkeypatch.setattr("vocab.api.learning.get_ordered_unlearned_words", capture_count)

    response = client.get("/api/study/cards?count=999999")

    assert response.status_code == 200
    assert captured["count"] == 20
