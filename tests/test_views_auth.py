from __future__ import annotations

import json
from io import BytesIO

import pytest
from django.contrib.auth.hashers import make_password
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from vocab.models import AppErrorLog, TelegramUser
from vocab.services import create_web_login_token


@pytest.mark.django_db
def test_app_config_returns_public_auth_configuration(client):
    response = client.get("/api/app-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "bot_username" in payload
    assert "webapp_url" in payload


@pytest.mark.django_db
def test_auth_web_register_creates_user_and_session(client):
    response = client.post(
        "/api/auth/web/register",
        data=json.dumps({"email": "user@example.com", "password": "supersecret"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["user"]["email"] == "user@example.com"

    me_response = client.get("/api/auth/me")
    me_payload = me_response.json()
    assert me_payload["authenticated"] is True
    assert me_payload["user"]["email"] == "user@example.com"


@pytest.mark.django_db
def test_auth_web_login_rejects_invalid_password(client):
    TelegramUser.objects.create(
        chat_id=-1,
        username="user",
        email="user@example.com",
        password_hash=make_password("supersecret"),
        auth_provider="web",
    )

    response = client.post(
        "/api/auth/web/login",
        data=json.dumps({"email": "user@example.com", "password": "wrong-password"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid email or password."


@pytest.mark.django_db
def test_auth_logout_clears_session(client):
    user = TelegramUser.objects.create(chat_id=2000, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post("/api/auth/logout")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert client.get("/api/auth/me").json()["authenticated"] is False


@pytest.mark.django_db
def test_auth_request_link_returns_deep_link(client):
    response = client.post("/api/auth/telegram/request-link")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["token"]
    assert "start=login_" in payload["deep_link"]


@pytest.mark.django_db
def test_auth_poll_link_authenticates_bound_token(client):
    user = TelegramUser.objects.create(chat_id=2002, username="tester")
    token = create_web_login_token()
    token.user = user
    token.save(update_fields=["user"])

    response = client.get(f"/api/auth/telegram/poll/{token.token}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is True
    assert payload["user"]["chat_id"] == 2002


@pytest.mark.django_db
def test_auth_poll_link_returns_not_authenticated_for_unknown_token(client):
    response = client.get("/api/auth/telegram/poll/unknown-token")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "authenticated": False}


@pytest.mark.django_db
def test_auth_telegram_widget_uses_verified_payload(client, monkeypatch):
    monkeypatch.setattr(
        "vocab.views.verify_login_widget",
        lambda payload, token: {"id": "301", "username": "telegram_user"},
    )

    response = client.post(
        "/api/auth/telegram/widget",
        data=json.dumps({"id": "301", "hash": "whatever"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["chat_id"] == 301
    assert payload["user"]["username"] == "telegram_user"


@pytest.mark.django_db
def test_auth_telegram_webapp_uses_verified_payload(client, monkeypatch):
    monkeypatch.setattr(
        "vocab.views.verify_webapp_init_data",
        lambda init_data, token: {"user": {"id": 302, "username": "webapp_user"}},
    )

    response = client.post(
        "/api/auth/telegram/webapp",
        data=json.dumps({"init_data": "signed"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["chat_id"] == 302
    assert payload["user"]["username"] == "webapp_user"


@pytest.mark.django_db
def test_words_endpoint_requires_authenticated_user(client):
    response = client.get("/api/words")

    assert response.status_code == 401
    assert response.json()["error"] == "Authentication required."


@pytest.mark.django_db
def test_words_post_rejects_empty_batch_for_authenticated_user(client):
    user = TelegramUser.objects.create(chat_id=2001, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/words",
        data=json.dumps({"text": "   "}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Add at least one word."


@pytest.mark.django_db
def test_words_get_returns_filtered_items_for_authenticated_user(client):
    user = TelegramUser.objects.create(chat_id=2003, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    from vocab.models import VocabularyItem

    VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="An apple a day.",
        example_translation="Яблоко в день.",
        is_learned=False,
    )
    VocabularyItem.objects.create(
        user=user,
        word="pear",
        normalized_word="pear",
        translation="груша",
        transcription="",
        example="A pear a day.",
        example_translation="Груша в день.",
        is_learned=True,
    )

    response = client.get("/api/words?search=app&status=learning")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["word"] == "apple"


@pytest.mark.django_db
def test_client_error_log_persists_context(client):
    user = TelegramUser.objects.create(chat_id=2004, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/client-error",
        data=json.dumps(
            {
                "category": "client",
                "level": "error",
                "message": "boom",
                "status_code": 500,
                "url": "/words",
                "detail": "trace",
                "meta": {"nested": True},
            }
        ),
        content_type="application/json",
    )

    assert response.status_code == 200
    log = AppErrorLog.objects.get()
    assert log.user == user
    assert log.message == "boom"
    assert log.context["url"] == "/words"


@pytest.mark.django_db
def test_learn_question_returns_empty_when_service_has_no_question(client, monkeypatch):
    user = TelegramUser.objects.create(
        chat_id=2005, username="tester", session_question_limit=99
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()
    monkeypatch.setattr(
        "vocab.views.build_learning_question", lambda user, exclude_ids=None: None
    )

    response = client.get("/api/learn/question")

    assert response.status_code == 200
    payload = response.json()
    assert payload["empty"] is True
    assert payload["session_limit"] == 50


@pytest.mark.django_db
def test_alphabet_list_returns_course_scoped_letters(client):
    user = TelegramUser.objects.create(
        chat_id=2006, username="tester", active_studied_language="ka"
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.get("/api/alphabet/list?page=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["course_code"] == "ka"
    assert payload["items"][0]["symbol"] == "ა"
    assert payload["items"][0]["transcription"] == "ɑ"


@pytest.mark.django_db
def test_alphabet_question_returns_current_course_payload(client, monkeypatch):
    user = TelegramUser.objects.create(chat_id=2007, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()
    monkeypatch.setattr(
        "vocab.views.build_alphabet_question",
        lambda current_user: {
            "course_code": "en",
            "letter": {
                "symbol": "B",
                "name": "B",
                "transcription": "biː",
                "hint": "би",
            },
            "correct_symbol": "B",
            "options": ["A", "B", "C", "D"],
        },
    )

    response = client.get("/api/alphabet/question")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["question"]["letter"]["symbol"] == "B"
    assert payload["question"]["options"] == ["A", "B", "C", "D"]


@pytest.mark.django_db
def test_alphabet_answer_returns_validation_error_for_bad_payload(client):
    user = TelegramUser.objects.create(chat_id=2008, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/alphabet/answer",
        data="{bad json",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid alphabet payload."


@pytest.mark.django_db
def test_alphabet_answer_returns_result_payload(client, monkeypatch):
    user = TelegramUser.objects.create(
        chat_id=2009, username="tester", active_studied_language="ka"
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()
    monkeypatch.setattr(
        "vocab.views.submit_alphabet_answer",
        lambda current_user, symbol, answer: {
            "correct": True,
            "correct_answer": symbol,
            "letter": {
                "symbol": symbol,
                "name": "განი",
                "transcription": "ɡ",
                "hint": "г",
            },
            "progress": {"course_code": "ka", "practice_correct": 1},
        },
    )

    response = client.post(
        "/api/alphabet/answer",
        data=json.dumps({"symbol": "გ", "answer": "გ"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["correct"] is True
    assert payload["letter"]["transcription"] == "ɡ"
    assert payload["progress"]["course_code"] == "ka"


@pytest.mark.django_db
def test_alphabet_audio_returns_mp3_for_current_course(client, monkeypatch, tmp_path):
    user = TelegramUser.objects.create(
        chat_id=2010, username="tester", active_studied_language="ka"
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    audio_path = tmp_path / "alphabet.mp3"
    audio_path.write_bytes(b"fake mp3")

    monkeypatch.setattr("vocab.tts.get_audio_path", lambda text, language_code=None: str(audio_path))

    response = client.get("/api/alphabet/audio?symbol=ა")

    assert response.status_code == 200
    assert response["Content-Type"] == "audio/mpeg"


@pytest.mark.django_db
def test_settings_view_includes_georgian_display_mode_fields(client):
    user = TelegramUser.objects.create(
        chat_id=2011,
        username="tester",
        active_studied_language="ka",
        georgian_display_mode="both",
        has_selected_georgian_display_mode=True,
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()["settings"]
    assert payload["georgian_display_mode"] == "both"
    assert payload["has_selected_georgian_display_mode"] is True
    assert payload["georgian_display_mode_options"][0]["code"] == "both"


@pytest.mark.django_db
def test_learn_answer_rejects_unknown_exercise_type(client):
    user = TelegramUser.objects.create(chat_id=2006, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/learn/answer",
        data=json.dumps({"word_id": 1, "answer": "x", "exercise_type": "bad"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Unknown exercise type."


@pytest.mark.django_db
def test_practice_question_rejects_unknown_mode(client):
    user = TelegramUser.objects.create(chat_id=2007, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.get("/api/practice/question?mode=bad")

    assert response.status_code == 400
    assert response.json()["error"] == "Unknown practice mode."


@pytest.mark.django_db
def test_practice_question_returns_empty_when_service_has_no_question(
    client, monkeypatch
):
    user = TelegramUser.objects.create(chat_id=2008, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()
    monkeypatch.setattr("vocab.views.build_choice_question", lambda user, mode: None)

    response = client.get("/api/practice/question")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "empty": True}


@pytest.mark.django_db
def test_listening_question_rejects_unknown_mode(client):
    user = TelegramUser.objects.create(chat_id=2009, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.get("/api/listening/question?mode=bad")

    assert response.status_code == 400
    assert response.json()["error"] == "Unknown listening mode."


@pytest.mark.django_db
def test_listening_answer_invalid_payload_returns_error(client):
    user = TelegramUser.objects.create(chat_id=2010, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/listening/answer",
        data=json.dumps({"word_id": "bad"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid listening answer payload."


@pytest.mark.django_db
def test_settings_get_returns_current_values(client):
    user = TelegramUser.objects.create(chat_id=2011, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["settings"]["exercise_goal"] >= 2


@pytest.mark.django_db
def test_settings_post_updates_and_clamps_values(client):
    user = TelegramUser.objects.create(chat_id=2012, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/settings",
        data=json.dumps(
            {
                "exercise_goal": 99,
                "session_question_limit": 999,
                "days_before_review": 999,
                "reminder_interval_days": 999,
                "reminder_time": "09:30",
                "reminder_timezone": "UTC+3",
                "enable_review_old_words": False,
                "reminder_enabled": True,
            }
        ),
        content_type="application/json",
    )

    assert response.status_code == 200
    user.refresh_from_db()
    assert user.repeat_threshold == 5
    assert user.session_question_limit == 50
    assert user.days_before_review == 365
    assert user.reminder_interval_days == 30
    assert user.reminder_timezone == "UTC+03:00"


@pytest.mark.django_db
def test_settings_post_updates_custom_avatar_url(client):
    user = TelegramUser.objects.create(chat_id=2018, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/settings",
        data=json.dumps({"custom_avatar_url": "https://example.com/avatar.png"}),
        content_type="application/json",
    )

    assert response.status_code == 200
    user.refresh_from_db()
    assert user.custom_avatar_url == "https://example.com/avatar.png"


@pytest.mark.django_db
def test_profile_avatar_upload_updates_user_avatar(client):
    user = TelegramUser.objects.create(chat_id=2019, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    image = Image.new("RGB", (128, 128), color=(120, 30, 60))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    upload = SimpleUploadedFile(
        "avatar.png", buffer.getvalue(), content_type="image/png"
    )

    response = client.post("/api/profile/avatar", data={"avatar": upload})

    assert response.status_code == 200
    user.refresh_from_db()
    payload = response.json()
    assert payload["ok"] is True
    assert payload["user"]["avatar_url"].startswith("/api/profile/avatar")
    assert user.avatar_path.endswith(".webp")


@pytest.mark.django_db
def test_settings_post_rejects_invalid_payload(client):
    user = TelegramUser.objects.create(chat_id=2013, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/settings",
        data=json.dumps({"reminder_time": "bad-time"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "Invalid settings payload" in response.json()["error"]


@pytest.mark.django_db
def test_word_detail_patch_requires_translation(client):
    user = TelegramUser.objects.create(chat_id=2014, username="tester")
    from vocab.models import VocabularyItem

    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.generic(
        "PATCH",
        f"/api/words/{item.id}",
        data=json.dumps({"translation": ""}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Translation is required."


@pytest.mark.django_db
def test_word_detail_delete_removes_word(client):
    user = TelegramUser.objects.create(chat_id=2015, username="tester")
    from vocab.models import VocabularyItem

    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.delete(f"/api/words/{item.id}")

    assert response.status_code == 200
    assert VocabularyItem.objects.filter(id=item.id).exists() is False


@pytest.mark.django_db
def test_word_image_regenerate_rejects_limit_exhausted(client):
    user = TelegramUser.objects.create(chat_id=2016, username="tester")
    from vocab.models import VocabularyItem

    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        image_regeneration_count=3,
    )
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(f"/api/words/{item.id}/image/regenerate")

    assert response.status_code == 400
    assert "Лимит перегенерации" in response.json()["error"]


@pytest.mark.django_db
def test_packs_add_rejects_non_list_selected_words(client):
    user = TelegramUser.objects.create(chat_id=2017, username="tester")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/packs/add",
        data=json.dumps({"pack_id": "a", "level_id": "b", "selected_words": "apple"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "selected_words must be a list."
