from __future__ import annotations

import pytest
from vocab.models import AddWordDraft, TelegramUser, UserCourseProgress, VocabularyItem
from vocab.services import (
    authenticate_web_user,
    add_pack_words_to_user,
    apply_user_settings,
    build_choice_question,
    build_learning_question,
    build_listening_question,
    build_user_progress,
    build_speaking_question,
    consume_web_login_token,
    create_web_login_token,
    create_web_user,
    create_word_draft,
    create_word,
    finalize_word_draft,
    evaluate_speaking_answer,
    ensure_draft_image,
    get_active_course_code,
    get_completed_exercise_types,
    get_exercise_goal,
    get_or_create_user_course_progress,
    list_word_packs,
    get_pending_exercise_types,
    get_required_exercise_types,
    get_session_question_limit,
    is_single_typo_match,
    is_translation_answer_correct,
    parse_word_batch,
    recalculate_user_word_progress,
    refresh_draft_language_data,
    request_draft_image_generation,
    request_word_image_generation,
    submit_choice_answer,
    submit_learning_text_answer,
    submit_listening_answer,
    split_translation_variants,
    sync_word_learning_state,
)


@pytest.mark.django_db
def test_create_web_user_normalizes_email_and_hashes_password():
    user = create_web_user("  User@Example.COM ", "supersecret")

    assert user.email == "user@example.com"
    assert user.auth_provider == "web"
    assert user.chat_id < 0
    assert user.password_hash
    assert user.password_hash != "supersecret"


@pytest.mark.django_db
def test_create_web_user_rejects_duplicate_email_case_insensitive():
    create_web_user("user@example.com", "supersecret")

    with pytest.raises(ValueError, match="already exists"):
        create_web_user("USER@example.com", "anothersecret")


@pytest.mark.django_db
def test_authenticate_web_user_returns_matching_user():
    user = create_web_user("user@example.com", "supersecret")

    assert authenticate_web_user("USER@example.com", "supersecret") == user
    assert authenticate_web_user("user@example.com", "wrong-password") is None


@pytest.mark.django_db
def test_create_and_consume_web_login_token_is_single_use():
    user = TelegramUser.objects.create(chat_id=1001, username="tester")
    token = create_web_login_token()
    token.user = user
    token.save(update_fields=["user"])

    consumed = consume_web_login_token(token.token)

    token.refresh_from_db()
    assert consumed == user
    assert token.consumed_at is not None
    assert consume_web_login_token(token.token) is None


def test_parse_word_batch_supports_translation_hints_and_bullets():
    entries = parse_word_batch("• pride (n) - гордость\nrun: бежать\ncalm")

    assert [entry.word for entry in entries] == ["pride", "run", "calm"]
    assert [entry.translation_hint for entry in entries] == ["гордость", "бежать", None]


def test_translation_helpers_handle_variants_and_typos():
    assert split_translation_variants("дом, жилище / домик") == [
        "дом, жилище / домик",
        "дом",
        "жилище",
        "домик",
    ]
    assert is_translation_answer_correct("жилище", "дом, жилище / домик") is True
    assert is_translation_answer_correct("квартира", "дом, жилище / домик") is False
    assert is_single_typo_match("aplpe", "apple") is False
    assert is_single_typo_match("appl", "apple") is True
    assert is_single_typo_match("apple", "apple") is False


@pytest.mark.django_db
def test_create_word_normalizes_fields_and_generates_example_translation(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1002, username="tester")
    monkeypatch.setattr(
        "vocab.services.translate_to_ru", lambda text: "пример перевода"
    )

    item = create_word(
        user,
        {
            "word": "  Apple ",
            "translation": "яблоко",
            "transcription": "эпл",
            "example": "An apple a day.",
            "part_of_speech": "noun",
        },
    )

    assert item.word == "apple"
    assert item.normalized_word == "apple"
    assert item.transcription == ""
    assert item.example_translation == "пример перевода"
    assert item.part_of_speech == "noun"


@pytest.mark.django_db
def test_create_word_reuses_shared_image_path(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1003, username="tester")
    VocabularyItem.objects.create(
        user=TelegramUser.objects.create(chat_id=1004, username="other"),
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        image_path="media/card_images/shared.jpg",
    )
    monkeypatch.setattr("vocab.services.translate_to_ru", lambda text: "пример")

    item = create_word(
        user,
        {
            "word": "apple",
            "translation": "яблоко",
            "transcription": "",
            "example": "An apple a day.",
        },
    )

    assert item.image_path == "media/card_images/shared.jpg"


@pytest.mark.django_db
def test_word_progress_helpers_normalize_completed_types_and_learning_state():
    user = TelegramUser.objects.create(
        chat_id=1005, username="tester", repeat_threshold=3, session_question_limit=99
    )
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        completed_exercise_types=[
            "practice_en_ru",
            "practice_en_ru",
            "unknown",
            "listening_word",
        ],
    )

    assert get_exercise_goal(user) == 3
    assert get_session_question_limit(user) == 50
    assert get_required_exercise_types(user) == [
        "practice_en_ru",
        "listening_word",
        "practice_ru_en",
    ]
    assert get_completed_exercise_types(item) == ["practice_en_ru", "listening_word"]
    assert get_pending_exercise_types(item) == ["practice_ru_en"]

    item.completed_exercise_types.append("practice_ru_en")
    sync_word_learning_state(item)

    assert item.correct_count == 3
    assert item.is_learned is True
    assert item.learned_at is not None


@pytest.mark.django_db
def test_recalculate_user_word_progress_persists_learning_state():
    user = TelegramUser.objects.create(
        chat_id=1006, username="tester", repeat_threshold=2
    )
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        completed_exercise_types=["practice_en_ru", "listening_word"],
    )

    recalculate_user_word_progress(user)
    item.refresh_from_db()

    assert item.correct_count == 2
    assert item.is_learned is True
    assert item.learned_at is not None


@pytest.mark.django_db
def test_create_word_draft_uses_translation_hint_and_marks_confirmation(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1007, username="tester")
    monkeypatch.setattr(
        "vocab.services.translate_to_ru", lambda text: "пример перевода"
    )

    draft = create_word_draft(
        user,
        "apple - яблоко",
        {
            "word": "apple",
            "translation": "яблоко",
            "transcription": "ap",
            "example": "An apple a day.",
            "part_of_speech": "noun",
        },
        translation_hint="яблоко",
    )

    assert draft.translation == "яблоко"
    assert draft.translation_confirmed is True
    assert draft.example_translation == "пример перевода"


@pytest.mark.django_db
def test_refresh_draft_language_data_updates_generated_fields(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1008, username="tester")
    draft = AddWordDraft.objects.create(
        user=user,
        source_text="apple",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        part_of_speech="noun",
    )
    monkeypatch.setattr(
        "vocab.services.generate_word_data",
        lambda word, part_hint=None, translation_hint=None: {
            "transcription": "/apple/",
            "example": "An apple a day.",
            "example_translation": "Яблоко в день.",
            "part_of_speech": "noun",
        },
    )

    updated = refresh_draft_language_data(draft, "яблоко")

    assert updated.translation_confirmed is True
    assert updated.transcription == "/apple/"
    assert updated.example == "An apple a day."
    assert updated.example_translation == "Яблоко в день."


@pytest.mark.django_db
def test_build_learning_question_returns_choice_payload(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1010, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )
    monkeypatch.setattr("vocab.services.random.choice", lambda values: values[0])
    monkeypatch.setattr("vocab.services.random.shuffle", lambda values: None)

    question = build_learning_question(user)

    assert question is not None
    assert question["kind"] == "choice"
    assert question["item"]["id"] == item.id
    assert question["answer_mode"] == "practice_en_ru"


@pytest.mark.django_db
def test_build_choice_question_reverse_mode_returns_word_options(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1011, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )
    VocabularyItem.objects.create(
        user=TelegramUser.objects.create(chat_id=1012, username="other"),
        word="pear",
        normalized_word="pear",
        translation="груша",
        transcription="",
        example="Example",
        example_translation="Пример",
    )
    monkeypatch.setattr("vocab.services.random.shuffle", lambda values: None)

    question = build_choice_question(user, "reverse")

    assert question is not None
    assert question["mode"] == "reverse"
    assert question["item"]["id"] == item.id
    assert "apple" in question["options"]


@pytest.mark.django_db
def test_build_listening_and_speaking_questions_return_first_candidate():
    user = TelegramUser.objects.create(chat_id=1013, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    listening = build_listening_question(user, "word")
    speaking = build_speaking_question(user)

    assert listening is not None
    assert listening["item"]["id"] == item.id
    assert speaking is not None
    assert speaking["item"]["id"] == item.id


@pytest.mark.django_db
def test_submit_choice_answer_updates_practice_metric():
    user = TelegramUser.objects.create(chat_id=1014, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = submit_choice_answer(user, item.id, "яблоко", "classic")
    item.refresh_from_db()
    progress = get_or_create_user_course_progress(user)

    assert result["correct"] is True
    assert progress.practice_correct == 1
    assert "practice_en_ru" in item.completed_exercise_types


@pytest.mark.django_db
def test_submit_choice_answer_review_mode_resets_progress_on_wrong_answer():
    user = TelegramUser.objects.create(chat_id=1015, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        is_learned=True,
        correct_count=2,
        completed_exercise_types=["practice_en_ru", "listening_word"],
    )

    result = submit_choice_answer(user, item.id, "неверно", "review")
    item.refresh_from_db()

    assert result["correct"] is False
    assert item.is_learned is False
    assert item.correct_count == 0


@pytest.mark.django_db
def test_submit_listening_answer_accepts_single_typo():
    user = TelegramUser.objects.create(chat_id=1016, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = submit_listening_answer(user, item.id, "appl", "word")
    progress = get_or_create_user_course_progress(user)

    assert result["correct"] is True
    assert result["accepted_with_typo"] is True
    assert progress.listening_correct == 1


@pytest.mark.django_db
def test_submit_learning_text_answer_reverse_mode_accepts_typo():
    user = TelegramUser.objects.create(chat_id=1017, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = submit_learning_text_answer(user, item.id, "appl", "practice_ru_en")
    progress = get_or_create_user_course_progress(user)

    assert result["correct"] is True
    assert result["accepted_with_typo"] is True
    assert progress.practice_correct == 1


@pytest.mark.django_db
def test_evaluate_speaking_answer_returns_close_for_similar_transcript():
    user = TelegramUser.objects.create(chat_id=1018, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = evaluate_speaking_answer(user, item.id, "appl")

    assert result["status"] == "close"
    assert result["correct_answer"] == "apple"


@pytest.mark.django_db
def test_evaluate_speaking_answer_marks_correct_attempt():
    user = TelegramUser.objects.create(chat_id=1019, username="tester")
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    result = evaluate_speaking_answer(user, item.id, "apple")
    item.refresh_from_db()
    progress = get_or_create_user_course_progress(user)

    assert result["status"] == "correct"
    assert progress.speaking_correct == 1
    assert "speaking" in item.completed_exercise_types


@pytest.mark.django_db
def test_request_word_image_generation_reuses_shared_image_path():
    user = TelegramUser.objects.create(chat_id=1020, username="tester")
    VocabularyItem.objects.create(
        user=TelegramUser.objects.create(chat_id=1021, username="other"),
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        image_path="media/card_images/shared.jpg",
    )
    item = VocabularyItem.objects.create(
        user=user,
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
    )

    updated = request_word_image_generation(item)

    assert updated.image_path == "media/card_images/shared.jpg"
    assert updated.image_generation_in_progress is False


@pytest.mark.django_db
def test_request_draft_image_generation_reuses_shared_image_path():
    user = TelegramUser.objects.create(chat_id=1022, username="tester")
    VocabularyItem.objects.create(
        user=TelegramUser.objects.create(chat_id=1023, username="other"),
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        image_path="media/card_images/shared.jpg",
    )
    draft = AddWordDraft.objects.create(
        user=user,
        source_text="apple",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        translation_confirmed=True,
    )

    updated = request_draft_image_generation(draft)

    assert updated.image_path == "media/card_images/shared.jpg"
    assert updated.image_generation_in_progress is False


@pytest.mark.django_db
def test_ensure_draft_image_generates_path_when_prompt_available(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1024, username="tester")
    draft = AddWordDraft.objects.create(
        user=user,
        source_text="apple",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        translation_confirmed=True,
        example="An apple a day.",
        part_of_speech="noun",
    )
    monkeypatch.setattr(
        "vocab.services.build_visual_prompt", lambda *args: "visual prompt"
    )
    monkeypatch.setattr(
        "vocab.services.generate_card_image",
        lambda prompt, slug: "media/draft_images/generated.webp",
    )

    updated = ensure_draft_image(draft)

    assert updated.image_prompt == "visual prompt"
    assert updated.image_path == "media/draft_images/generated.webp"


@pytest.mark.django_db
def test_finalize_word_draft_creates_word_and_deletes_draft(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1025, username="tester")
    draft = AddWordDraft.objects.create(
        user=user,
        source_text="apple",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        translation_confirmed=True,
        transcription="",
        example="An apple a day.",
        example_translation="Яблоко в день.",
        part_of_speech="noun",
    )
    monkeypatch.setattr("vocab.services.translate_to_ru", lambda text: "Яблоко в день.")

    item = finalize_word_draft(draft, use_image=False)

    assert item.word == "apple"
    assert AddWordDraft.objects.filter(id=draft.id).exists() is False


@pytest.mark.django_db
def test_add_pack_words_to_user_creates_prepared_and_fallback_items(monkeypatch):
    user = TelegramUser.objects.create(chat_id=1026, username="tester")
    monkeypatch.setattr(
        "vocab.services.get_pack_level",
        lambda pack_id, level_id: {
            "items": [
                {"word": "apple", "translation": "яблоко"},
                {"word": "pear", "translation": "груша"},
            ]
        },
    )
    monkeypatch.setattr(
        "vocab.services.ensure_pack_preparation", lambda pack_id, level_id: None
    )
    monkeypatch.setattr(
        "vocab.services.request_word_image_generation", lambda item: item
    )
    monkeypatch.setattr(
        "vocab.services.generate_word_data_batch", lambda entries: [None]
    )
    monkeypatch.setattr("vocab.services.translate_to_ru", lambda text: "")

    from vocab.models import PackPreparedWord

    PackPreparedWord.objects.create(
        pack_id="pack",
        level_id="level",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="/apple/",
        example="An apple a day.",
        example_translation="Яблоко в день.",
        part_of_speech="noun",
        image_path="media/card_images/apple.jpg",
    )

    result = add_pack_words_to_user(user, "pack", "level", ["apple", "pear"])

    assert len(result["created"]) == 2
    assert result["skipped"] == []
    assert (
        VocabularyItem.objects.filter(user=user, normalized_word="apple").exists()
        is True
    )
    assert (
        VocabularyItem.objects.filter(user=user, normalized_word="pear").exists()
        is True
    )


@pytest.mark.django_db
def test_apply_user_settings_switches_active_studied_language():
    user = TelegramUser.objects.create(chat_id=1030, username="tester")

    apply_user_settings(user, {"active_studied_language": "ka"})
    user.refresh_from_db()

    assert get_active_course_code(user) == "ka"
    assert UserCourseProgress.objects.filter(user=user, course_code="ka").exists()


@pytest.mark.django_db
def test_build_user_progress_is_isolated_per_course():
    user = TelegramUser.objects.create(chat_id=1031, username="tester")
    VocabularyItem.objects.create(
        user=user,
        course_code="en",
        word="apple",
        normalized_word="apple",
        translation="яблоко",
        transcription="",
        example="Example",
        example_translation="Пример",
        is_learned=True,
        correct_count=2,
        completed_exercise_types=["practice_en_ru", "listening_word"],
    )
    VocabularyItem.objects.create(
        user=user,
        course_code="ka",
        word="gamarjoba",
        normalized_word="gamarjoba",
        translation="привет",
        transcription="",
        example="Gamarjoba!",
        example_translation="Привет!",
        is_learned=False,
    )
    progress_en = get_or_create_user_course_progress(user, "en")
    progress_en.practice_correct = 2
    progress_en.save(update_fields=["practice_correct"])

    apply_user_settings(user, {"active_studied_language": "ka"})
    payload_ka = build_user_progress(user)

    assert payload_ka["course_code"] == "ka"
    assert payload_ka["total"] == 1
    assert payload_ka["learned"] == 0
    assert payload_ka["practice_correct"] == 0

    apply_user_settings(user, {"active_studied_language": "en"})
    payload_en = build_user_progress(user)

    assert payload_en["course_code"] == "en"
    assert payload_en["total"] == 1
    assert payload_en["learned"] == 1
    assert payload_en["practice_correct"] == 2


@pytest.mark.django_db
def test_list_word_packs_returns_georgian_starter_for_ka_course():
    user = TelegramUser.objects.create(
        chat_id=1032, username="tester", active_studied_language="ka"
    )

    packs = list_word_packs(user)

    assert len(packs) == 1
    assert packs[0]["id"] == "georgian_starter"
    assert packs[0]["levels"][0]["id"] == "starter"
    assert len(packs[0]["levels"][0]["items"]) == 10
