import pytest

from vocab.application.irregular_questions import (
    issue_irregular_question,
    submit_issued_irregular_answer,
)
from vocab.irregular_verbs import IRREGULAR_VERBS
from vocab.models import IssuedIrregularQuestion, TelegramUser, UserCourseProgress


@pytest.mark.django_db
def test_irregular_question_hides_answer_and_is_consumed_once(monkeypatch):
    user = TelegramUser.objects.create(chat_id=9911, username="irregular-test")
    verb = IRREGULAR_VERBS[0]
    monkeypatch.setattr(
        "vocab.application.irregular_questions.random.choice", lambda _items: verb
    )

    question = issue_irregular_question(user)

    assert question["verb"] == {"base": verb["base"]}
    assert "past" not in question["verb"]
    assert question["question_id"]
    result = submit_issued_irregular_answer(
        user, str(question["question_id"]), f"{verb['past']} {verb['participle']}"
    )

    assert result["correct"] is True
    issued = IssuedIrregularQuestion.objects.get(id=question["question_id"])
    assert issued.answered_at is not None
    assert UserCourseProgress.objects.get(user=user, course_code="en").irregular_correct == 1
    with pytest.raises(ValueError, match="already answered"):
        submit_issued_irregular_answer(
            user, str(question["question_id"]), f"{verb['past']} {verb['participle']}"
        )


@pytest.mark.django_db
def test_irregular_answer_requires_server_issued_question(client):
    user = TelegramUser.objects.create(chat_id=9912, username="irregular-test")
    session = client.session
    session["telegram_user_id"] = user.id
    session.save()

    response = client.post(
        "/api/irregular/answer",
        data='{"base": "be", "answer": "was/were been"}',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Irregular question was not found."
