"""Server-authoritative irregular-verb question lifecycle."""

from __future__ import annotations

import random
from datetime import timedelta
from uuid import UUID

from django.db import transaction
from django.utils import timezone

from vocab.irregular_verbs import IRREGULAR_VERBS, get_random_pairs
from vocab.models import IssuedIrregularQuestion, TelegramUser
from vocab.services import (
    build_user_progress,
    get_active_course_code,
    update_irregular_progress,
    update_learning_streak,
)

QUESTION_TTL = timedelta(minutes=15)


def issue_irregular_question(user: TelegramUser) -> dict[str, object]:
    """Create an attempt and return only the data needed to render it."""
    verb = random.choice(IRREGULAR_VERBS)
    correct_pair = f"{verb['past']} {verb['participle']}"
    options: list[str] = []
    for candidate in [correct_pair, *verb["wrong_pairs"], *get_random_pairs(verb, 2)]:
        if candidate not in options:
            options.append(candidate)
    random.shuffle(options)
    issued = IssuedIrregularQuestion.objects.create(
        user=user,
        course_code=get_active_course_code(user),
        verb_base=verb["base"],
        expires_at=timezone.now() + QUESTION_TTL,
    )
    return {
        "verb": {"base": verb["base"]},
        "options": options[:4],
        "question_id": str(issued.id),
    }


def submit_issued_irregular_answer(
    user: TelegramUser, question_id: str, answer: str
) -> dict[str, object]:
    try:
        parsed_question_id = UUID(question_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("Irregular question was not found.") from exc

    with transaction.atomic():
        issued = (
            IssuedIrregularQuestion.objects.select_for_update()
            .filter(id=parsed_question_id, user=user)
            .first()
        )
        if issued is None:
            raise ValueError("Irregular question was not found.")
        if issued.answered_at is not None:
            raise ValueError("Irregular question was already answered.")
        if issued.expires_at <= timezone.now():
            raise ValueError("Irregular question has expired.")
        if issued.course_code != get_active_course_code(user):
            raise ValueError("Irregular question belongs to another course.")

        verb = next(
            (item for item in IRREGULAR_VERBS if item["base"] == issued.verb_base), None
        )
        if verb is None:  # Defensive: dataset changes must not grant points.
            raise ValueError("Irregular verb was not found.")
        correct_pair = f"{verb['past']} {verb['participle']}"
        correct = answer.strip() == correct_pair
        issued.answered_at = timezone.now()
        issued.save(update_fields=["answered_at"])
        if correct:
            update_irregular_progress(user, issued.verb_base, True)
            update_learning_streak(user)

    return {
        "correct": correct,
        "correct_answer": correct_pair,
        "points_earned": 1 if correct else 0,
        "progress": build_user_progress(user),
    }
