"""Read-only queries that select words for a learner's next session."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from django.utils.timezone import now

from vocab.domain.course_settings import normalize_course_code, normalize_word_priority
from vocab.models import TelegramUser, VocabularyItem


def _active_course_code(user: TelegramUser) -> str:
    return normalize_course_code(user.active_studied_language)


def ordered_new_words_queryset(
    user: TelegramUser,
    *,
    exclude_ids: Iterable[int] | None = None,
    part_of_speech: str | None = None,
):
    queryset = VocabularyItem.objects.filter(
        user=user, course_code=_active_course_code(user), is_learned=False
    ).exclude(id__in=list(exclude_ids or []))
    if part_of_speech:
        queryset = queryset.filter(part_of_speech=part_of_speech)
    if normalize_word_priority(user.word_priority) == "new_first":
        return queryset.order_by("-created_at", "-id")
    return queryset.order_by("created_at", "id")


def ordered_review_words_queryset(
    user: TelegramUser,
    *,
    exclude_ids: Iterable[int] | None = None,
    part_of_speech: str | None = None,
):
    if not user.enable_review_old_words:
        return VocabularyItem.objects.none()
    threshold = now() - timedelta(days=user.days_before_review)
    queryset = VocabularyItem.objects.filter(
        user=user,
        course_code=_active_course_code(user),
        is_learned=True,
        updated_at__lt=threshold,
    ).exclude(id__in=list(exclude_ids or []))
    if part_of_speech:
        queryset = queryset.filter(part_of_speech=part_of_speech)
    return queryset.order_by("updated_at", "id")


def get_priority_study_words(
    user: TelegramUser,
    *,
    count: int = 10,
    exclude_ids: Iterable[int] | None = None,
    part_of_speech: str | None = None,
) -> list[VocabularyItem]:
    excluded = list(exclude_ids or [])
    word_priority = normalize_word_priority(user.word_priority)
    new_words = list(
        ordered_new_words_queryset(
            user, exclude_ids=excluded, part_of_speech=part_of_speech
        )[:count]
    )
    if word_priority == "new_first":
        if len(new_words) >= count:
            return new_words[:count]
        seen = {item.id for item in new_words}
        review_words = list(
            ordered_review_words_queryset(
                user,
                exclude_ids=[*excluded, *seen],
                part_of_speech=part_of_speech,
            )[: max(0, count - len(new_words))]
        )
        return [*new_words, *review_words][:count]

    review_words = list(
        ordered_review_words_queryset(
            user, exclude_ids=excluded, part_of_speech=part_of_speech
        )[:count]
    )
    if len(review_words) >= count:
        return review_words[:count]
    seen = {item.id for item in review_words}
    new_tail = list(
        ordered_new_words_queryset(
            user,
            exclude_ids=[*excluded, *seen],
            part_of_speech=part_of_speech,
        )[: max(0, count - len(review_words))]
    )
    return [*review_words, *new_tail][:count]


def get_ordered_unlearned_words(
    user: TelegramUser,
    count: int = 10,
    exclude_ids: Iterable[int] | None = None,
) -> list[VocabularyItem]:
    return get_priority_study_words(user, count=count, exclude_ids=exclude_ids)


def get_unlearned_words(
    user: TelegramUser, count: int = 10, part_of_speech: str | None = None
) -> list[VocabularyItem]:
    return get_priority_study_words(user, count=count, part_of_speech=part_of_speech)


def get_learned_words(user: TelegramUser) -> list[VocabularyItem]:
    return list(
        VocabularyItem.objects.filter(
            user=user, course_code=_active_course_code(user), is_learned=True
        ).order_by("updated_at", "id")
    )
