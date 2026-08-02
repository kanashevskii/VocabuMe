"""Pure normalization rules for course and learning-preference values."""

from __future__ import annotations

from vocab.models import (
    DEFAULT_STUDIED_LANGUAGE,
    DEFAULT_WORD_PRIORITY,
    STUDIED_LANGUAGE_CHOICES,
    WORD_PRIORITY_CHOICES,
)

SUPPORTED_COURSE_CODES = frozenset(code for code, _ in STUDIED_LANGUAGE_CHOICES)
SUPPORTED_WORD_PRIORITIES = frozenset(code for code, _ in WORD_PRIORITY_CHOICES)


def normalize_course_code(course_code: str | None) -> str:
    value = (course_code or DEFAULT_STUDIED_LANGUAGE).strip().lower()
    return value if value in SUPPORTED_COURSE_CODES else DEFAULT_STUDIED_LANGUAGE


def normalize_word_priority(value: str | None) -> str:
    normalized = (value or DEFAULT_WORD_PRIORITY).strip().lower()
    return (
        normalized if normalized in SUPPORTED_WORD_PRIORITIES else DEFAULT_WORD_PRIORITY
    )
