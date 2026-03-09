from __future__ import annotations

import re

from django.db import migrations


PACK_TRANSLATION_UPDATES = {
    "en": {
        "reservation": "бронь / бронирование",
        "check in": "зарегистрироваться / заселиться",
        "accommodation": "жилье / проживание",
        "currency exchange": "обмен валюты / обмен денег",
        "public transport": "общественный транспорт / транспорт",
        "travel insurance": "страховка для поездки / туристическая страховка",
        "layover": "пересадка / стыковка",
    },
    "ka": {
        "გამარჯობა": "привет / здравствуйте",
        "ნახვამდის": "пока / до свидания",
        "როგორ ხარ?": "как дела / как ты",
        "მადლობა": "спасибо / благодарю",
        "გთხოვ": "пожалуйста / прошу",
        "ბოდიში": "извините / простите",
        "არ მესმის": "я не понимаю / не понимаю",
        "რა ღირს?": "сколько стоит / сколько это стоит",
    },
}


def normalize_translation_value(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower().replace("ё", "е"))
    return normalized.strip(" \t\r\n.,;:!?\"'")


def extract_translation_parts(translation: str) -> list[str]:
    source = (translation or "").strip()
    if not source:
        return []

    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in source:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1

        if depth == 0 and char in {",", "/"}:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def merge_translation_variants(current: str, canonical: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()

    for value in (current, canonical):
        raw = (value or "").strip()
        if not raw:
            continue
        for part in extract_translation_parts(raw) or [raw]:
            normalized = normalize_translation_value(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)

    return " / ".join(merged)


def apply_pack_translation_synonyms(apps, schema_editor):
    VocabularyItem = apps.get_model("vocab", "VocabularyItem")
    PackPreparedWord = apps.get_model("vocab", "PackPreparedWord")

    for course_code, items in PACK_TRANSLATION_UPDATES.items():
        for word, canonical_translation in items.items():
            for model in (VocabularyItem, PackPreparedWord):
                queryset = model.objects.filter(course_code=course_code, word=word)
                for record in queryset.iterator():
                    merged = merge_translation_variants(
                        record.translation, canonical_translation
                    )
                    if merged and merged != record.translation:
                        record.translation = merged
                        record.save(update_fields=["translation"])


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0030_telegramuser_georgian_display_mode"),
    ]

    operations = [
        migrations.RunPython(
            apply_pack_translation_synonyms, migrations.RunPython.noop
        ),
    ]
