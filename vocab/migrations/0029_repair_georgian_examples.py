import logging
import re

from django.db import migrations


logger = logging.getLogger(__name__)
GEORGIAN_PATTERN = re.compile(r"[\u10A0-\u10FF]")


def _example_is_invalid(example: str) -> bool:
    text = (example or "").strip()
    return bool(text) and not GEORGIAN_PATTERN.search(text)


def repair_georgian_examples(apps, schema_editor):
    from vocab.openai_utils import generate_word_data
    from vocab.utils import translate_to_ru

    VocabularyItem = apps.get_model("vocab", "VocabularyItem")
    PackPreparedWord = apps.get_model("vocab", "PackPreparedWord")
    AddWordDraft = apps.get_model("vocab", "AddWordDraft")

    targets = (
        VocabularyItem.objects.filter(course_code="ka").exclude(example=""),
        PackPreparedWord.objects.filter(course_code="ka").exclude(example=""),
        AddWordDraft.objects.filter(course_code="ka").exclude(example=""),
    )

    for queryset in targets:
        for item in queryset.iterator():
            if not _example_is_invalid(item.example):
                continue
            try:
                generated = generate_word_data(
                    item.word,
                    part_hint=getattr(item, "part_of_speech", None),
                    translation_hint=item.translation,
                    course_code="ka",
                )
            except Exception:
                logger.exception(
                    "Failed to repair Georgian example for %s(%s)",
                    item.__class__.__name__,
                    item.pk,
                )
                continue

            if not generated:
                continue

            example = (generated.get("example") or "").strip()
            if not example or _example_is_invalid(example):
                continue

            item.translation = (generated.get("translation") or item.translation).strip()
            item.transcription = (generated.get("transcription") or item.transcription).strip()
            item.example = example
            item.example_translation = (
                generated.get("example_translation") or translate_to_ru(example)
            ).strip()
            item.part_of_speech = (
                generated.get("part_of_speech") or item.part_of_speech
            ).strip()

            update_fields = [
                "translation",
                "transcription",
                "example",
                "example_translation",
                "part_of_speech",
            ]
            if hasattr(item, "prepared_at"):
                update_fields.append("prepared_at")
            if hasattr(item, "updated_at"):
                update_fields.append("updated_at")
            item.save(update_fields=update_fields)


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0028_studied_language_selection_flag"),
    ]

    operations = [
        migrations.RunPython(
            repair_georgian_examples, migrations.RunPython.noop
        ),
    ]
