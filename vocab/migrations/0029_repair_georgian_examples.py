from django.db import migrations


STARTER_REPAIRS = {
    "გამარჯობა": {
        "translation": "привет",
        "transcription": "ɡɑmɑɾd͡ʒɔbɑ",
        "example": "გამარჯობა, როგორ ხარ?",
        "example_translation": "Привет, как дела?",
        "part_of_speech": "interjection",
    },
    "ნახვამდის": {
        "translation": "пока",
        "transcription": "nɑχvɑmdis",
        "example": "ნახვამდის, ხვალ გნახავ.",
        "example_translation": "Пока, увидимся завтра.",
        "part_of_speech": "interjection",
    },
    "როგორ ხარ?": {
        "translation": "как дела?",
        "transcription": "ɾɔɡɔɾ χɑɾ",
        "example": "როგორ ხარ დღეს?",
        "example_translation": "Как ты сегодня?",
        "part_of_speech": "phrase",
    },
    "მადლობა": {
        "translation": "спасибо",
        "transcription": "mɑdlɔbɑ",
        "example": "დიდი მადლობა დახმარებისთვის.",
        "example_translation": "Большое спасибо за помощь.",
        "part_of_speech": "interjection",
    },
    "გთხოვ": {
        "translation": "пожалуйста",
        "transcription": "ɡtχɔv",
        "example": "ერთი ყავა, გთხოვ.",
        "example_translation": "Один кофе, пожалуйста.",
        "part_of_speech": "interjection",
    },
    "დიახ": {
        "translation": "да",
        "transcription": "diɑχ",
        "example": "დიახ, მე მზად ვარ.",
        "example_translation": "Да, я готов.",
        "part_of_speech": "interjection",
    },
    "არა": {
        "translation": "нет",
        "transcription": "ɑɾɑ",
        "example": "არა, ეს არ მინდა.",
        "example_translation": "Нет, я этого не хочу.",
        "part_of_speech": "interjection",
    },
    "ბოდიში": {
        "translation": "извините",
        "transcription": "bɔdiʃi",
        "example": "ბოდიში, სად არის ბანკი?",
        "example_translation": "Извините, где банк?",
        "part_of_speech": "interjection",
    },
    "არ მესმის": {
        "translation": "я не понимаю",
        "transcription": "ɑɾ mɛsmis",
        "example": "ბოდიში, არ მესმის.",
        "example_translation": "Извините, я не понимаю.",
        "part_of_speech": "phrase",
    },
    "რა ღირს?": {
        "translation": "сколько стоит?",
        "transcription": "ɾɑ ɣiɾs",
        "example": "ეს რამდენი ღირს? რა ღირს?",
        "example_translation": "Сколько это стоит? Сколько стоит?",
        "part_of_speech": "phrase",
    },
}


def repair_starter_pack_examples(apps, schema_editor):
    VocabularyItem = apps.get_model("vocab", "VocabularyItem")
    PackPreparedWord = apps.get_model("vocab", "PackPreparedWord")
    AddWordDraft = apps.get_model("vocab", "AddWordDraft")

    models_to_repair = (VocabularyItem, PackPreparedWord, AddWordDraft)

    for model in models_to_repair:
        queryset = model.objects.filter(course_code="ka", word__in=STARTER_REPAIRS.keys())
        for item in queryset.iterator():
            repair = STARTER_REPAIRS.get(item.word)
            if not repair:
                continue
            item.translation = repair["translation"]
            item.transcription = repair["transcription"]
            item.example = repair["example"]
            item.example_translation = repair["example_translation"]
            item.part_of_speech = repair["part_of_speech"]
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
            repair_starter_pack_examples, migrations.RunPython.noop
        ),
    ]
