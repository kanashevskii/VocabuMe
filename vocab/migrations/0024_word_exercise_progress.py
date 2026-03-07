from django.db import migrations, models


EXERCISE_PRIORITY = [
    "practice_en_ru",
    "listening_word",
    "practice_ru_en",
    "speaking",
    "listening_translate",
]


def forwards(apps, schema_editor):
    TelegramUser = apps.get_model("vocab", "TelegramUser")
    VocabularyItem = apps.get_model("vocab", "VocabularyItem")

    TelegramUser.objects.all().update(repeat_threshold=4)

    for item in VocabularyItem.objects.select_related("user").all().iterator():
        goal = 4
        completed_count = max(0, min(int(item.correct_count or 0), len(EXERCISE_PRIORITY)))
        if item.is_learned:
            completed_count = max(completed_count, goal)
        completed_types = EXERCISE_PRIORITY[:completed_count]
        item.completed_exercise_types = completed_types
        item.correct_count = len(completed_types)
        item.is_learned = len(completed_types) >= goal
        if not item.is_learned:
            item.learned_at = None
        item.save(update_fields=["completed_exercise_types", "correct_count", "is_learned", "learned_at", "updated_at"])


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0023_async_image_generation_fields"),
    ]

    operations = [
        migrations.AlterField(
            model_name="telegramuser",
            name="repeat_threshold",
            field=models.PositiveIntegerField(default=4),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="session_question_limit",
            field=models.PositiveIntegerField(default=12),
        ),
        migrations.AddField(
            model_name="vocabularyitem",
            name="completed_exercise_types",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
