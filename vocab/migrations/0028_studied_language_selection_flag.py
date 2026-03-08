from django.db import migrations, models


def mark_existing_users_as_selected(apps, schema_editor):
    TelegramUser = apps.get_model("vocab", "TelegramUser")
    TelegramUser.objects.all().update(has_selected_studied_language=True)


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0027_course_tracks"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="has_selected_studied_language",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(
            mark_existing_users_as_selected, migrations.RunPython.noop
        ),
    ]
