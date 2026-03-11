from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0037_telegramuser_temporary_practice_pauses"),
    ]

    operations = [
        migrations.AddField(
            model_name="packpreparedword",
            name="failure_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="packpreparedword",
            name="last_failure_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
