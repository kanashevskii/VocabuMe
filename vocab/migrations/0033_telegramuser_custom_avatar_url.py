from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0032_telegramuser_has_completed_onboarding"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="custom_avatar_url",
            field=models.URLField(blank=True, default="", max_length=500),
        ),
    ]
