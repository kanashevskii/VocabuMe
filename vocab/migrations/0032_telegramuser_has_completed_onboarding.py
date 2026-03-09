from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0031_pack_translation_synonyms"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="has_completed_onboarding",
            field=models.BooleanField(default=False),
        ),
    ]
