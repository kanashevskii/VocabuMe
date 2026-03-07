from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0016_weblogintoken"),
    ]

    operations = [
        migrations.CreateModel(
            name="AddWordDraft",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("source_text", models.CharField(max_length=255)),
                ("word", models.CharField(max_length=255)),
                ("normalized_word", models.CharField(max_length=255)),
                ("translation", models.CharField(blank=True, default="", max_length=255)),
                ("translation_confirmed", models.BooleanField(default=False)),
                ("transcription", models.CharField(blank=True, default="", max_length=255)),
                ("example", models.TextField(blank=True, default="")),
                ("example_translation", models.TextField(blank=True, default="")),
                ("part_of_speech", models.CharField(default="unknown", max_length=50)),
                ("image_prompt", models.TextField(blank=True, default="")),
                ("image_path", models.CharField(blank=True, default="", max_length=500)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("user", models.ForeignKey(on_delete=models.deletion.CASCADE, to="vocab.telegramuser")),
            ],
            options={"ordering": ["-updated_at", "-id"]},
        ),
    ]
