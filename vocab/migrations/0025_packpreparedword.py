from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0024_word_exercise_progress"),
    ]

    operations = [
        migrations.CreateModel(
            name="PackPreparedWord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("pack_id", models.CharField(max_length=64)),
                ("level_id", models.CharField(max_length=64)),
                ("word", models.CharField(max_length=255)),
                ("normalized_word", models.CharField(max_length=255)),
                ("translation", models.CharField(max_length=255)),
                ("transcription", models.CharField(blank=True, default="", max_length=255)),
                ("example", models.TextField(blank=True, default="")),
                ("example_translation", models.TextField(blank=True, default="")),
                ("part_of_speech", models.CharField(default="unknown", max_length=50)),
                ("image_path", models.CharField(blank=True, default="", max_length=500)),
                ("image_generation_in_progress", models.BooleanField(default=False)),
                ("prepared_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["pack_id", "level_id", "word"],
                "unique_together": {("pack_id", "level_id", "normalized_word")},
            },
        ),
    ]
