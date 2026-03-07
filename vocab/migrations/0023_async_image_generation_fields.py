from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0022_image_regeneration_counts"),
    ]

    operations = [
        migrations.AddField(
            model_name="addworddraft",
            name="image_generation_in_progress",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="addworddraft",
            name="image_generation_version",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="vocabularyitem",
            name="image_generation_in_progress",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="vocabularyitem",
            name="image_generation_version",
            field=models.PositiveIntegerField(default=0),
        ),
    ]
