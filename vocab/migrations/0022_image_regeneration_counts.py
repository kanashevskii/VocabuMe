from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0021_apperrorlog"),
    ]

    operations = [
        migrations.AddField(
            model_name="addworddraft",
            name="image_regeneration_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="vocabularyitem",
            name="image_regeneration_count",
            field=models.PositiveIntegerField(default=0),
        ),
    ]
