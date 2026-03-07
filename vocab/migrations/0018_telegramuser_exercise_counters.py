from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0017_addworddraft"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="listening_correct",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="practice_correct",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="review_correct",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="speaking_correct",
            field=models.PositiveIntegerField(default=0),
        ),
    ]
