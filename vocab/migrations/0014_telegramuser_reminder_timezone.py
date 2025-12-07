from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0013_irregularverbprogress"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="reminder_timezone",
            field=models.CharField(default="UTC", max_length=50),
        ),
    ]
