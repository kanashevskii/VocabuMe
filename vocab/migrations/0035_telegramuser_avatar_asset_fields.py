from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0034_usercourseprogress_total_points"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="avatar_path",
            field=models.CharField(blank=True, default="", max_length=500),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="avatar_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
