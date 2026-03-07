from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0020_vocabularyitem_learned_at"),
    ]

    operations = [
        migrations.CreateModel(
            name="AppErrorLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("category", models.CharField(default="server", max_length=50)),
                ("level", models.CharField(default="error", max_length=20)),
                ("message", models.TextField()),
                ("path", models.CharField(blank=True, default="", max_length=255)),
                ("method", models.CharField(blank=True, default="", max_length=10)),
                ("status_code", models.PositiveIntegerField(blank=True, null=True)),
                ("context", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="vocab.telegramuser")),
            ],
            options={"ordering": ["-created_at", "-id"]},
        ),
    ]
