from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("vocab", "0042_issuedlearningquestion")]

    operations = [
        migrations.CreateModel(
            name="BackgroundJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(max_length=64)),
                ("priority", models.PositiveSmallIntegerField(default=100)),
                ("deduplication_key", models.CharField(max_length=255, unique=True)),
                ("payload", models.JSONField(default=dict)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("running", "Running"), ("succeeded", "Succeeded"), ("failed", "Failed")], default="queued", max_length=16)),
                ("attempts", models.PositiveSmallIntegerField(default=0)),
                ("max_attempts", models.PositiveSmallIntegerField(default=3)),
                ("run_after", models.DateTimeField()),
                ("locked_at", models.DateTimeField(blank=True, null=True)),
                ("last_error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["priority", "run_after", "id"]},
        ),
        migrations.AddIndex(
            model_name="backgroundjob",
            index=models.Index(fields=["status", "priority", "run_after"], name="vocab_bgjob_queue_idx"),
        ),
    ]
