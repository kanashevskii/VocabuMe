import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("vocab", "0046_apperrorlog_created_at_index")]

    operations = [
        migrations.CreateModel(
            name="IssuedIrregularQuestion",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "course_code",
                    models.CharField(
                        choices=[("en", "English"), ("ka", "Georgian")],
                        default="en",
                        max_length=10,
                    ),
                ),
                ("verb_base", models.CharField(max_length=50)),
                ("expires_at", models.DateTimeField()),
                ("answered_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="vocab.telegramuser",
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="issuedirregularquestion",
            index=models.Index(
                fields=["user", "expires_at"], name="vocab_irregular_issue_idx"
            ),
        ),
    ]
