import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("vocab", "0041_paymentattempt_charge_idempotency")]

    operations = [
        migrations.CreateModel(
            name="IssuedLearningQuestion",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("exercise_type", models.CharField(max_length=40)),
                ("expires_at", models.DateTimeField()),
                ("answered_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("item", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="vocab.vocabularyitem")),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="vocab.telegramuser")),
            ],
        ),
        migrations.AddIndex(
            model_name="issuedlearningquestion",
            index=models.Index(fields=["user", "expires_at"], name="vocab_issue_user_id_d712f8_idx"),
        ),
    ]
