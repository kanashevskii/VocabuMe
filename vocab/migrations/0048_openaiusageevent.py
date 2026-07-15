from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("vocab", "0047_issuedirregularquestion")]

    operations = [
        migrations.CreateModel(
            name="OpenAIUsageEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("operation", models.CharField(max_length=64)),
                ("model", models.CharField(max_length=128)),
                ("input_tokens", models.PositiveIntegerField(default=0)),
                ("output_tokens", models.PositiveIntegerField(default=0)),
                ("cached_input_tokens", models.PositiveIntegerField(default=0)),
                ("image_count", models.PositiveSmallIntegerField(default=0)),
                ("cost_microusd", models.PositiveBigIntegerField()),
                ("usage_available", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="vocab.telegramuser")),
            ],
        ),
        migrations.AddIndex(
            model_name="openaiusageevent",
            index=models.Index(fields=["created_at"], name="vocab_openai_usage_created_idx"),
        ),
        migrations.AddIndex(
            model_name="openaiusageevent",
            index=models.Index(fields=["user", "created_at"], name="vocab_openai_usage_user_idx"),
        ),
    ]
