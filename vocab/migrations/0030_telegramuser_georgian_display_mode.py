from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vocab", "0029_repair_georgian_examples"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="georgian_display_mode",
            field=models.CharField(
                choices=[("both", "Georgian + Latin"), ("native", "Georgian only")],
                default="both",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="has_selected_georgian_display_mode",
            field=models.BooleanField(default=False),
        ),
    ]
