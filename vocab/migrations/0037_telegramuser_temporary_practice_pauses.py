from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0036_subscriptionplan_paymentattempt_usersubscription"),
    ]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="listening_paused_until",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="telegramuser",
            name="speaking_paused_until",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
