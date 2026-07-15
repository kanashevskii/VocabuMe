# Generated manually to make payment-provider idempotency explicit.

from django.db import migrations, models
from django.db.models import Q


class Migration(migrations.Migration):
    dependencies = [
        ("vocab", "0040_telegramuser_word_priority"),
    ]

    operations = [
        migrations.AddConstraint(
            model_name="paymentattempt",
            constraint=models.UniqueConstraint(
                fields=("telegram_payment_charge_id",),
                condition=~Q(telegram_payment_charge_id=""),
                name="unique_nonempty_telegram_payment_charge",
            ),
        ),
        migrations.AddConstraint(
            model_name="paymentattempt",
            constraint=models.UniqueConstraint(
                fields=("provider_payment_charge_id",),
                condition=~Q(provider_payment_charge_id=""),
                name="unique_nonempty_provider_payment_charge",
            ),
        ),
    ]
