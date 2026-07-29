"""Quarantine legacy non-Telegram identities before enforcing chat ID integrity."""

from django.db import migrations, models
from django.db.models import F, Q

# Telegram user IDs are many orders of magnitude below this reserved range.
# Keeping rows instead of deleting them preserves all related user data.
LEGACY_QUARANTINE_CHAT_ID_BASE = 9_000_000_000_000_000_000


def quarantine_legacy_identities(apps, schema_editor):
    TelegramUser = apps.get_model("vocab", "TelegramUser")
    TelegramUser.objects.filter(chat_id__lte=0).update(
        chat_id=F("id") + LEGACY_QUARANTINE_CHAT_ID_BASE,
        legacy_identity_quarantined=True,
    )


class Migration(migrations.Migration):
    dependencies = [("vocab", "0049_product_event_analytics")]

    operations = [
        migrations.AddField(
            model_name="telegramuser",
            name="legacy_identity_quarantined",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(
            quarantine_legacy_identities,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name="telegramuser",
            constraint=models.CheckConstraint(
                condition=Q(("chat_id__gt", 0)),
                name="telegram_user_chat_id_positive",
            ),
        ),
    ]
