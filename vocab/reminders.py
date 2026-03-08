from django.utils.timezone import now
from telegram import Bot
from .models import TelegramUser
import logging

from core.env import get_telegram_token

TELEGRAM_TOKEN = get_telegram_token()
bot = Bot(token=TELEGRAM_TOKEN)
logger = logging.getLogger(__name__)


def send_reminders():
    current_time = now()
    users = TelegramUser.objects.exclude(notification_time__isnull=True)

    for user in users:
        if not user.notification_time:
            continue

        # Проверка по времени
        should_remind = (
            current_time.hour == user.notification_time.hour
            and current_time.minute == user.notification_time.minute
        )

        if not should_remind:
            continue

        # Проверка интервала
        if user.last_notified_at:
            delta = current_time.date() - user.last_notified_at.date()
            if delta.days < user.notification_interval_days:
                continue

        try:
            bot.send_message(
                chat_id=user.chat_id,
                text="👋 Пора повторить слова! Запусти /learn и продолжим!",
            )
            user.last_notified_at = current_time
            user.save()
        except Exception:
            logger.exception("Reminder send failed for chat_id=%s", user.chat_id)
