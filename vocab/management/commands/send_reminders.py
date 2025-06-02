from django.core.management.base import BaseCommand
from django.utils.timezone import now
from vocab.models import TelegramUser
from telegram import Bot
from decouple import config

class Command(BaseCommand):
    help = "Send learning reminders to users"

    def handle(self, *args, **kwargs):
        bot = Bot(token=config("TELEGRAM_TOKEN"))
        today = now().date()
        current_time = now().time()

        self.stdout.write(f"⏰ Запуск напоминаний: {now().strftime('%Y-%m-%d %H:%M')}")

        users = TelegramUser.objects.filter(reminder_enabled=True)

        for user in users:
            if user.reminder_time and current_time < user.reminder_time:
                self.stdout.write(f"⏳ Ещё не время для {user.chat_id}")
                continue

            if user.last_reminder_sent_at:
                days_since_last = (today - user.last_reminder_sent_at).days
                if days_since_last < user.reminder_interval_days:
                    self.stdout.write(f"⏭ Пропущен {user.chat_id} — уже было {user.last_reminder_sent_at}")
                    continue

            try:
                bot.send_message(
                    chat_id=user.chat_id,
                    text="🕒 Время для повторения слов! Напиши /learn, чтобы продолжить обучение."
                )
                user.last_reminder_sent_at = today
                user.save()
                self.stdout.write(self.style.SUCCESS(f"✅ Напоминание отправлено {user.chat_id}"))
            except Exception as e:
                self.stderr.write(f"❌ Ошибка отправки {user.chat_id}: {e}")
