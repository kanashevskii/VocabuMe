import asyncio
from asgiref.sync import sync_to_async
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from vocab.models import TelegramUser
from telegram import Bot
from decouple import config

class Command(BaseCommand):
    help = "Send learning reminders to users"

    def handle(self, *args, **kwargs):
        asyncio.run(self._async_handle())

    @sync_to_async
    def _get_users(self):
        # Fetch in a thread to avoid SynchronousOnlyOperation inside asyncio.run
        return list(TelegramUser.objects.filter(reminder_enabled=True))

    @sync_to_async
    def _mark_sent(self, user, today):
        user.last_reminder_sent_at = today
        user.save(update_fields=["last_reminder_sent_at"])

    async def _async_handle(self):
        bot = Bot(token=config("TELEGRAM_TOKEN"))
        today = now().date()
        current_time = now().time()

        self.stdout.write(f"⏰ Запуск напоминаний: {now().strftime('%Y-%m-%d %H:%M')}")

        users = await self._get_users()

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
                await bot.send_message(
                    chat_id=user.chat_id,
                    text="🕒 Время для повторения слов! Напиши /learn, чтобы продолжить обучение."
                )
                await self._mark_sent(user, today)
                self.stdout.write(self.style.SUCCESS(f"✅ Напоминание отправлено {user.chat_id}"))
            except Exception as e:
                self.stderr.write(f"❌ Ошибка отправки {user.chat_id}: {e}")
