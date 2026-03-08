import asyncio
from asgiref.sync import sync_to_async
from django.core.management.base import BaseCommand
from django.utils.timezone import now
from vocab.models import TelegramUser
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from vocab.utils import timezone_from_name

from core.env import get_telegram_token, get_webapp_url

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
        bot = Bot(token=get_telegram_token())
        webapp_url = get_webapp_url()
        self.stdout.write(f"⏰ Запуск напоминаний: {now().strftime('%Y-%m-%d %H:%M')}")

        users = await self._get_users()

        for user in users:
            user_tz = timezone_from_name(getattr(user, "reminder_timezone", "UTC"))
            current_local = now().astimezone(user_tz)
            today_local = current_local.date()

            if user.reminder_time and current_local.time() < user.reminder_time:
                self.stdout.write(f"⏳ Ещё не время для {user.chat_id} ({user.reminder_time} {user.reminder_timezone})")
                continue

            if user.last_reminder_sent_at:
                days_since_last = (today_local - user.last_reminder_sent_at).days
                if days_since_last < user.reminder_interval_days:
                    self.stdout.write(f"⏭ Пропущен {user.chat_id} — уже было {user.last_reminder_sent_at}")
                    continue

            try:
                reply_markup = None
                if webapp_url:
                    reply_markup = InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Открыть VocabuMe", web_app=WebAppInfo(url=webapp_url))]]
                    )
                await bot.send_message(
                    chat_id=user.chat_id,
                    text="🕒 Пора продолжить занятие. Открой VocabuMe и пройди короткую практику.",
                    reply_markup=reply_markup,
                )
                await self._mark_sent(user, today_local)
                self.stdout.write(self.style.SUCCESS(f"✅ Напоминание отправлено {user.chat_id}"))
            except Exception as e:
                self.stderr.write(f"❌ Ошибка отправки {user.chat_id}: {e}")
