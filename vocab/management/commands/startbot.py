from django.core.management.base import BaseCommand
from vocab.bot import run_telegram_bot

class Command(BaseCommand):
    help = "Запускает Telegram-бота"

    def handle(self, *args, **kwargs):
        run_telegram_bot()
