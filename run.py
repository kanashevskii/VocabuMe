import os
import threading
import time
import logging

from decouple import config
from django.core.management import call_command, execute_from_command_line
from telegram import Bot

from core.logging_config import setup_logging

setup_logging()

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

import django
django.setup()

from vocab.bot import run_telegram_bot


def send_alert(message: str):
    chat_id = config("ALERT_CHAT_ID", default=None)
    token = config("TELEGRAM_TOKEN", default=None)
    if not chat_id or not token:
        return
    try:
        Bot(token=token).send_message(chat_id=chat_id, text=message)
    except Exception as e:
        logging.exception("Failed to send alert: %s", e)


def reminder_loop():
    while True:
        try:
            call_command("send_reminders")
        except Exception as e:
            logging.exception("send_reminders failed: %s", e)
            send_alert(f"send_reminders failed: {e}")
        time.sleep(60)


def run_server():
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])


def main():
    bot_thread = threading.Thread(target=run_telegram_bot)
    bot_thread.start()

    reminder_thread = threading.Thread(target=reminder_loop, daemon=True)
    reminder_thread.start()

    run_server()


if __name__ == '__main__':
    main()
