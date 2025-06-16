import os
import threading
import django
from django.core.management import execute_from_command_line
from vocab.bot import run_telegram_bot

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()


def run_server():
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])


def main():
    bot_thread = threading.Thread(target=run_telegram_bot)
    bot_thread.start()
    run_server()


if __name__ == '__main__':
    main()
