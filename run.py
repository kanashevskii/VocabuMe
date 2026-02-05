import os
import sys
import threading
import time
import logging
import atexit
import signal
from pathlib import Path

from decouple import config
from django.core.management import call_command, execute_from_command_line
from telegram import Bot

from core.logging_config import setup_logging

LOCK_FILE = "/tmp/englishbot.lock"
_shutdown_event = threading.Event()
_lock_handle = None


def ensure_single_instance():
    """
    Prevent running multiple local instances that conflict on Telegram getUpdates.
    Uses an OS-level lock on a lock file and exits if another instance holds it.
    """
    # Django autoreload spawns a child with RUN_MAIN=true; skip lock there.
    if os.environ.get("RUN_MAIN") == "true":
        return

    lock_path = Path(LOCK_FILE)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    global _lock_handle
    _lock_handle = open(lock_path, "a+", encoding="utf-8")

    try:
        import fcntl  # Unix
    except ImportError:
        fcntl = None

    if fcntl is not None:
        try:
            fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _lock_handle.seek(0)
            other_pid = (_lock_handle.read() or "").strip()
            msg = f"Another englishbot instance is already running (lock: {lock_path})."
            if other_pid:
                msg += f" PID in lock file: {other_pid}."
            print(msg)
            sys.exit(1)

    _lock_handle.seek(0)
    _lock_handle.truncate()
    _lock_handle.write(str(os.getpid()))
    _lock_handle.flush()

    def _cleanup():
        """Release the lock handle (idempotent)."""
        global _lock_handle
        if _lock_handle is None:
            return
        try:
            _lock_handle.close()
        except Exception:
            pass
        _lock_handle = None

    def _signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()
        # Call sys.exit() to terminate the process gracefully
        # The atexit handler will call _cleanup() which is now idempotent
        # This ensures cleanup happens even if the signal handler is called
        sys.exit(0)

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)


ensure_single_instance()

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
    while not _shutdown_event.is_set():
        try:
            call_command("send_reminders")
        except Exception as e:
            logging.exception("send_reminders failed: %s", e)
            send_alert(f"send_reminders failed: {e}")
        # Check shutdown event with timeout instead of blocking sleep
        if _shutdown_event.wait(timeout=60):
            break


def run_server():
    # Disable Django's autoreloader to avoid starting duplicate bot instances
    execute_from_command_line([
        'manage.py',
        'runserver',
        '--noreload',
        '0.0.0.0:8000',
    ])


def main():
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()

    reminder_thread = threading.Thread(target=reminder_loop, daemon=True)
    reminder_thread.start()

    try:
        run_server()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down...")
    finally:
        # Wait a moment for threads to finish
        logging.info("Waiting for threads to finish...")
        _shutdown_event.set()
        # Give threads a moment to clean up (they're daemon threads, so they'll be killed on exit)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
