import os
import sys
import threading
import time
import logging
import atexit
import signal

from decouple import config
from django.core.management import call_command, execute_from_command_line
from telegram import Bot

from core.logging_config import setup_logging

LOCK_FILE = "/tmp/englishbot.lock"
_shutdown_event = threading.Event()
_cleanup_done = threading.Lock()
_cleanup_called = False


def ensure_single_instance():
    """
    Prevent running multiple local instances that conflict on Telegram getUpdates.
    Creates a pidfile and exits if another live process holds it.
    """
    # Django autoreload spawns a child with RUN_MAIN=true; skip lock there.
    if os.environ.get("RUN_MAIN") == "true":
        return

    pid = os.getpid()
    # Try to acquire, stopping another local instance if needed
    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(pid).encode())
            os.close(fd)
            break
        except FileExistsError:
            other_pid = None
            try:
                with open(LOCK_FILE) as f:
                    other_pid = int(f.read().strip() or 0)
            except Exception:
                pass

            if not other_pid:
                os.remove(LOCK_FILE)
                continue

            try:
                os.kill(other_pid, 0)
            except ProcessLookupError:
                # stale lock, remove and retry
                os.remove(LOCK_FILE)
                continue

            # Another live process found: try to stop it gracefully
            print(f"Another englishbot instance is running (PID {other_pid}). Sending SIGTERM...")
            try:
                os.kill(other_pid, signal.SIGTERM)
            except Exception:
                pass

            process_stopped = False
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(other_pid, 0)
                except ProcessLookupError:
                    # stopped, remove old lock and retry acquiring
                    try:
                        os.remove(LOCK_FILE)
                    except FileNotFoundError:
                        pass
                    process_stopped = True
                    break
            
            if process_stopped:
                continue
            
            # Process didn't stop with SIGTERM, try SIGKILL
            print(f"Instance PID {other_pid} still running. Sending SIGKILL...")
            try:
                os.kill(other_pid, signal.SIGKILL)
                time.sleep(0.5)
            except Exception:
                pass

            try:
                os.kill(other_pid, 0)
            except ProcessLookupError:
                # Process stopped after SIGKILL, remove lock and retry
                try:
                    os.remove(LOCK_FILE)
                except FileNotFoundError:
                    pass
                continue

            # Process is still running after SIGKILL
            print(f"Instance PID {other_pid} still running. Stop it manually: kill {other_pid}")
            sys.exit(1)

    def _cleanup():
        """Remove lock file. Should only be called once."""
        global _cleanup_called
        with _cleanup_done:
            if _cleanup_called:
                return
            _cleanup_called = True
        
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass

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
