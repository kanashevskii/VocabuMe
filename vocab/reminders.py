"""Compatibility entrypoint for sending learning reminders.

The production scheduler invokes ``vocab.tasks.send_reminders`` which delegates
to the async-aware management command.  Keep this module for callers of the
legacy public function, but do not maintain a second sender implementation:
``python-telegram-bot`` message methods are coroutines and must be awaited by
the command's async execution path.
"""

from django.core.management import call_command


def send_reminders() -> None:
    """Run the canonical async-aware reminder command synchronously."""
    call_command("send_reminders")
