from datetime import time

import pytest

from vocab import bot
from vocab.integrations.telegram import settings_handlers


def test_bot_keeps_settings_handler_compatibility_imports():
    assert bot.settings is settings_handlers.settings
    assert bot.handle_settings_callback is settings_handlers.handle_settings_callback
    assert bot.set_reminder_time is settings_handlers.set_reminder_time
    assert bot.set_reminder_timezone is settings_handlers.set_reminder_timezone


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("08:30", time(8, 30)), ("0830", time(8, 30)), ("08.30", time(8, 30))],
)
def test_parse_reminder_time_accepts_existing_input_variants(raw_value, expected):
    assert settings_handlers.parse_reminder_time(raw_value) == expected


def test_parse_reminder_time_rejects_invalid_input():
    with pytest.raises(ValueError):
        settings_handlers.parse_reminder_time("soon")
