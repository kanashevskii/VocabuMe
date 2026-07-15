from datetime import time

from vocab.integrations.telegram.settings_ui import (
    main_settings_text,
    review_settings_keyboard,
)
from vocab.models import TelegramUser


def test_review_settings_keyboard_has_one_callback_per_review_interval():
    user = TelegramUser(enable_review_old_words=True, days_before_review=30)

    callbacks = [
        button.callback_data for row in review_settings_keyboard(user) for button in row
    ]

    assert callbacks.count("set_review_days_7") == 1
    assert {"set_review_days_30", "set_review_days_90"}.issubset(callbacks)


def test_main_settings_text_includes_user_configuration():
    user = TelegramUser(
        repeat_threshold=3,
        enable_review_old_words=False,
        reminder_enabled=True,
        reminder_interval_days=2,
        reminder_time=time(9, 30),
        reminder_timezone="UTC",
    )

    text = main_settings_text(user)

    assert "*3*" in text
    assert "выключено" in text
    assert "включены" in text
    assert "09:30" in text
