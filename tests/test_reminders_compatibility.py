from vocab import reminders


def test_legacy_reminder_entrypoint_uses_canonical_command(monkeypatch):
    calls = []

    monkeypatch.setattr(reminders, "call_command", lambda name: calls.append(name))

    reminders.send_reminders()

    assert calls == ["send_reminders"]
