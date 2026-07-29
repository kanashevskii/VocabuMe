"""Process-local state for legacy Telegram bot practice flows.

The Mini App uses durable server-side question state.  The legacy bot still has
short-lived interactive flows; keeping their transient queues in one module
avoids accidental state forks while its handlers are split into focused files.
"""

from __future__ import annotations

from typing import Any

user_lessons: dict[object, list[Any]] = {}
