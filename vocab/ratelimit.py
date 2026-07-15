"""Small cache-backed request limiter.

The production cache must be a shared Redis-compatible backend.  Django's local
memory cache is deliberately only a development fallback and is not suitable
for a multi-worker deployment.
"""

from __future__ import annotations

import hashlib

from django.core.cache import cache


class RateLimitExceeded(Exception):
    """Raised when a request exceeds its fixed-window quota."""


def enforce_rate_limit(*, scope: str, subject: str, limit: int, window: int) -> None:
    """Atomically count a subject in a cache fixed window where supported."""
    if limit <= 0 or window <= 0:
        raise ValueError("Rate-limit limit and window must be positive.")

    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    key = f"ratelimit:{scope}:{digest}"
    if cache.add(key, 1, timeout=window):
        return

    try:
        count = cache.incr(key)
    except ValueError:
        # The key expired between add() and incr(); retry once as a new window.
        if cache.add(key, 1, timeout=window):
            return
        count = cache.incr(key)
    if count > limit:
        raise RateLimitExceeded
