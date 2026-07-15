"""Redis-backed concurrency limits for paid OpenAI operations.

The limiter intentionally fails closed when Redis is configured but unavailable:
running an unbounded number of expensive requests is worse than returning a
temporary, retryable error. Local development without a Redis URL remains
usable and does not pretend to provide cross-process protection.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from redis import Redis
from redis.exceptions import RedisError

from core.env import env

logger = logging.getLogger(__name__)

OPENAI_REDIS_URL = env(
    "OPENAI_REDIS_URL", default=env("CELERY_BROKER_URL", default="")
)
OPENAI_MAX_CONCURRENCY = env("OPENAI_MAX_CONCURRENCY", cast=int, default=8)
OPENAI_MAX_INFLIGHT_PER_USER = env(
    "OPENAI_MAX_INFLIGHT_PER_USER", cast=int, default=1
)
OPENAI_SLOT_WAIT_SECONDS = env("OPENAI_SLOT_WAIT_SECONDS", cast=int, default=10)
OPENAI_SLOT_TTL_SECONDS = env("OPENAI_SLOT_TTL_SECONDS", cast=int, default=300)

_ACQUIRE_SEMAPHORE = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local maximum = tonumber(ARGV[1])
if current >= maximum then
    return 0
end
redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""

_RELEASE_SEMAPHORE = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current <= 1 then
    redis.call('DEL', KEYS[1])
else
    redis.call('DECR', KEYS[1])
end
return 1
"""


class OpenAIConcurrencyExceeded(RuntimeError):
    """Raised when OpenAI capacity is exhausted or its limiter is unavailable."""


@lru_cache(maxsize=1)
def _redis_client() -> Redis | None:
    if not OPENAI_REDIS_URL:
        return None
    return Redis.from_url(OPENAI_REDIS_URL, decode_responses=True)


def _try_acquire(client: Redis, key: str, maximum: int) -> bool:
    try:
        return bool(
            client.eval(
                _ACQUIRE_SEMAPHORE,
                1,
                key,
                maximum,
                OPENAI_SLOT_TTL_SECONDS,
            )
        )
    except RedisError as exc:
        raise OpenAIConcurrencyExceeded("OpenAI is temporarily unavailable.") from exc


def _release(client: Redis, key: str) -> None:
    try:
        client.eval(_RELEASE_SEMAPHORE, 1, key)
    except RedisError:
        logger.exception("Failed to release OpenAI concurrency slot key=%s", key)


@contextmanager
def openai_slot(label: str, *, user_id: int | None = None) -> Iterator[None]:
    """Reserve global and optional per-user OpenAI capacity for one operation."""
    del label  # Labels are deliberately not part of Redis keys or user-visible errors.
    client = _redis_client()
    if client is None:
        yield
        return

    global_key = "vocabume:openai:inflight"
    user_key = f"vocabume:openai:user:{user_id}:inflight" if user_id else None
    deadline = time.monotonic() + max(0, OPENAI_SLOT_WAIT_SECONDS)
    acquired_global = False
    acquired_user = False
    try:
        while True:
            acquired_global = _try_acquire(client, global_key, OPENAI_MAX_CONCURRENCY)
            if not acquired_global:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
                continue
            if user_key:
                acquired_user = _try_acquire(
                    client, user_key, OPENAI_MAX_INFLIGHT_PER_USER
                )
                if not acquired_user:
                    _release(client, global_key)
                    acquired_global = False
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.1)
                    continue
            break
        if not acquired_global or (user_key and not acquired_user):
            raise OpenAIConcurrencyExceeded(
                "OpenAI is busy. Please try again in a moment."
            )
        yield
    finally:
        if acquired_user and user_key:
            _release(client, user_key)
        if acquired_global:
            _release(client, global_key)
