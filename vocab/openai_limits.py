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
from contextvars import ContextVar
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
OPENAI_CIRCUIT_FAILURE_THRESHOLD = env(
    "OPENAI_CIRCUIT_FAILURE_THRESHOLD", cast=int, default=5
)
OPENAI_CIRCUIT_RESET_SECONDS = env("OPENAI_CIRCUIT_RESET_SECONDS", cast=int, default=60)

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

_current_user_id: ContextVar[int | None] = ContextVar(
    "openai_current_user_id", default=None
)


class OpenAIConcurrencyExceeded(RuntimeError):
    """Raised when OpenAI capacity is exhausted or its limiter is unavailable."""


@contextmanager
def openai_user_scope(user_id: int | None) -> Iterator[None]:
    """Attach a trusted application user to nested OpenAI client calls."""
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)


def current_openai_user_id() -> int | None:
    return _current_user_id.get()


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


def _circuit_is_open(client: Redis) -> bool:
    try:
        failures = int(client.get("vocabume:openai:failures") or 0)
    except RedisError as exc:
        raise OpenAIConcurrencyExceeded("OpenAI is temporarily unavailable.") from exc
    return failures >= OPENAI_CIRCUIT_FAILURE_THRESHOLD


def _record_failure(client: Redis) -> None:
    try:
        key = "vocabume:openai:failures"
        client.incr(key)
        client.expire(key, OPENAI_CIRCUIT_RESET_SECONDS)
    except RedisError:
        logger.exception("Failed to record OpenAI circuit-breaker failure")


def _record_success(client: Redis) -> None:
    try:
        client.delete("vocabume:openai:failures")
    except RedisError:
        logger.exception("Failed to clear OpenAI circuit-breaker failures")


@contextmanager
def openai_slot(label: str, *, user_id: int | None = None) -> Iterator[None]:
    """Reserve global and optional per-user OpenAI capacity for one operation."""
    del label  # Labels are deliberately not part of Redis keys or user-visible errors.
    client = _redis_client()
    if client is None:
        yield
        return
    if _circuit_is_open(client):
        raise OpenAIConcurrencyExceeded(
            "OpenAI is temporarily unavailable. Please try again shortly."
        )

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
        try:
            yield
        except Exception:
            _record_failure(client)
            raise
        else:
            _record_success(client)
    finally:
        if acquired_user and user_key:
            _release(client, user_key)
        if acquired_global:
            _release(client, global_key)
