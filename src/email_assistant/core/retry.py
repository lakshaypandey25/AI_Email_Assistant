"""Async retry decorator with exponential backoff, jitter and ``Retry-After`` support."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import structlog

P = ParamSpec("P")
T = TypeVar("T")

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff configuration.

    Attributes:
        max_attempts: Total attempts including the first call.
        base_delay: Delay before the 2nd attempt (seconds); doubles each retry.
        max_delay: Upper bound for a single computed backoff delay.
        jitter: Apply "equal jitter" (delay/2 + U(0, delay/2)) to avoid thundering herds.
        max_retry_after: If the server asks us to wait longer than this, give up immediately
            instead of blocking a worker (e.g. exhausted daily quota).
        sleep: Awaitable sleep function - injectable so tests run instantly.
    """

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    max_retry_after: float = 90.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def compute_backoff(
    policy: RetryPolicy,
    attempt: int,
    retry_after: float | None = None,
    rand: Callable[[], float] = random.random,
) -> float:
    """Return the delay (seconds) to wait after the ``attempt``-th failure (1-based)."""
    exp = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
    delay = exp / 2 + rand() * exp / 2 if policy.jitter else exp
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay


def retry_async(
    policy: RetryPolicy,
    *,
    is_retryable: Callable[[BaseException], bool],
    get_retry_after: Callable[[BaseException], float | None] | None = None,
    operation: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorate an ``async def`` so transient failures are retried with backoff.

    Args:
        policy: Backoff parameters.
        is_retryable: Predicate deciding whether an exception is transient.
        get_retry_after: Optional extractor of a server-provided wait time.
        operation: Label used in logs (defaults to the function name).
    """

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        label = operation or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return await fn(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not is_retryable(exc) or attempt >= policy.max_attempts:
                        raise
                    retry_after = get_retry_after(exc) if get_retry_after else None
                    if retry_after is not None and retry_after > policy.max_retry_after:
                        logger.error(
                            "retry_abandoned_long_wait",
                            operation=label,
                            retry_after=retry_after,
                            error_type=type(exc).__name__,
                        )
                        raise
                    delay = compute_backoff(policy, attempt, retry_after)
                    logger.warning(
                        "retrying_after_error",
                        operation=label,
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        delay_seconds=round(delay, 2),
                        error_type=type(exc).__name__,
                        error=str(exc)[:200],
                    )
                    await policy.sleep(delay)

        return wrapper

    return decorator
