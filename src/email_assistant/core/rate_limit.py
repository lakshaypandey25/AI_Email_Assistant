"""Async token-bucket rate limiter (client-side throttling for Google and Groq quotas)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from collections import deque

class AsyncRateLimiter:
    """Token bucket: ``rate_per_sec`` sustained, ``burst`` instantaneous capacity."""

    def __init__(
        self,
        rate_per_sec: float,
        burst: int = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._rate = rate_per_sec
        self._capacity = float(max(1, burst))
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one token is available, then consume it."""
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
                result = self._sleep(wait)
                if asyncio.iscoroutine(result):
                    await result

class TokenWindow:
    """Sliding-window token budget (e.g. Groq tokens-per-minute) to avoid 429s *before* they happen.

    A budget of 0 disables the limiter.
    """

    def __init__(self, budget_per_minute: int, *, window: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._budget = budget_per_minute
        self._window = window
        self._clock = clock
        self._entries: deque[list[float]] = deque()

    def _prune(self, now: float) -> None:
        while self._entries and now - self._entries[0][0] >= self._window:
            self._entries.popleft()

    def wait_needed(self, estimate: int) -> float:
        if self._budget <= 0:
            return 0.0
        now = self._clock()
        self._prune(now)
        used = sum(e[1] for e in self._entries)
        excess = used + min(estimate, self._budget) - self._budget
        if excess <= 0:
            return 0.0
        freed = 0.0
        for stamp, tokens in self._entries:
            freed += tokens
            if freed >= excess:
                return max(0.0, stamp + self._window - now)
        return self._window

    def reserve(self, estimate: int) -> list[float]:
        entry = [self._clock(), float(estimate)]
        self._entries.append(entry)
        return entry