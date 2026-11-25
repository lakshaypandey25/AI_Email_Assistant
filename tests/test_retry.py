import pytest

from email_assistant.core.rate_limit import AsyncRateLimiter
from email_assistant.core.retry import RetryPolicy, compute_backoff, retry_async


class Transient(Exception):
    retry_after = None


def _policy(sleeps, **kw):
    async def fake_sleep(delay):
        sleeps.append(delay)

    return RetryPolicy(max_attempts=4, base_delay=1, max_delay=8, jitter=False, sleep=fake_sleep, **kw)


async def test_retries_with_exponential_backoff_then_succeeds():
    sleeps, calls = [], {"n": 0}

    @retry_async(_policy(sleeps), is_retryable=lambda e: isinstance(e, Transient))
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Transient("boom")
        return "ok"

    assert await flaky() == "ok"
    assert sleeps == [1, 2]


async def test_non_retryable_raises_immediately():
    sleeps = []

    @retry_async(_policy(sleeps), is_retryable=lambda e: isinstance(e, Transient))
    async def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await bad()
    assert sleeps == []


async def test_gives_up_after_max_attempts():
    sleeps = []

    @retry_async(_policy(sleeps), is_retryable=lambda e: True)
    async def always():
        raise Transient("x")

    with pytest.raises(Transient):
        await always()
    assert len(sleeps) == 3


async def test_long_retry_after_is_not_waited_out():
    sleeps = []

    class Quota(Transient):
        retry_after = 3600

    @retry_async(_policy(sleeps, max_retry_after=10), is_retryable=lambda e: True, get_retry_after=lambda e: e.retry_after)
    async def quota():
        raise Quota("daily quota")

    with pytest.raises(Quota):
        await quota()
    assert sleeps == []


def test_backoff_honours_retry_after_and_cap():
    p = RetryPolicy(base_delay=1, max_delay=4, jitter=False)
    assert compute_backoff(p, 10) == 4
    assert compute_backoff(p, 1, retry_after=20) == 20


async def test_rate_limiter_spaces_calls():
    now, slept = {"t": 0.0}, []

    async def fake_sleep(d):
        slept.append(d)
        now["t"] += d

    rl = AsyncRateLimiter(2.0, burst=1, clock=lambda: now["t"], sleep=fake_sleep)
    for _ in range(3):
        await rl.acquire()
    assert len(slept) == 2 and slept[0] == pytest.approx(0.5)
