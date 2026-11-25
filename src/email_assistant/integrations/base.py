"""Shared plumbing for Google API clients: threading, throttling, retries, error translation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from email_assistant.auth.google_auth import GoogleAuthManager
from email_assistant.config import Settings
from email_assistant.core.rate_limit import AsyncRateLimiter
from email_assistant.core.retry import RetryPolicy, retry_async
from email_assistant.integrations.google_errors import (
    NETWORK_ERRORS,
    google_retry_after,
    is_retryable_google_error,
    translate_http_error,
    translate_network_error,
)


class GoogleApiClient:
    """Base class: executes blocking ``googleapiclient`` requests off the event loop.

    A fresh service object is built per call (``httplib2`` is not thread-safe), calls are
    throttled by a token bucket + concurrency semaphore, and transient failures
    (429 / quota 403 / 5xx / network) are retried with exponential backoff.
    """

    service_name: str = ""
    service_version: str = ""

    def __init__(
        self,
        auth: GoogleAuthManager,
        settings: Settings,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._auth = auth
        self._settings = settings
        self._limiter = AsyncRateLimiter(settings.google_requests_per_second, burst=settings.google_max_concurrency)
        self._semaphore = asyncio.Semaphore(settings.google_max_concurrency)
        policy = retry_policy or RetryPolicy(max_attempts=settings.google_retry_max_attempts)
        self._execute = retry_async(
            policy,
            is_retryable=is_retryable_google_error,
            get_retry_after=google_retry_after,
            operation=f"google_{self.service_name}",
        )(self._execute_once)

    def _build_service(self, creds: Credentials) -> Any:
        return build(self.service_name, self.service_version, credentials=creds, cache_discovery=False)

    def _run_sync(self, creds: Credentials, request_factory: Callable[[Any], Any]) -> Any:
        try:
            service = self._build_service(creds)
            return request_factory(service).execute()
        except HttpError as exc:
            raise translate_http_error(exc) from exc
        except NETWORK_ERRORS as exc:
            raise translate_network_error(exc) from exc

    async def _execute_once(self, request_factory: Callable[[Any], Any]) -> Any:
        creds = await self._auth.get_credentials()
        async with self._semaphore:
            await self._limiter.acquire()
            return await asyncio.to_thread(self._run_sync, creds, request_factory)
