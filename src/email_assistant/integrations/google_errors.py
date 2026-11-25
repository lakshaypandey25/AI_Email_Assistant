"""Translate ``googleapiclient`` / network errors into domain errors with retry metadata."""

from __future__ import annotations

import json
import ssl

import httplib2
from googleapiclient.errors import HttpError

from email_assistant.exceptions import GoogleAPIError, GoogleRateLimitError

_RATE_LIMIT_REASONS = {
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "quotaExceeded",
    "dailyLimitExceeded",
    "RATE_LIMIT_EXCEEDED",
}
_RETRYABLE_STATUSES = {500, 502, 503, 504}


def _extract_reason(exc: HttpError) -> str | None:
    try:
        payload = json.loads(exc.content.decode("utf-8"))
        error = payload.get("error", {})
        errors = error.get("errors") or []
        if errors and isinstance(errors[0], dict) and errors[0].get("reason"):
            return str(errors[0]["reason"])
        details = error.get("details") or []
        for item in details:
            if isinstance(item, dict) and item.get("reason"):
                return str(item["reason"])
        return error.get("status")
    except (ValueError, AttributeError, UnicodeDecodeError, TypeError):
        return None


def _parse_retry_after(exc: HttpError) -> float | None:
    try:
        value = exc.resp.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def translate_http_error(exc: HttpError) -> GoogleAPIError:
    """Map an :class:`HttpError` to :class:`GoogleAPIError` / :class:`GoogleRateLimitError`."""
    status = int(getattr(exc.resp, "status", 0) or 0)
    reason = _extract_reason(exc)
    rate_limited = status == 429 or (status == 403 and reason in _RATE_LIMIT_REASONS)
    retryable = rate_limited or status in _RETRYABLE_STATUSES
    cls = GoogleRateLimitError if rate_limited else GoogleAPIError
    return cls(
        f"Google API error {status} ({reason or 'unknown'})",
        status=status,
        reason=reason,
        retryable=retryable,
        retry_after=_parse_retry_after(exc),
    )


def translate_network_error(exc: BaseException) -> GoogleAPIError:
    """Wrap transport-level failures as retryable errors."""
    return GoogleAPIError(f"Network error talking to Google: {type(exc).__name__}", retryable=True)


NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httplib2.HttpLib2Error,
    ssl.SSLError,
    ConnectionError,
    TimeoutError,
)


def is_retryable_google_error(exc: BaseException) -> bool:
    """Predicate for :func:`retry_async`."""
    return isinstance(exc, GoogleAPIError) and exc.retryable


def google_retry_after(exc: BaseException) -> float | None:
    """Extractor for :func:`retry_async`."""
    return exc.retry_after if isinstance(exc, GoogleAPIError) else None
