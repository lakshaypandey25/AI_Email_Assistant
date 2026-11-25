"""Domain exception hierarchy.

Every failure that crosses a layer boundary is expressed as one of these types so
that callers can make retry / fallback / escalation decisions without knowing
which third-party library raised the original error.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for all application errors."""

    #: Optional server-suggested wait (seconds) before retrying.
    retry_after: float | None = None


class ConfigurationError(AssistantError):
    """Invalid or missing configuration (never retryable)."""


class AuthenticationRequiredError(AssistantError):
    """Google credentials are missing, revoked, expired or lack required scopes.

    Resolution: run ``python -m email_assistant auth``.
    """


class GoogleAPIError(AssistantError):
    """Error returned by a Google API (or the network path to it)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        reason: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.retryable = retryable
        self.retry_after = retry_after


class GoogleRateLimitError(GoogleAPIError):
    """Quota / rate limit exceeded (HTTP 429 or 403 rateLimitExceeded)."""

class ToolTransportError(GoogleAPIError):
    """The MCP connection to the tool server failed (server died, timeout...). Retryable: the email is deferred."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class RemoteToolError(AssistantError):
    """A tool failed on the MCP server with an error type this side does not know."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type

class LLMError(AssistantError):
    """Base class for LLM failures."""


class LLMOutputError(LLMError):
    """The model never produced output that satisfies the requested schema."""


class LLMRateLimitError(LLMError):
    """Provider rate limit that could not be absorbed by retrying."""


class LLMUnavailableError(LLMError):
    """Provider unreachable / 5xx after retries."""


class ToolNotFoundError(AssistantError):
    """Requested tool is not registered."""


class ToolInputError(AssistantError):
    """Tool arguments failed schema validation."""


class ApprovalRequiredError(AssistantError):
    """A high-risk tool was invoked without human approval."""


class ApprovalStateError(AssistantError):
    """Illegal approval state transition (e.g. approving an already-executed action)."""


class GuardrailViolationError(AssistantError):
    """Generated content violated a safety guardrail."""


class SchedulingError(AssistantError):
    """Meeting could not be scheduled for a deterministic reason."""
