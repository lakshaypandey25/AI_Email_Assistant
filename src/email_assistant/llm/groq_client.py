"""Groq client returning Pydantic-validated structured output, built to survive free-tier limits.

1. Model fallback chain: first-choice model, then the other tier's model, then GROQ_FALLBACK_MODELS.
   Groq limits are per model, so a rate-limited model gets a cool-down and the next one is tried immediately.
2. Token budget per model (sliding minute window): fail over BEFORE Groq answers 429.
3. Short retry with backoff (+ Retry-After) on the same model.
4. Validation repair loop for invalid tool output.
5. If every model fails the caller gets a typed error so services can defer, use rules, or use templates.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import groq
import structlog
from groq import AsyncGroq
from pydantic import ValidationError

from email_assistant.config import Settings
from email_assistant.core.rate_limit import AsyncRateLimiter, TokenWindow
from email_assistant.core.retry import RetryPolicy, retry_async
from email_assistant.exceptions import (
    ConfigurationError,
    LLMError,
    LLMOutputError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from email_assistant.llm.base import ModelTier, T
from email_assistant.llm.tooling import pydantic_to_tool

logger = structlog.get_logger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
Recorder = Callable[[dict[str, Any]], Awaitable[None]]


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(
        exc, (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)
    )


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        value = response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


class GroqLLMClient:
    """StructuredLLM implementation backed by Groq."""

    #: Longest we sleep for the token budget before failing over to another model.
    MAX_BUDGET_WAIT = 20.0

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        recorder: Recorder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        if client is None and not settings.groq_api_key.get_secret_value():
            raise ConfigurationError("GROQ_API_KEY is not set. Add it in the UI (Settings > Connections) or in .env")
        self._client = client or AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=settings.groq_timeout_seconds,
            max_retries=0,
        )
        self._limiter = rate_limiter or AsyncRateLimiter(
            settings.groq_requests_per_minute / 60.0, burst=min(5, settings.groq_requests_per_minute)
        )
        policy = retry_policy or RetryPolicy(max_attempts=2, base_delay=2.0, max_delay=15.0, max_retry_after=15.0)
        self._sleep = policy.sleep
        self._clock = clock
        self._recorder = recorder
        self._cooldowns: dict[str, float] = {}
        self._windows: dict[str, TokenWindow] = {}
        self._complete = retry_async(
            policy, is_retryable=_is_retryable, get_retry_after=_retry_after, operation="groq_chat_completion"
        )(self._complete_once)

    # ------------------------------------------------------------------ model chain / cool-downs
    def candidates(self, tier: ModelTier) -> list[str]:
        s = self._settings
        primary, other = (s.groq_model_quality, s.groq_model_fast) if tier == "quality" else (s.groq_model_fast, s.groq_model_quality)
        return list(dict.fromkeys(m for m in (primary, other, *s.fallback_models_list) if m))

    def _cool(self, model: str, seconds: float) -> None:
        self._cooldowns[model] = self._clock() + max(1.0, seconds)

    def _cooling(self, model: str) -> bool:
        return self._cooldowns.get(model, 0.0) > self._clock()

    def _window(self, model: str) -> TokenWindow:
        if model not in self._windows:
            self._windows[model] = TokenWindow(self._settings.groq_tokens_per_minute, clock=self._clock)
        return self._windows[model]

    async def _record(self, **fields: Any) -> None:
        if self._recorder is None:
            return
        try:
            await self._recorder(fields)
        except Exception:  # noqa: BLE001 - telemetry must never break the pipeline
            logger.warning("llm_metric_record_failed")

    # ------------------------------------------------------------------ low level
    async def _complete_once(self, *, model: str, messages: list[dict[str, str]], tool: dict[str, Any]) -> Any:
        await self._limiter.acquire()
        extra: dict[str, Any] = {}
        if model.startswith("openai/gpt-oss"):
            extra["extra_body"] = {"reasoning_effort": self._settings.llm_reasoning_effort}
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.groq_max_output_tokens,
            **extra,
        )

    @staticmethod
    def _extract_arguments(response: Any, tool_name: str) -> Any:
        message = response.choices[0].message
        for call in getattr(message, "tool_calls", None) or []:
            if call.function.name == tool_name:
                return json.loads(call.function.arguments)
        content = (getattr(message, "content", None) or "").strip()
        if content:
            return json.loads(_FENCE_RE.sub("", content).strip())
        raise ValueError("model returned neither a tool call nor JSON content")

    @staticmethod
    def _summarise_error(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
        return str(exc)

    @staticmethod
    def _repair_message(problem: str) -> str:
        return (
            f"Your previous function call was invalid ({problem}). "
            "Call the function again with corrected arguments that satisfy the schema exactly."
        )

    # ------------------------------------------------------------------ one model
    async def _call_model(
        self, model: str, system: str, user: str, schema: type[T], tool: dict[str, Any], tool_name: str, fallback_used: bool
    ) -> T:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        attempts = self._settings.llm_max_validation_attempts
        window = self._window(model)
        estimate = int((len(system) + len(user)) / 3.2) + 600
        last_problem = "unknown"
        meta = {"model": model, "tool": tool_name, "fallback_used": fallback_used}

        for attempt in range(1, attempts + 1):
            wait = window.wait_needed(estimate)
            if wait > self.MAX_BUDGET_WAIT:
                self._cool(model, wait)
                raise LLMRateLimitError(f"{model}: token budget exhausted (would wait {wait:.0f}s)")
            if wait > 0:
                await self._sleep(wait)
            reservation = window.reserve(estimate)
            started = self._clock()
            try:
                response = await self._complete(model=model, messages=messages, tool=tool)
            except groq.AuthenticationError as exc:
                raise ConfigurationError("Groq rejected the API key (GROQ_API_KEY)") from exc
            except groq.RateLimitError as exc:
                self._cool(model, _retry_after(exc) or 60.0)
                await self._record(**meta, ok=False, valid=False, latency_ms=0, total_tokens=0, error_type="RateLimit")
                raise LLMRateLimitError(f"{model}: Groq rate limit exceeded") from exc
            except groq.NotFoundError as exc:
                self._cool(model, 3600.0)  # retired / not available to this key
                await self._record(**meta, ok=False, valid=False, latency_ms=0, total_tokens=0, error_type="ModelNotFound")
                raise LLMUnavailableError(f"{model}: model not available to this API key") from exc
            except (groq.APIConnectionError, groq.InternalServerError) as exc:
                self._cool(model, 20.0)
                await self._record(**meta, ok=False, valid=False, latency_ms=0, total_tokens=0, error_type=type(exc).__name__)
                raise LLMUnavailableError(f"{model}: Groq unavailable ({type(exc).__name__})") from exc
            except groq.BadRequestError as exc:
                if "tool_use_failed" in str(exc):  # model emitted a malformed / schema-violating tool call
                    last_problem = "the tool call did not match the schema"
                    logger.warning("llm_tool_use_failed", tool=tool_name, model=model, attempt=attempt, detail=str(exc)[:600])
                    await self._record(**meta, ok=True, valid=False, latency_ms=0, total_tokens=0, error_type="ToolUseFailed")
                    messages.append({"role": "user", "content": self._repair_message(last_problem)})
                    continue
                logger.error("groq_bad_request", detail=str(exc)[:400])
                raise LLMError(f"{model}: Groq rejected the request: {str(exc)[:200]}") from exc
            except groq.APIStatusError as exc:
                logger.error("groq_api_error", status=exc.status_code, detail=str(exc)[:400])
                raise LLMError(f"{model}: Groq API error {exc.status_code}") from exc

            usage = getattr(response, "usage", None)
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if tokens:
                reservation[1] = float(tokens)  # replace the estimate with real usage
            latency_ms = int((self._clock() - started) * 1000)
            try:
                result = schema.model_validate(self._extract_arguments(response, tool_name))
            except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
                last_problem = self._summarise_error(exc)
                logger.warning("llm_output_invalid", tool=tool_name, model=model, attempt=attempt, problem=last_problem[:300])
                await self._record(**meta, ok=True, valid=False, latency_ms=latency_ms, total_tokens=tokens, error_type="InvalidOutput")
                messages.append({"role": "user", "content": self._repair_message(last_problem)})
                continue
            logger.info("llm_call_completed", tool=tool_name, model=model, attempt=attempt, total_tokens=tokens, fallback=fallback_used)
            await self._record(**meta, ok=True, valid=True, latency_ms=latency_ms, total_tokens=tokens, error_type=None)
            return result

        raise LLMOutputError(f"{tool_name}: no valid output from {model} after {attempts} attempts ({last_problem[:200]})")

    # ------------------------------------------------------------------ public API
    async def structured_call(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tool_name: str,
        tool_description: str,
        tier: ModelTier = "fast",
    ) -> T:
        """Return a validated ``schema`` instance, trying the model chain until one succeeds.

        Raises LLMRateLimitError (all models limited: callers defer and retry), LLMUnavailableError (outage / models
        retired), LLMOutputError (never valid output) or ConfigurationError (bad API key).
        """
        tool = pydantic_to_tool(schema, tool_name, tool_description)
        chain = self.candidates(tier)
        rate_limited = unavailable = False
        last_error: LLMError | None = None

        for index, model in enumerate(chain):
            if self._cooling(model):
                rate_limited = True
                logger.info("llm_model_skipped_cooling_down", model=model)
                continue
            try:
                return await self._call_model(model, system, user, schema, tool, tool_name, fallback_used=index > 0)
            except LLMRateLimitError as exc:
                rate_limited, last_error = True, exc
            except LLMUnavailableError as exc:
                unavailable, last_error = True, exc
            except LLMError as exc:  # includes LLMOutputError
                last_error = exc
            logger.warning("llm_model_failed_trying_next", failed=model, remaining=len(chain) - index - 1)

        if rate_limited:
            raise LLMRateLimitError("All Groq models are rate-limited right now") from last_error
        if unavailable:
            raise LLMUnavailableError("No Groq model is reachable right now") from last_error
        raise last_error or LLMUnavailableError("No Groq model configured")