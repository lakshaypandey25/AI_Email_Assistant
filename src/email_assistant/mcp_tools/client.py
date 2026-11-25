"""The agent's MCP client: EVERY Gmail / Calendar action the agent takes goes through here, over MCP.

    agent services  ->  McpToolClient.call(name, args)  --JSON-RPC-->  MCP server  ->  Gmail / Calendar APIs

* The agent process holds no Google credentials and no Gmail / Calendar client objects.
* Arguments are validated against the shared catalog before they are sent.
* HIGH-risk tools are refused unless the approval controller passes ``approved=True``; the client then attaches a
  signed single-use token that the server verifies independently (defence in depth).
* The MCP session lives in one background task, so any task may call tools concurrently and anyio's "enter and exit in
  the same task" rule is respected. The server process is started on first use and restarted if it dies.
* Server-side errors come back as JSON and are rebuilt as the original exception types, so retry / defer logic works.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from email_assistant.core.approval import SECRET_ENV, ApprovalSigner
from email_assistant.exceptions import (
    ApprovalRequiredError,
    AuthenticationRequiredError,
    ConfigurationError,
    GoogleAPIError,
    GoogleRateLimitError,
    RemoteToolError,
    ToolInputError,
    ToolNotFoundError,
    ToolTransportError,
)
from email_assistant.mcp_tools.catalog import CATALOG, ToolDef, get_def
from email_assistant.schemas.enums import RiskLevel

logger = structlog.get_logger(__name__)

Connector = Callable[[], AbstractAsyncContextManager[Any]]
_STOP = object()
_SIMPLE_ERRORS: dict[str, type[Exception]] = {
    "ToolInputError": ToolInputError,
    "ApprovalRequiredError": ApprovalRequiredError,
    "ToolNotFoundError": ToolNotFoundError,
    "AuthenticationRequiredError": AuthenticationRequiredError,
    "ConfigurationError": ConfigurationError,
}


def _decode_error(text: str) -> Exception:
    """Rebuild the server-side exception from the JSON the server put in the error text."""
    try:
        payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except ValueError:
        return RemoteToolError("RemoteError", text[:300])
    kind, message = str(payload.get("error_type", "RemoteError")), str(payload.get("message", ""))
    if kind in {"GoogleAPIError", "GoogleRateLimitError"}:
        cls = GoogleRateLimitError if kind == "GoogleRateLimitError" else GoogleAPIError
        return cls(
            message, status=payload.get("status"), reason=payload.get("reason"),
            retryable=bool(payload.get("retryable")), retry_after=payload.get("retry_after"),
        )  # fmt: skip
    if kind in _SIMPLE_ERRORS:
        return _SIMPLE_ERRORS[kind](message)
    return RemoteToolError(kind, message)


def _decode_result(result: Any) -> dict[str, Any]:
    text = "".join(getattr(part, "text", "") for part in result.content)
    if result.isError:
        raise _decode_error(text)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ToolTransportError("The MCP server returned something that is not JSON") from exc
    return data if isinstance(data, dict) else {"result": data}


# --------------------------------------------------------------------------- transports
@asynccontextmanager
async def _stdio_session(command: str, args: list[str], env: dict[str, str], cwd: str) -> AsyncIterator[Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _memory_session(server: Any) -> AsyncIterator[Any]:
    """In-process transport: the real MCP protocol (JSON-RPC messages) without a subprocess. Used by the tests."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server) as session:
        yield session


class McpToolClient:
    """Drop-in for :class:`~email_assistant.mcp_tools.registry.ToolRegistry` that calls tools over MCP."""

    def __init__(self, connector: Connector, signer: ApprovalSigner, *, call_timeout: float = 180.0) -> None:
        self._connector = connector
        self._signer = signer
        self._timeout = call_timeout
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[Any] | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def stdio(cls, *, extra_args: Sequence[str] = ("--no-ai",)) -> McpToolClient:
        """Spawn ``python -m email_assistant mcp-server`` as a child process on first use."""
        secret = secrets.token_bytes(32)
        src = str(Path(__file__).resolve().parents[2])
        env = {
            **os.environ,
            SECRET_ENV: secret.hex(),
            "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p),
        }
        args = ["-m", "email_assistant", "mcp-server", *extra_args]
        return cls(lambda: _stdio_session(sys.executable, args, env, os.getcwd()), ApprovalSigner(secret))

    @classmethod
    def in_memory(cls, server: Any, signer: ApprovalSigner) -> McpToolClient:
        """Connect to a server object in the same process (tests)."""
        return cls(lambda: _memory_session(server), signer)

    # ------------------------------------------------------------------ registry-compatible surface
    def get_spec(self, name: str) -> ToolDef:
        return get_def(name)

    def specs(self, *, max_risk: RiskLevel | None = None) -> list[ToolDef]:
        order = {RiskLevel.READ: 0, RiskLevel.LOW: 1, RiskLevel.HIGH: 2}
        return [d for d in CATALOG.values() if max_risk is None or order[d.risk] <= order[max_risk]]

    async def call(self, name: str, arguments: Mapping[str, Any], *, approved: bool = False) -> dict[str, Any]:
        """Validate, (sign if approved HIGH-risk), send over MCP and return the tool's result dictionary."""
        spec = get_def(name)
        if spec.risk is RiskLevel.HIGH and not approved:
            raise ApprovalRequiredError(f"Tool '{name}' is high-risk and requires human approval")
        try:
            payload = spec.input_model.model_validate(dict(arguments)).model_dump(mode="json")
        except ValidationError as exc:
            raise ToolInputError(f"Invalid arguments for '{name}': {exc.errors()[:3]}") from exc
        if spec.risk is RiskLevel.HIGH:
            payload["approval_token"] = self._signer.issue(name, payload)

        await self._ensure_started()
        assert self._queue is not None
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put((name, payload, future))
        logger.info("mcp_call", tool=name, risk=spec.risk.value, approved=approved)
        try:
            result = await future
        except (ToolTransportError, ApprovalRequiredError):
            raise
        except Exception as exc:  # noqa: BLE001 - connection problems surface as a retryable error
            raise ToolTransportError(f"MCP call failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        return _decode_result(result)

    async def aclose(self) -> None:
        """Close the MCP session (and stop the server process)."""
        task, queue = self._task, self._queue
        if task is not None and not task.done() and queue is not None:
            queue.put_nowait(_STOP)
            try:
                await asyncio.wait_for(task, timeout=15)
            except Exception:  # noqa: BLE001
                task.cancel()
        self._task = None

    # ------------------------------------------------------------------ session task
    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._task is not None and not self._task.done():
                return
            self._queue = asyncio.Queue()
            ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._task = asyncio.create_task(self._runner(self._queue, ready))
            await ready  # raises ToolTransportError if the server could not be started

    async def _runner(self, queue: asyncio.Queue[Any], ready: asyncio.Future[None]) -> None:
        inflight: set[asyncio.Task[None]] = set()
        try:
            async with self._connector() as session:
                if not ready.done():
                    ready.set_result(None)
                while True:
                    item = await queue.get()
                    if item is _STOP:
                        break
                    task = asyncio.create_task(self._dispatch(session, item, queue))
                    inflight.add(task)
                    task.add_done_callback(inflight.discard)
                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - includes anyio exception groups
            logger.error("mcp_session_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            if not ready.done():
                ready.set_exception(ToolTransportError(f"Could not start the MCP tool server: {type(exc).__name__}: {str(exc)[:200]}"))
        finally:
            while not queue.empty():
                item = queue.get_nowait()
                if item is not _STOP and not item[2].done():
                    item[2].set_exception(ToolTransportError("The MCP tool server stopped"))
            if not ready.done():
                ready.set_exception(ToolTransportError("The MCP tool server stopped"))

    async def _dispatch(self, session: Any, item: tuple[str, dict[str, Any], asyncio.Future[Any]], queue: asyncio.Queue[Any]) -> None:
        name, payload, future = item
        try:
            result = await asyncio.wait_for(session.call_tool(name, payload), timeout=self._timeout)
            if not future.done():
                future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            if not future.done():
                future.set_exception(exc)
            if type(exc).__name__ != "McpError":  # a broken pipe / timeout: stop so the next call reconnects
                queue.put_nowait(_STOP)