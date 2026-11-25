"""Bridges the synchronous Streamlit script to the async application.

Each call builds a fresh container inside its own event loop and disposes it afterwards, which avoids
sharing asyncio primitives / DB connections between loops (Streamlit re-runs the script on every click).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from email_assistant.auth.google_auth import GoogleAuthManager
from email_assistant.config import Settings
from email_assistant.container import Container, build_container
from email_assistant.exceptions import ConfigurationError
from email_assistant.mcp_tools.names import TOOL_PROFILE

T = TypeVar("T")


class _UnavailableLLM:
    """Placeholder used by pages that never call the model (approvals, history) so they work without a key."""

    async def structured_call(self, **_: Any) -> Any:
        raise ConfigurationError("GROQ_API_KEY is not set. Add it under Settings > Connections.")


def run_app(settings: Settings, fn: Callable[[Container], Awaitable[T]], *, need_llm: bool = False) -> T:
    """Run ``fn(container)`` to completion and return its result."""

    async def _main() -> T:
        llm = None if need_llm else _UnavailableLLM()
        app = build_container(settings, llm=llm)  # type: ignore[arg-type]
        await app.db.create_all()
        try:
            return await fn(app)
        finally:
            await app.aclose()

    return asyncio.run(_main())


def list_groq_models(api_key: str) -> list[str]:
    """Model ids visible to this Groq key (model names change over time - never hard-code them)."""
    from groq import Groq

    client = Groq(api_key=api_key, max_retries=0, timeout=20)
    return sorted(m.id for m in client.models.list().data)


def test_groq_model(api_key: str, model: str) -> tuple[bool, str]:
    """Send a one-token request to verify a model works with this key."""
    from groq import Groq

    try:
        Groq(api_key=api_key, max_retries=0, timeout=20).chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Say OK"}], max_tokens=16
        )
        return True, "works"
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"

async def _profile(app: Container) -> str:
    return str((await app.tools.call(TOOL_PROFILE, {}))["email"])

def google_mailbox(settings: Settings) -> str:
    """Return the authenticated mailbox address (verifies the whole chain: agent, MCP server, Google token)."""
    return run_app(settings, _profile)

def connect_google(settings: Settings) -> None:
    """Open the browser consent flow on this machine and store the token."""
    GoogleAuthManager(settings).run_interactive_flow()


def disconnect_google(settings: Settings) -> None:
    GoogleAuthManager(settings).logout()

def run_benchmark_sync(settings: Settings, *, quick: bool, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Run the labelled benchmark with the real model (needs a Groq key)."""
    from email_assistant.evaluation.runner import run_benchmark
    from email_assistant.llm.groq_client import GroqLLMClient

    async def _go() -> dict[str, Any]:
        return await run_benchmark(settings, GroqLLMClient(settings), quick=quick, progress=progress)

    return asyncio.run(_go())