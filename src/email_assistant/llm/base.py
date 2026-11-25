"""LLM interface used by services (allows swapping providers and injecting fakes in tests)."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

ModelTier = Literal["fast", "quality"]


class StructuredLLM(Protocol):
    """An LLM that returns a validated Pydantic object via function calling."""

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
        """Return an instance of ``schema`` or raise :class:`~email_assistant.exceptions.LLMError`."""
        ...
