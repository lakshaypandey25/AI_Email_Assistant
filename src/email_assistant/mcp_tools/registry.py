"""Typed tool registry: the single choke point through which every external side effect flows.

Each tool declares an input schema, handler and :class:`RiskLevel`. ``ToolRegistry.call``
validates the arguments and **refuses to run HIGH-risk tools unless ``approved=True``** - a flag
only the human-in-the-loop controller sets after a person (or an explicit opt-in policy) approved.
This gives defence-in-depth: even a bug in the agent cannot send an email or create an invite.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from email_assistant.exceptions import ApprovalRequiredError, ToolInputError, ToolNotFoundError
from email_assistant.schemas.enums import RiskLevel

logger = structlog.get_logger(__name__)

ToolHandler = Callable[[Any], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of a tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    risk: RiskLevel


class ToolRegistry:
    """Registry + validated dispatcher for tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get_spec(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Unknown tool: {name}") from exc

    def specs(self, *, max_risk: RiskLevel | None = None) -> list[ToolSpec]:
        """List tools, optionally only those at or below ``max_risk`` (READ < LOW < HIGH)."""
        order = {RiskLevel.READ: 0, RiskLevel.LOW: 1, RiskLevel.HIGH: 2}
        return [s for s in self._tools.values() if max_risk is None or order[s.risk] <= order[max_risk]]

    async def call(self, name: str, arguments: Mapping[str, Any], *, approved: bool = False) -> dict[str, Any]:
        """Validate ``arguments`` and execute the tool.

        Raises:
            ToolNotFoundError: unknown tool.
            ToolInputError: arguments failed validation.
            ApprovalRequiredError: HIGH-risk tool invoked with ``approved=False``.
        """
        spec = self.get_spec(name)
        if spec.risk is RiskLevel.HIGH and not approved:
            raise ApprovalRequiredError(f"Tool '{name}' is high-risk and requires human approval")
        try:
            parsed = spec.input_model.model_validate(dict(arguments))
        except ValidationError as exc:
            raise ToolInputError(f"Invalid arguments for '{name}': {exc.errors()[:3]}") from exc
        logger.info("tool_call", tool=name, risk=spec.risk.value, approved=approved)
        return await spec.handler(parsed)
