"""The MCP server (stdio): the ONLY process that talks to Gmail and Google Calendar.

Two ways it runs
----------------
* **Spawned by the agent** (secret in the environment): exposes every tool. HIGH-risk tools (send a draft, create an
  event) still need a signed, single-use approval token that only the agent's approval controller can mint.
* **Standalone** (``python -m email_assistant mcp-server``, for MCP Inspector / Claude Desktop): exposes only READ and
  LOW-risk tools, plus an optional ``ai_draft_reply`` tool. Nobody outside can send mail or create invites.

stdout carries the protocol; all logs go to stderr.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from email_assistant.config import get_settings
from email_assistant.core.approval import SECRET_ENV, ApprovalSigner
from email_assistant.exceptions import ApprovalRequiredError, GoogleAPIError, ToolNotFoundError
from email_assistant.logging_config import configure_logging
from email_assistant.mcp_tools.catalog import CATALOG
from email_assistant.schemas.enums import RiskLevel

logger = structlog.get_logger(__name__)

#: name -> (description, JSON schema, handler) for tools that are not part of the shared catalog.
ExtraTool = tuple[str, dict[str, Any], Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]]

_APPROVAL_HELP = "Requires a signed human-approval token that only the application's approval controller can issue."


def encode_error(exc: BaseException) -> str:
    """Errors cross the protocol as JSON so the client can rebuild the original exception type and retry flags."""
    payload: dict[str, Any] = {"error_type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, GoogleAPIError):
        payload.update(status=exc.status, reason=exc.reason, retryable=exc.retryable, retry_after=exc.retry_after)
    return json.dumps(payload)


def create_mcp_server(
    registry: Any,
    *,
    signer: ApprovalSigner,
    expose_high_risk: bool,
    extra: dict[str, ExtraTool] | None = None,
) -> Any:
    """Build the MCP server over ``registry`` (the validated tool dispatcher bound to Gmail / Calendar)."""
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("ai-email-assistant")
    defs = {name: d for name, d in CATALOG.items() if expose_high_risk or d.risk is not RiskLevel.HIGH}
    extras = dict(extra or {})

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for d in defs.values():
            schema = d.input_model.model_json_schema()
            description = d.description
            if d.risk is RiskLevel.HIGH:
                schema.setdefault("properties", {})["approval_token"] = {"type": "string", "description": _APPROVAL_HELP}
                description = f"{description} {_APPROVAL_HELP}"
            tools.append(types.Tool(name=d.name, description=description, inputSchema=schema))
        for name, (description, schema, _) in extras.items():
            tools.append(types.Tool(name=name, description=description, inputSchema=schema))
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        args = dict(arguments or {})
        try:
            if name in extras:
                result = await extras[name][2](args)
            else:
                definition = defs.get(name)
                if definition is None:
                    raise ToolNotFoundError(f"Tool not available on this server: {name}")
                token = args.pop("approval_token", "")
                approved = False
                if definition.risk is RiskLevel.HIGH:
                    if not signer.verify(str(token), name, args):
                        raise ApprovalRequiredError(f"'{name}' needs a valid, unused human-approval token")
                    approved = True
                result = await registry.call(name, args, approved=approved)
            logger.info("mcp_server_call", tool=name)
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as exc:  # noqa: BLE001 - every failure is returned to the client in a structured form
            logger.warning("mcp_server_call_failed", tool=name, error_type=type(exc).__name__)
            raise RuntimeError(encode_error(exc)) from exc

    return server


def build_ai_tools(app: Any, settings: Any) -> dict[str, ExtraTool]:
    """Optional tool for standalone use: let the AI write the reply (never sends)."""
    from email_assistant.core.pii import PiiMasker
    from email_assistant.mcp_tools import names as N
    from email_assistant.schemas.parsed_email import ParsedEmail
    from email_assistant.schemas.tools import AiDraftInput

    state = {"db_ready": False}

    async def ai_draft_reply(raw: dict[str, Any]) -> dict[str, Any]:
        args = AiDraftInput.model_validate(raw)
        if not state["db_ready"]:
            await app.db.create_all()
            state["db_ready"] = True
        email = ParsedEmail.model_validate(await app.tools.call(N.TOOL_GET_MESSAGE, {"message_id": args.message_id}))
        masker = PiiMasker()
        memory = await app.memory.get_context(email)
        outcome = await app.classifier.classify(email, masker, memory)
        knowledge = await app.knowledge.retrieve(f"{email.subject}\n{email.body_text}", settings.kb_max_snippets)
        own = (await app.tools.call(N.TOOL_PROFILE, {}))["email"]
        result = await app.drafter.draft_reply(email, outcome.result, masker, memory, own_email=own, knowledge_text=knowledge)
        info: dict[str, Any] = {
            "category": outcome.result.category.value,
            "summary": outcome.result.summary,
            "suggested_mode": outcome.result.effective_response_mode,
        }
        if result.blocked:
            return {**info, "status": "blocked", "issues": [i.code for i in result.issues]}
        info.update(status="ok", to=email.sender_email, reply_subject=result.subject, reply_body=result.body, warnings=result.warnings)
        if args.save_draft:
            draft = await app.tools.call(
                N.TOOL_CREATE_DRAFT,
                {
                    "thread_id": email.thread_id, "to": email.sender_email, "subject": result.subject, "body": result.body,
                    "in_reply_to": email.rfc_message_id, "references": email.references,
                },
            )
            info["draft_id"] = draft["draft_id"]
        return info

    description = (
        "Read an email, classify it, and WRITE a reply with the AI (tone, knowledge base and memory applied). "
        "Saves a Gmail draft and returns the text; never sends. For meeting requests use the full pipeline in the app."
    )
    return {"ai_draft_reply": (description, AiDraftInput.model_json_schema(), ai_draft_reply)}


async def _serve(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(no_ai: bool = False) -> None:
    """Entry point: ``python -m email_assistant mcp-server [--no-ai]``."""
    from email_assistant.container import build_container, build_google_stack

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    secret_hex = os.environ.get(SECRET_ENV, "")
    spawned_by_agent = bool(secret_hex)
    signer = ApprovalSigner(bytes.fromhex(secret_hex)) if spawned_by_agent else ApprovalSigner.random()
    google = build_google_stack(settings)
    extra: dict[str, ExtraTool] = {}
    if not spawned_by_agent and not no_ai and settings.groq_api_key.get_secret_value():
        extra = build_ai_tools(build_container(settings, tools=google.registry), settings)
    server = create_mcp_server(google.registry, signer=signer, expose_high_risk=spawned_by_agent, extra=extra)
    asyncio.run(_serve(server))


if __name__ == "__main__":
    main()