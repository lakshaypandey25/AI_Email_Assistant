"""The tool contract: the single list of tools, their input schemas and risk levels.

Used by BOTH sides of the MCP connection:
* the MCP server (``server.py``) exposes these tools and binds them to the Gmail / Calendar clients (``tools.py``);
* the agent's MCP client (``client.py``) validates arguments against the same models and knows each tool's risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from email_assistant.exceptions import ToolNotFoundError
from email_assistant.mcp_tools import names as N
from email_assistant.schemas.enums import RiskLevel
from email_assistant.schemas.tools import (
    CheckAvailabilityInput,
    CreateEventInput,
    CreateReplyDraftInput,
    DraftForMessageInput,
    DraftStatusInput,
    FreeWindowsInput,
    GetMessageInput,
    ListEventsInput,
    ProfileInput,
    SearchMessagesInput,
    SendDraftInput,
    ThreadRepliedInput,
)


@dataclass(frozen=True)
class ToolDef:
    """Declarative description of one tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    risk: RiskLevel


_DEFS = (
    ToolDef(N.TOOL_SEARCH, "Search the inbox with a Gmail query; returns message ids.", SearchMessagesInput, RiskLevel.READ),
    ToolDef(N.TOOL_GET_MESSAGE, "Fetch one message, cleaned and parsed (no quotes/signatures).", GetMessageInput, RiskLevel.READ),
    ToolDef(N.TOOL_PROFILE, "Return the authenticated mailbox address.", ProfileInput, RiskLevel.READ),
    ToolDef(N.TOOL_DRAFT_STATUS, "Report whether a draft still exists, was sent from Gmail, or was deleted.", DraftStatusInput, RiskLevel.READ),
    ToolDef(N.TOOL_THREAD_REPLIED, "Has the user already sent a reply in this thread since a given time?", ThreadRepliedInput, RiskLevel.READ),
    ToolDef(
        N.TOOL_CHECK_AVAILABILITY,
        "Return BUSY intervals (UTC) between two ISO-8601 instants, e.g. 2026-11-24T09:00:00Z. Empty = free for the whole window.",
        CheckAvailabilityInput, RiskLevel.READ,
    ),
    ToolDef(N.TOOL_LIST_EVENTS, "List existing calendar events between two instants (titles for the user's notes).", ListEventsInput, RiskLevel.READ),
    ToolDef(
        N.TOOL_FREE_WINDOWS,
        "Free time per day: the user's availability (Settings > Scheduling) minus meetings already on the calendar.",
        FreeWindowsInput, RiskLevel.READ,
    ),
    ToolDef(N.TOOL_CREATE_DRAFT, "Create a Gmail reply DRAFT from explicit fields (does not send).", CreateReplyDraftInput, RiskLevel.LOW),
    ToolDef(
        N.TOOL_DRAFT_FOR_MESSAGE,
        "Save your text as a reply DRAFT to an existing message; recipient, thread, subject and headers are filled in. Never sent.",
        DraftForMessageInput, RiskLevel.LOW,
    ),
    ToolDef(N.TOOL_SEND_DRAFT, "Send an existing Gmail draft.", SendDraftInput, RiskLevel.HIGH),
    ToolDef(N.TOOL_CREATE_EVENT, "Create a calendar event and send invites.", CreateEventInput, RiskLevel.HIGH),
)

CATALOG: dict[str, ToolDef] = {d.name: d for d in _DEFS}


def get_def(name: str) -> ToolDef:
    """Look up a tool definition (raises :class:`ToolNotFoundError`)."""
    try:
        return CATALOG[name]
    except KeyError as exc:
        raise ToolNotFoundError(f"Unknown tool: {name}") from exc