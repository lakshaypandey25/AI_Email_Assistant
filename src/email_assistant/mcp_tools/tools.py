"""Server-side tool handlers: bind the catalog's tools to the Gmail / Calendar clients.

This code runs inside the MCP server process - the only process that talks to Google.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from typing import Any

from email_assistant.config import Settings
from email_assistant.integrations.calendar_client import CalendarClient
from email_assistant.integrations.gmail_client import GmailClient
from email_assistant.mcp_tools import names as N
from email_assistant.mcp_tools.catalog import CATALOG
from email_assistant.mcp_tools.registry import ToolRegistry, ToolSpec
from email_assistant.scheduling.slots import Interval, format_free_windows, free_windows
from email_assistant.services.drafting import reply_subject
from email_assistant.services.fetcher import EmailFetcher

Handler = Callable[[Any], Awaitable[dict[str, Any]]]


def build_registry(
    *,
    gmail: GmailClient,
    calendar: CalendarClient,
    fetcher: EmailFetcher,
    settings: Settings,
) -> ToolRegistry:
    """Create the server-side registry: every catalog tool bound to its handler."""

    async def search(args: Any) -> dict[str, Any]:
        return {"message_ids": await fetcher.list_new_ids(query=args.query, max_results=args.max_results)}

    async def get_message(args: Any) -> dict[str, Any]:
        return (await fetcher.fetch(args.message_id)).model_dump(mode="json")

    async def profile(_: Any) -> dict[str, Any]:
        return {"email": await gmail.own_email()}

    async def draft_status(args: Any) -> dict[str, Any]:
        return {"status": await gmail.draft_status(args.draft_id, args.thread_id, args.since)}

    async def thread_replied(args: Any) -> dict[str, Any]:
        return {"replied": await gmail.thread_has_reply_since(args.thread_id, args.since)}

    async def check_availability(args: Any) -> dict[str, Any]:
        busy = await calendar.query_freebusy(args.time_min, args.time_max, args.calendar_ids)
        return {"busy": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in busy]}

    async def list_events(args: Any) -> dict[str, Any]:
        return {"events": await calendar.list_events(args.time_min, args.time_max, args.calendar_id)}

    async def free_windows_handler(args: Any) -> dict[str, Any]:
        tz = settings.tzinfo
        first = date.fromisoformat(args.start_date) if args.start_date else datetime.now(tz).date()
        start = datetime.combine(first, time(0, 0), tzinfo=tz)
        end = start + timedelta(days=args.days)
        busy = [Interval(s, e) for s, e in await calendar.query_freebusy(start, end, None)]
        windows = free_windows(
            availability=settings.availability_schedule, busy=busy, start=max(start, datetime.now(tz)), end=end, tz=tz,
            min_length=timedelta(minutes=args.duration_minutes), buffer=timedelta(minutes=settings.meeting_buffer_minutes),
        )
        return {"timezone": settings.user_timezone, "free": format_free_windows(windows)}

    async def create_draft(args: Any) -> dict[str, Any]:
        draft_id = await gmail.create_draft(
            thread_id=args.thread_id, to=args.to, subject=args.subject, body=args.body,
            in_reply_to=args.in_reply_to, references=args.references, cc=args.cc,
        )
        return {"draft_id": draft_id}

    async def draft_for_message(args: Any) -> dict[str, Any]:
        email = await fetcher.fetch(args.message_id)
        subject = reply_subject(email.subject)
        draft_id = await gmail.create_draft(
            thread_id=email.thread_id, to=email.sender_email, subject=subject,
            body=f"{args.body.strip()}\n\n{settings.signature}", in_reply_to=email.rfc_message_id, references=email.references,
        )
        return {"draft_id": draft_id, "to": email.sender_email, "subject": subject}

    async def send_draft(args: Any) -> dict[str, Any]:
        return await gmail.send_draft(args.draft_id)

    async def create_event(args: Any) -> dict[str, Any]:
        return await calendar.create_event(
            calendar_id=args.calendar_id or settings.calendar_id, summary=args.summary, description=args.description,
            start=args.start, end=args.end, timezone=args.timezone, attendees=args.attendees,
            add_meet_link=args.add_meet_link, send_updates=args.send_updates, idempotency_key=args.idempotency_key,
        )

    handlers: dict[str, Handler] = {
        N.TOOL_SEARCH: search, N.TOOL_GET_MESSAGE: get_message, N.TOOL_PROFILE: profile,
        N.TOOL_DRAFT_STATUS: draft_status, N.TOOL_THREAD_REPLIED: thread_replied,
        N.TOOL_CHECK_AVAILABILITY: check_availability, N.TOOL_LIST_EVENTS: list_events,
        N.TOOL_FREE_WINDOWS: free_windows_handler, N.TOOL_CREATE_DRAFT: create_draft,
        N.TOOL_DRAFT_FOR_MESSAGE: draft_for_message, N.TOOL_SEND_DRAFT: send_draft, N.TOOL_CREATE_EVENT: create_event,
    }
    missing = set(CATALOG) - set(handlers)
    if missing:
        raise RuntimeError(f"No handler bound for tools: {sorted(missing)}")

    registry = ToolRegistry()
    for definition in CATALOG.values():
        registry.register(
            ToolSpec(definition.name, definition.description, definition.input_model, handlers[definition.name], definition.risk)
        )
    return registry