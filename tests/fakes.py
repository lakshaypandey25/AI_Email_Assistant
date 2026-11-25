"""Test doubles for Gmail, Calendar and the LLM (no network, no credentials)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any

from email_assistant.exceptions import GoogleAPIError, LLMOutputError
from email_assistant.integrations.gmail_client import RawMessage


def make_raw(
    *,
    msg_id: str = "m1",
    thread_id: str = "t1",
    subject: str = "Hello",
    body: str = "Hi Dana, can you review the doc?",
    sender: str = "Alice Smith <alice@acme.com>",
    to: str = "dana@example.com",
    html: str | None = None,
    headers: dict[str, str] | None = None,
) -> RawMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg["Date"] = "Mon, 03 Mar 2025 09:00:00 +0000"
    msg["Message-ID"] = f"<{msg_id}@acme.com>"
    for k, v in (headers or {}).items():
        msg[k] = v
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    return RawMessage(id=msg_id, thread_id=thread_id, raw=msg.as_bytes(), label_ids=["INBOX"], snippet=body[:50])


class FakeGmail:
    def __init__(self, messages: list[RawMessage] | None = None, own: str = "dana@example.com") -> None:
        self.messages = {m.id: m for m in (messages or [])}
        self.own = own
        self.drafts: dict[str, dict[str, Any]] = {}
        self.sent: list[str] = []
        self.sent_externally: set[str] = set()  # drafts the user sent by hand
        self.replied_threads: set[str] = set()  # threads the user already answered by hand

    async def thread_has_reply_since(self, thread_id: str, since: datetime) -> bool:
        return thread_id in self.replied_threads

    async def draft_status(self, draft_id: str, thread_id: str, since: datetime) -> str:
        if draft_id in self.drafts:
            return "draft"
        return "sent" if draft_id in self.sent_externally else "gone"

    async def own_email(self) -> str:
        return self.own

    async def list_message_ids(self, query: str, max_results: int = 20) -> list[str]:
        return list(self.messages)[:max_results]

    async def get_raw_message(self, message_id: str) -> RawMessage:
        return self.messages[message_id]

    async def create_draft(self, **kwargs: Any) -> str:
        draft_id = f"draft-{len(self.drafts) + 1}"
        self.drafts[draft_id] = kwargs
        return draft_id

    async def send_draft(self, draft_id: str) -> dict[str, str]:
        if draft_id not in self.drafts:
            raise GoogleAPIError("not found", status=404)
        self.sent.append(draft_id)
        return {"message_id": "sent-1", "thread_id": "t1"}


class FakeCalendar:
    def __init__(self, busy: list[tuple[datetime, datetime]] | None = None, events_list: list[dict[str, Any]] | None = None) -> None:
        self.busy = busy or []
        self.events_list = events_list or []  # existing events (titles) returned by list_events
        self.events: list[dict[str, Any]] = []

    async def list_events(self, time_min, time_max, calendar_id=None):
        return self.events_list

    async def query_freebusy(self, time_min, time_max, calendar_ids=None):
        return [(s, e) for s, e in self.busy if e > time_min and s < time_max]

    async def create_event(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append(kwargs)
        return {"event_id": "ev1", "html_link": "https://cal/ev1", "meet_link": "", "status": "confirmed"}


class FakeLLM:
    """Returns canned objects keyed by tool name; an Exception value is raised instead."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def structured_call(self, *, system, user, schema, tool_name, tool_description, tier="fast"):
        self.calls.append({"tool": tool_name, "system": system, "user": user, "tier": tier})
        response = self.responses.get(tool_name)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise LLMOutputError(f"no canned response for {tool_name}")
        return response if isinstance(response, schema) else schema.model_validate(response)


def next_weekday(days_ahead: int = 3) -> datetime:
    """Midnight UTC of the first Mon-Thu at least ``days_ahead`` days from now."""
    day = (datetime.now(UTC) + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
    while day.weekday() > 3:
        day += timedelta(days=1)
    return day


CLASSIFY_MEETING = {
    "category": "meeting_request", "confidence": 0.93, "priority_score": 3,
    "reasoning": "Sender proposes a call.", "summary": "Alice wants a sync.", "requires_reply": True,
}  # fmt: skip
CLASSIFY_ACTION = {**CLASSIFY_MEETING, "category": "action_required", "summary": "Review the doc."}
DRAFT_OK = {"body": "Hi Alice, thanks for reaching out. Happy to help.", "confidence": 0.9}
