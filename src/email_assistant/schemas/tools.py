"""Input schemas for registry / MCP tools (single source of truth for tool contracts)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchMessagesInput(_ToolInput):
    query: str = Field(default="in:inbox is:unread", description="Gmail search query")
    max_results: int = Field(default=20, ge=1, le=100)


class GetMessageInput(_ToolInput):
    message_id: str = Field(min_length=1)


class CreateReplyDraftInput(_ToolInput):
    thread_id: str
    to: str
    subject: str
    body: str
    in_reply_to: str = ""
    references: str = ""
    cc: list[str] = Field(default_factory=list)


class SendDraftInput(_ToolInput):
    draft_id: str = Field(min_length=1)


class CheckAvailabilityInput(_ToolInput):
    time_min: datetime
    time_max: datetime
    calendar_ids: list[str] | None = None


class CreateEventInput(_ToolInput):
    summary: str
    description: str = ""
    start: datetime
    end: datetime
    timezone: str
    attendees: list[str] = Field(default_factory=list)
    add_meet_link: bool = True
    send_updates: Literal["all", "none"] = "all"
    idempotency_key: str
    calendar_id: str = "primary"

class DraftStatusInput(_ToolInput):
    draft_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    since: datetime

class ListEventsInput(_ToolInput):
      time_min: datetime
      time_max: datetime
      calendar_id: str | None = None


class ThreadRepliedInput(_ToolInput):
      thread_id: str = Field(min_length=1)
      since: datetime

class ProfileInput(_ToolInput):
    """No arguments."""


class FreeWindowsInput(_ToolInput):
    start_date: str = Field(default="", description="YYYY-MM-DD in the user's timezone (empty = today)")
    days: int = Field(default=5, ge=1, le=14)
    duration_minutes: int = Field(default=30, ge=5, le=480, description="Shortest free block to list")


class DraftForMessageInput(_ToolInput):
    message_id: str = Field(min_length=1)
    body: str = Field(min_length=1, description="Your reply text (the signature is appended automatically)")


class AiDraftInput(_ToolInput):
    message_id: str = Field(min_length=1)
    save_draft: bool = True