"""Schemas for meeting extraction (LLM output) and calendar payloads (code output)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from email_assistant.schemas.llm import _coerce_probability


class ProposedSlot(BaseModel):
    """One candidate time proposed by the sender, exactly as written in the email."""

    model_config = ConfigDict(extra="ignore")

    start: str = Field(
        description=(
            "Start as ISO 8601 local time WITHOUT offset, e.g. 2025-03-04T15:00:00. "
            "Resolve relative dates ('next Tuesday') using the provided current date."
        )
    )
    end: str | None = Field(default=None, description="End as ISO 8601 local time, if stated")
    weekday: str | None = Field(
        default=None,
        description="Weekday name EXACTLY as written in the email for this slot (e.g. 'Friday'), or null if none is mentioned.",
    )

class MeetingExtraction(BaseModel):
    """Output of the ``extract_meeting_request`` tool."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    is_meeting_request: bool
    request_type: Literal["new", "reschedule", "cancel"] = "new"
    confidence: float = Field(ge=0.0, le=1.0)
    title: str | None = Field(default=None, max_length=200, description="Short meeting title")
    duration_minutes: int | None = Field(default=None, ge=5, le=600)
    proposed_slots: list[ProposedSlot] = Field(default_factory=list, max_length=8)
    timezone_hint: str | None = Field(
        default=None,
        description="Timezone mentioned in the email: IANA name (Europe/Berlin) or abbreviation (PST, IST).",
    )
    attendee_emails: list[str] = Field(
        default_factory=list, description="Extra participants explicitly mentioned (placeholders allowed)"
    )
    location_preference: Literal["google_meet", "in_person", "phone", "unspecified"] = "unspecified"
    is_flexible: bool = Field(default=False, description="Sender said times are flexible / asks for availability")
    is_recurring: bool = Field(default=False, description="True if the sender asks for a repeating meeting (every week, daily, monthly...)")
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value: Any) -> Any:
        return _coerce_probability(value)

    @field_validator("duration_minutes", mode="before")
    @classmethod
    def _coerce_duration(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


class CalendarEventPayload(BaseModel):
    """Fully-resolved calendar event, ready for ``calendar_create_event``."""

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


class SchedulingDecision(BaseModel):
    """Deterministic outcome of availability checking."""

    kind: Literal["accept", "counter", "cancel", "no_meeting"]
    facts: str = Field(description="Plain-language facts handed to the drafting LLM")
    chosen_start: datetime | None = None
    chosen_end: datetime | None = None
    alternatives: list[tuple[datetime, datetime]] = Field(default_factory=list)
    event: CalendarEventPayload | None = None
    reasons: list[str] = Field(default_factory=list, description="Why a proposed time was refused: clash, buffer, unavailable, time_off, too_soon, too_far, past")
    user_notes: list[str] = Field(default_factory=list, description="Shown to the user in Approvals (may name your other meetings)")
    availability_lines: list[str] = Field(default_factory=list, description="One line per day with the user's free hours (from Settings > Scheduling)")
