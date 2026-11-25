"""Async Google Calendar client: free/busy queries and idempotent event creation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from email_assistant.exceptions import GoogleAPIError
from email_assistant.integrations.base import GoogleApiClient

logger = structlog.get_logger(__name__)


def event_id_from_key(idempotency_key: str) -> str:
    """Deterministic Calendar event id (hex is a valid subset of Google's base32hex alphabet)."""
    return hashlib.sha1(idempotency_key.encode("utf-8")).hexdigest()


def _parse_rfc3339(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class CalendarClient(GoogleApiClient):
    """Google Calendar API v3 operations used by the assistant."""

    service_name = "calendar"
    service_version = "v3"

    async def query_freebusy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: list[str] | None = None,
    ) -> list[tuple[datetime, datetime]]:
        """Return merged-unaware busy intervals (UTC) across ``calendar_ids``."""
        ids = calendar_ids or [self._settings.calendar_id]
        body = {
            "timeMin": time_min.astimezone(UTC).isoformat(),
            "timeMax": time_max.astimezone(UTC).isoformat(),
            "timeZone": "UTC",
            "items": [{"id": cid} for cid in ids],
        }
        resp = await self._execute(lambda s: s.freebusy().query(body=body))
        busy: list[tuple[datetime, datetime]] = []
        for cid, data in (resp.get("calendars") or {}).items():
            errors = data.get("errors")
            if errors:
                raise GoogleAPIError(
                    f"Free/busy lookup failed for calendar {cid}: {errors[0].get('reason', 'unknown')}",
                    reason=errors[0].get("reason"),
                )
            for slot in data.get("busy", []):
                busy.append((_parse_rfc3339(slot["start"]), _parse_rfc3339(slot["end"])))
        return sorted(busy)

    async def list_events(self, time_min: datetime, time_max: datetime, calendar_id: str | None = None) -> list[dict[str, Any]]:
        """Existing events overlapping ``[time_min, time_max)`` (titles are only used for the user's own notes)."""
        resp = await self._execute(
            lambda s: s.events().list(
                calendarId=calendar_id or self._settings.calendar_id,
                timeMin=time_min.astimezone(UTC).isoformat(),
                timeMax=time_max.astimezone(UTC).isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=50,
            )
        )
        events: list[dict[str, Any]] = []
        for item in resp.get("items", []):
            if item.get("status") == "cancelled" or item.get("transparency") == "transparent":
                continue  # cancelled or marked "free": does not block time
            start = item.get("start", {})
            end = item.get("end", {})
            events.append({
                "title": item.get("summary") or "(busy)",
                "start": start.get("dateTime") or start.get("date", ""),
                "end": end.get("dateTime") or end.get("date", ""),
                "all_day": "date" in start,
            })
        return events

    async def create_event(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str,
        start: datetime,
        end: datetime,
        timezone: str,
        attendees: list[str],
        add_meet_link: bool,
        send_updates: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Create an event. Re-running with the same key returns the existing event (idempotent)."""
        event_id = event_id_from_key(idempotency_key)
        body: dict[str, Any] = {
            "id": event_id,
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": timezone},
            "attendees": [{"email": a} for a in attendees],
            "reminders": {"useDefault": True},
        }
        if add_meet_link:
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": uuid4().hex,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        try:
            event = await self._execute(
                lambda s: s.events().insert(
                    calendarId=calendar_id,
                    body=body,
                    sendUpdates=send_updates,
                    conferenceDataVersion=1 if add_meet_link else 0,
                )
            )
        except GoogleAPIError as exc:
            if exc.status != 409:  # 409 = id already exists -> previous attempt succeeded
                raise
            logger.info("calendar_event_already_exists", event_id=event_id)
            event = await self._execute(lambda s: s.events().get(calendarId=calendar_id, eventId=event_id))
        return {
            "event_id": event.get("id", event_id),
            "html_link": event.get("htmlLink", ""),
            "meet_link": event.get("hangoutLink", ""),
            "status": event.get("status", ""),
        }
