"""Read/maintenance queries for the UI: processed-email history, details, reprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from email_assistant.db.repositories import AuditRepository, EmailRepository, ApprovalRepository, FeedbackRepository
from email_assistant.db.session import Database
from email_assistant.schemas.enums import EmailStatus


@dataclass(frozen=True)
class EmailRow:
    """Flat, display-friendly view of a processed email."""

    id: int
    received_at: datetime | None
    sender: str
    subject: str
    status: str
    category: str | None
    confidence: float | None
    priority: int | None
    summary: str | None
    attempts: int
    error: str | None
    processed_at: datetime | None = None  # when the assistant first handled it
    updated_at: datetime | None = None  # last status change (draft saved, sent, approved...)
    reply: str = "-"  # human-readable state of the reply
    invite: str = "-"  # human-readable state of the calendar invite
    reason: str = ""  # machine reason for the current outcome (see ui.logic.reason_text)


def describe_action(approvals: list) -> str:
    """Latest approval of one kind -> '-' / 'Draft saved - waiting for you' / 'Sent (automatically)' ..."""
    if not approvals:
        return "-"
    last = approvals[-1]
    who = {"policy:auto": "automatically", "gmail-sync": "from Gmail"}.get(last.decided_by or "", "by you")
    sending = last.action_type == "send_email"
    return {
        "pending": "Draft saved - waiting for you" if sending else "Waiting for your approval",
        "executed": f"{'Sent' if sending else 'Created'} ({who})",
        "approved": "Approved - running",
        "failed": "Failed - can be retried",
        "rejected": "Rejected" if last.decided_by != "gmail-sync" else "Draft deleted in Gmail",
        "expired": "Expired",
    }.get(last.status, last.status)

class InboxService:
    """Queries over the processed-email table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def status_counts(self) -> dict[str, int]:
        async with self._db.session() as session:
            return await EmailRepository(session).count_by_status()

    async def recent(
        self,
        limit: int = 200,
        statuses: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        search: str = "",
    ) -> list[EmailRow]:
        async with self._db.session() as session:
            records = await EmailRepository(session).list_recent(limit, statuses, categories)
            reasons = await AuditRepository(session).last_finish_reasons([r.id for r in records])
            approvals = await ApprovalRepository(session).list_for_emails([r.id for r in records])
        by_email: dict[int, dict[str, list]] = {}
        for a in approvals:
            by_email.setdefault(a.email_id, {}).setdefault(a.action_type, []).append(a)
        rows = [
            EmailRow(
                id=r.id, received_at=r.received_at, sender=r.sender_email, subject=r.subject_redacted,
                status=r.status, category=r.category, confidence=r.confidence, priority=r.priority_score,
                summary=r.summary, attempts=r.attempts, error=r.last_error, processed_at=r.created_at, updated_at=r.updated_at,
                reply=describe_action(by_email.get(r.id, {}).get("send_email", [])),
                invite=describe_action(by_email.get(r.id, {}).get("create_calendar_event", [])),
                reason=reasons.get(r.id, ""),
            )
            for r in records
        ]
        needle = search.strip().lower()
        if needle:
            rows = [r for r in rows if needle in f"{r.sender} {r.subject} {r.summary or ''}".lower()]
        return rows

    async def details(self, email_id: int) -> dict[str, Any] | None:
        async with self._db.session() as session:
            rec = await EmailRepository(session).get(email_id)
            if rec is None:
                return None
            audit = await AuditRepository(session).for_email(email_id)
        return {
            "reasoning": rec.reasoning,
            "action_items": rec.action_items or [],
            "audit": [{"time": a.created_at, "actor": a.actor, "event": a.event, "detail": a.detail} for a in audit],
        }

    async def reprocess(self, email_id: int) -> bool:
        """Queue an email for another attempt (it is picked up again if it is still unread in Gmail)."""
        async with self._db.session() as session:
            rec = await EmailRepository(session).get(email_id)
            if rec is None:
                return False
            rec.status = EmailStatus.FAILED.value
            rec.attempts = 0
            rec.last_error = None
            return True

    async def record_feedback(self, email_id: int, correct: bool, actual_category: str | None = None) -> bool:
        """Store a human verdict on the AI's category (ground truth for accuracy metrics)."""
        async with self._db.session() as session:
            rec = await EmailRepository(session).get(email_id)
            if rec is None or not rec.category:
                return False
            actual = rec.category if correct else (actual_category or rec.category)
            await FeedbackRepository(session).upsert(email_id, rec.category, actual)
            await AuditRepository(session).add(
                "classification_feedback", email_id=email_id, actor="user",
                detail={"predicted": rec.category, "actual": actual, "correct": actual == rec.category},
            )
            return True

    async def feedback_for(self, email_id: int) -> str | None:
        """The human-verified category for an email, if any."""
        async with self._db.session() as session:
            row = await FeedbackRepository(session).for_email(email_id)
        return row.actual if row else None
