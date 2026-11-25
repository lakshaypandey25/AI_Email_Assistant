"""Repository layer: all SQL lives here; services never build queries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from email_assistant.core.pii import scrub_structure
from email_assistant.db.models import (
    ApprovalRequest,
    AuditLog,
    EmailRecord,
    MemoryEntry,
    ThreadState,
    utcnow,
    KnowledgeEntry,
    ClassificationFeedback,
    LlmCallMetric,
)
from email_assistant.schemas.enums import ApprovalStatus


class EmailRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, email_id: int) -> EmailRecord | None:
        return await self._s.get(EmailRecord, email_id)

    async def get_by_gmail_id(self, gmail_message_id: str) -> EmailRecord | None:
        result = await self._s.execute(select(EmailRecord).where(EmailRecord.gmail_message_id == gmail_message_id))
        return result.scalar_one_or_none()

    async def upsert(self, gmail_message_id: str, **fields: Any) -> EmailRecord:
        """Fetch-or-create by Gmail id, applying ``fields``."""
        record = await self.get_by_gmail_id(gmail_message_id)
        if record is None:
            record = EmailRecord(gmail_message_id=gmail_message_id, attempts=0, **fields)
            self._s.add(record)
            await self._s.flush()
        else:
            for key, value in fields.items():
                setattr(record, key, value)
        return record

    async def recent_for_sender(self, sender_email: str, exclude_id: int | None, limit: int = 5) -> Sequence[EmailRecord]:
        stmt = select(EmailRecord).where(EmailRecord.sender_email == sender_email.lower())
        if exclude_id is not None:
            stmt = stmt.where(EmailRecord.id != exclude_id)
        stmt = stmt.order_by(EmailRecord.created_at.desc(), EmailRecord.id.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()

    async def get_many(self, ids: Iterable[int]) -> Sequence[EmailRecord]:
        id_list = list(ids)
        if not id_list:
            return []
        return (await self._s.execute(select(EmailRecord).where(EmailRecord.id.in_(id_list)))).scalars().all()

    async def list_recent(
        self, limit: int = 200, statuses: Iterable[str] | None = None, categories: Iterable[str] | None = None
    ) -> Sequence[EmailRecord]:
        stmt = select(EmailRecord)
        if statuses:
            stmt = stmt.where(EmailRecord.status.in_(list(statuses)))
        if categories:
            stmt = stmt.where(EmailRecord.category.in_(list(categories)))
        stmt = stmt.order_by(EmailRecord.created_at.desc(), EmailRecord.id.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()

    async def ids_by_status(self, status: str) -> list[tuple[str, datetime]]:
        """(gmail_message_id, updated_at) of emails in ``status`` - used to retry deferred emails."""
        rows = await self._s.execute(
            select(EmailRecord.gmail_message_id, EmailRecord.updated_at).where(EmailRecord.status == status)
        )
        return [(gid, updated) for gid, updated in rows.all()]

    async def count_by_status(self) -> dict[str, int]:
        rows = await self._s.execute(select(EmailRecord.status, func.count()).group_by(EmailRecord.status))
        return {status: count for status, count in rows.all()}


class ThreadRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, thread_id: str, state: str, last_email_id: int | None) -> None:
        row = await self._s.get(ThreadState, thread_id)
        if row is None:
            self._s.add(ThreadState(thread_id=thread_id, state=state, last_email_id=last_email_id))
        else:
            row.state = state
            row.last_email_id = last_email_id


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, event: str, *, email_id: int | None = None, actor: str = "system", detail: dict[str, Any] | None = None) -> None:
        """Append an audit row. ``detail`` is PII-scrubbed before storage."""
        self._s.add(AuditLog(email_id=email_id, actor=actor, event=event, detail=scrub_structure(detail or {})))

    async def last_finish_reasons(self, email_ids: Iterable[int]) -> dict[int, str]:
        """Latest ``email_finished`` reason per email (why it ended up in its current state)."""
        ids = list(email_ids)
        if not ids:
            return {}
        stmt = (
            select(AuditLog.email_id, AuditLog.detail)
            .where(AuditLog.event == "email_finished", AuditLog.email_id.in_(ids))
            .order_by(AuditLog.id)
        )
        return {eid: (detail or {}).get("reason", "") for eid, detail in (await self._s.execute(stmt)).all()}

    async def count_events(self, since: datetime) -> dict[str, int]:
        stmt = select(AuditLog.event, func.count()).where(AuditLog.created_at >= since).group_by(AuditLog.event)
        return {event: count for event, count in (await self._s.execute(stmt)).all()}

    async def details_for(self, event: str, since: datetime) -> Sequence[dict[str, Any]]:
        stmt = select(AuditLog.detail).where(AuditLog.event == event, AuditLog.created_at >= since)
        return [d or {} for d in (await self._s.execute(stmt)).scalars().all()]

    async def for_email(self, email_id: int) -> Sequence[AuditLog]:
        stmt = select(AuditLog).where(AuditLog.email_id == email_id).order_by(AuditLog.id)
        return (await self._s.execute(stmt)).scalars().all()


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, scope: str, kind: str, content: str) -> MemoryEntry:
        entry = MemoryEntry(scope=scope, kind=kind, content=content)
        self._s.add(entry)
        await self._s.flush()
        return entry

    async def list_recent(self, limit: int = 200) -> Sequence[MemoryEntry]:
        stmt = select(MemoryEntry).order_by(MemoryEntry.created_at.desc(), MemoryEntry.id.desc()).limit(limit)
        return (await self._s.execute(stmt)).scalars().all()

    async def delete(self, entry_id: int) -> bool:
        entry = await self._s.get(MemoryEntry, entry_id)
        if entry is None:
            return False
        await self._s.delete(entry)
        return True

    async def for_scopes(self, scopes: Iterable[str], limit: int = 10) -> Sequence[MemoryEntry]:
        stmt = (
            select(MemoryEntry)
            .where(MemoryEntry.scope.in_(list(scopes)))
            .order_by(MemoryEntry.created_at.desc(), MemoryEntry.id.desc())
            .limit(limit)
        )
        return (await self._s.execute(stmt)).scalars().all()


class ApprovalRepository:
    """Approval ledger with atomic state transitions (safe under concurrent approvers)."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, **fields: Any) -> ApprovalRequest:
        row = ApprovalRequest(**fields)
        self._s.add(row)
        await self._s.flush()
        return row

    async def get(self, approval_id: int) -> ApprovalRequest | None:
        return await self._s.get(ApprovalRequest, approval_id)

    async def list_by_status(self, statuses: Iterable[ApprovalStatus]) -> Sequence[ApprovalRequest]:
        stmt = (
            select(ApprovalRequest)
            .where(ApprovalRequest.status.in_([s.value for s in statuses]))
            .order_by(ApprovalRequest.id)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def list_for_email(self, email_id: int, statuses: Iterable[ApprovalStatus]) -> Sequence[ApprovalRequest]:
        stmt = (
            select(ApprovalRequest)
            .where(ApprovalRequest.email_id == email_id, ApprovalRequest.status.in_([s.value for s in statuses]))
            .order_by(ApprovalRequest.id)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def list_for_emails(self, email_ids: Iterable[int]) -> Sequence[ApprovalRequest]:
        ids = list(email_ids)
        if not ids:
            return []
        stmt = select(ApprovalRequest).where(ApprovalRequest.email_id.in_(ids)).order_by(ApprovalRequest.id)
        return (await self._s.execute(stmt)).scalars().all()

    async def list_since(self, since: datetime) -> Sequence[ApprovalRequest]:
        stmt = select(ApprovalRequest).where(ApprovalRequest.created_at >= since).order_by(ApprovalRequest.id)
        return (await self._s.execute(stmt)).scalars().all()

    async def count_pending_for_email(self, email_id: int) -> int:
        stmt = select(func.count()).where(
            ApprovalRequest.email_id == email_id, ApprovalRequest.status == ApprovalStatus.PENDING.value
        )
        return int((await self._s.execute(stmt)).scalar_one())

    async def claim(self, approval_id: int, actor: str, allowed_from: Iterable[ApprovalStatus]) -> bool:
        """Atomically move to APPROVED. Returns False if another actor won the race / state is wrong."""
        stmt = (
            update(ApprovalRequest)
            .where(
                ApprovalRequest.id == approval_id,
                ApprovalRequest.status.in_([s.value for s in allowed_from]),
            )
            .values(status=ApprovalStatus.APPROVED.value, decided_at=utcnow(), decided_by=actor, error=None)
        )
        result = await self._s.execute(stmt)
        return result.rowcount == 1  # type: ignore[attr-defined]

    async def reject(self, approval_id: int, actor: str, reason: str | None) -> bool:
        stmt = (
            update(ApprovalRequest)
            .where(
                ApprovalRequest.id == approval_id,
                ApprovalRequest.status.in_([ApprovalStatus.PENDING.value, ApprovalStatus.FAILED.value]),
            )
            .values(
                status=ApprovalStatus.REJECTED.value,
                decided_at=utcnow(),
                decided_by=actor,
                decision_reason=reason,
            )
        )
        result = await self._s.execute(stmt)
        return result.rowcount == 1  # type: ignore[attr-defined]

    async def finish(
        self,
        approval_id: int,
        status: ApprovalStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status.value, "result": result, "error": error}
        if status is ApprovalStatus.EXECUTED:
            values["executed_at"] = utcnow()
        await self._s.execute(update(ApprovalRequest).where(ApprovalRequest.id == approval_id).values(**values))

    async def expire_stale(self, now: datetime) -> int:
        stmt = (
            update(ApprovalRequest)
            .where(
                ApprovalRequest.status == ApprovalStatus.PENDING.value,
                ApprovalRequest.expires_at.is_not(None),
                ApprovalRequest.expires_at < now,
            )
            .values(status=ApprovalStatus.EXPIRED.value, decided_at=now, decided_by="system")
        )
        result = await self._s.execute(stmt)
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add(self, title: str, content: str) -> KnowledgeEntry:
        entry = KnowledgeEntry(title=title.strip()[:200], content=content.strip(), enabled=True)
        self._s.add(entry)
        await self._s.flush()
        return entry

    async def list_all(self) -> Sequence[KnowledgeEntry]:
        return (await self._s.execute(select(KnowledgeEntry).order_by(KnowledgeEntry.id))).scalars().all()

    async def list_enabled(self) -> Sequence[KnowledgeEntry]:
        stmt = select(KnowledgeEntry).where(KnowledgeEntry.enabled.is_(True)).order_by(KnowledgeEntry.id)
        return (await self._s.execute(stmt)).scalars().all()

    async def set_enabled(self, entry_id: int, enabled: bool) -> bool:
        entry = await self._s.get(KnowledgeEntry, entry_id)
        if entry is None:
            return False
        entry.enabled = enabled
        return True

    async def delete(self, entry_id: int) -> bool:
        entry = await self._s.get(KnowledgeEntry, entry_id)
        if entry is None:
            return False
        await self._s.delete(entry)
        return True

class MetricsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def add_llm_call(self, **fields: Any) -> None:
        self._s.add(LlmCallMetric(**fields))

    async def llm_calls_since(self, since: datetime) -> Sequence[LlmCallMetric]:
        stmt = select(LlmCallMetric).where(LlmCallMetric.created_at >= since).order_by(LlmCallMetric.id)
        return (await self._s.execute(stmt)).scalars().all()

class FeedbackRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(self, email_id: int, predicted: str, actual: str) -> None:
        """One verdict per email (latest wins)."""
        existing = (
            await self._s.execute(select(ClassificationFeedback).where(ClassificationFeedback.email_id == email_id))
        ).scalar_one_or_none()
        if existing is None:
            self._s.add(ClassificationFeedback(email_id=email_id, predicted=predicted, actual=actual))
        else:
            existing.predicted, existing.actual, existing.created_at = predicted, actual, utcnow()

    async def list_all(self) -> Sequence[ClassificationFeedback]:
        return (await self._s.execute(select(ClassificationFeedback).order_by(ClassificationFeedback.id))).scalars().all()

    async def for_email(self, email_id: int) -> ClassificationFeedback | None:
        stmt = select(ClassificationFeedback).where(ClassificationFeedback.email_id == email_id)
        return (await self._s.execute(stmt)).scalar_one_or_none()