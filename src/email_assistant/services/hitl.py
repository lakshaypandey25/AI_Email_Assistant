"""Human-in-the-Loop controller: the approval ledger and the *only* path to HIGH-risk side effects.

Policy
------
* READ / LOW tools (e.g. creating a Gmail **draft**) execute immediately and are audited.
* HIGH tools (sending an email, creating a calendar event) are persisted as ``PENDING`` approvals and
  do nothing until a human approves them (CLI: ``approve <id>``).
* The user decides in Settings: ``send_mode`` = "draft" (replies stay Gmail drafts until approved) or "auto"
     (replies are sent immediately); ``auto_create_events`` does the same for calendar invites.
* ``force_review=True`` (set by the agent for risky replies) overrides automatic mode.
* Approvals expire after ``APPROVAL_TTL_HOURS``; state transitions are atomic (double-approval safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from email_assistant.config import Settings
from email_assistant.core.pii import PiiMasker
from email_assistant.db.repositories import ApprovalRepository, AuditRepository, EmailRepository
from email_assistant.db.session import Database
from email_assistant.exceptions import ApprovalStateError
from email_assistant.mcp_tools.names import TOOL_CHECK_AVAILABILITY, TOOL_CREATE_EVENT, TOOL_DRAFT_STATUS
from email_assistant.mcp_tools.registry import ToolRegistry
from email_assistant.schemas.enums import ActionType, ApprovalStatus, EmailStatus, RiskLevel

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SubmitResult:
    """Result of submitting an action to the ledger."""

    approval_id: int
    status: ApprovalStatus
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def executed(self) -> bool:
        return self.status is ApprovalStatus.EXECUTED

    @property
    def is_conflict(self) -> bool:
        return bool(self.error and self.error.startswith("CONFLICT"))


@dataclass(frozen=True)
class GroupResult:
    """Outcome of approving every pending action of one email together."""

    email_id: int
    results: list[SubmitResult]
    skipped: list[int]  # approval ids deliberately not run (reply withheld because the invite failed)

    @property
    def all_executed(self) -> bool:
        return bool(self.results) and not self.skipped and all(r.executed for r in self.results)


@dataclass(frozen=True)
class ApprovalView:
    """Read model for CLI / UI listings."""

    id: int
    email_id: int
    action_type: str
    status: str
    risk: str
    preview: str
    created_at: datetime
    expires_at: datetime | None
    error: str | None


@dataclass(frozen=True)
class ApprovalGroup:
    """All open approvals for one email, with context for the review screen."""

    email_id: int
    sender: str
    subject: str
    category: str | None
    priority: int | None
    summary: str | None
    items: list[ApprovalView]
    warnings: list[str]
    oldest: datetime | None = None  # when the oldest open item was created (for SLA tracking)


#: Execution order inside a group: the invite first, so the reply never promises an invite that failed.
_GROUP_ORDER = {ActionType.CREATE_CALENDAR_EVENT.value: 0, ActionType.SEND_EMAIL.value: 1}


def _storable_arguments(arguments: dict[str, Any], risk: RiskLevel) -> dict[str, Any]:
    """Keep full arguments only where they're needed to execute later (HIGH risk); omit bodies elsewhere."""
    if risk is RiskLevel.HIGH:
        return arguments
    return {k: (f"<omitted {len(v)} chars>" if k == "body" and isinstance(v, str) else v) for k, v in arguments.items()}


class HITLController:
    """Approval ledger + executor."""

    def __init__(self, db: Database, registry: ToolRegistry, settings: Settings) -> None:
        self._db = db
        self._registry = registry
        self._settings = settings

    # ------------------------------------------------------------------ submit
    async def submit(
        self,
        *,
        email_id: int,
        action_type: ActionType,
        tool_name: str,
        arguments: dict[str, Any],
        preview: str,
        force_review: bool = False,
    ) -> SubmitResult:
        """Record an intended action and execute it iff policy allows without a human."""
        spec = self._registry.get_spec(tool_name)
        needs_human = spec.risk is RiskLevel.HIGH and (force_review or not self._auto_allowed(action_type))
        now = datetime.now(UTC)
        async with self._db.session() as session:
            row = await ApprovalRepository(session).create(
                email_id=email_id,
                action_type=action_type.value,
                risk=spec.risk.value,
                status=(ApprovalStatus.PENDING if needs_human else ApprovalStatus.AUTO_APPROVED).value,
                tool_name=tool_name,
                arguments=_storable_arguments(arguments, spec.risk),
                preview=PiiMasker.redact(preview)[:1500] if spec.risk is not RiskLevel.HIGH else preview[:1500],
                expires_at=now + timedelta(hours=self._settings.approval_ttl_hours),
                decided_by=None if needs_human else "policy:auto",
                decided_at=None if needs_human else now,
            )
            approval_id = row.id
            await AuditRepository(session).add(
                "approval_requested" if needs_human else "action_auto_approved",
                email_id=email_id,
                detail={"approval_id": approval_id, "action": action_type.value, "risk": spec.risk.value},
            )
        if needs_human:
            logger.info("approval_pending", approval_id=approval_id, action=action_type.value)
            return SubmitResult(approval_id, ApprovalStatus.PENDING)
        return await self._run(approval_id, email_id, tool_name, arguments, actor="policy:auto")

    # ------------------------------------------------------------------ decisions
    def _auto_allowed(self, action_type: ActionType) -> bool:
        """The user's Settings decide which HIGH-risk actions may run without a click."""
        if action_type is ActionType.SEND_EMAIL:
            return self._settings.send_mode == "auto"
        if action_type is ActionType.CREATE_CALENDAR_EVENT:
            return self._settings.auto_create_events
        return False
    
    async def approve(self, approval_id: int, actor: str, *, ignore_conflicts: bool = False) -> SubmitResult:
        """Approve and execute a pending (or previously failed) action.

        For calendar invites availability is re-checked first; if the slot was taken in the meantime the
        action fails with a ``CONFLICT`` error unless ``ignore_conflicts`` is set.
        """
        await self.expire_stale()
        async with self._db.session() as session:
            repo = ApprovalRepository(session)
            row = await repo.get(approval_id)
            if row is None:
                raise ApprovalStateError(f"Approval {approval_id} does not exist")
            if not await repo.claim(approval_id, actor, (ApprovalStatus.PENDING, ApprovalStatus.FAILED)):
                raise ApprovalStateError(f"Approval {approval_id} is '{row.status}' and cannot be approved")
            email_id, tool_name, arguments = row.email_id, row.tool_name, dict(row.arguments)
            await AuditRepository(session).add("approval_granted", email_id=email_id, actor=actor, detail={"approval_id": approval_id})
        return await self._run(approval_id, email_id, tool_name, arguments, actor=actor, skip_preflight=ignore_conflicts)

    async def approve_email(self, email_id: int, actor: str, *, ignore_conflicts: bool = False) -> GroupResult:
        """Approve every open action of one email in a safe order (invite, then reply).

        If the invite cannot be created (e.g. calendar conflict) the reply is **not** sent, because it
        promises an invite.
        """
        async with self._db.session() as session:
            rows = await ApprovalRepository(session).list_for_email(email_id, (ApprovalStatus.PENDING, ApprovalStatus.FAILED))
            ordered = sorted(((r.id, r.action_type) for r in rows), key=lambda item: (_GROUP_ORDER.get(item[1], 9), item[0]))
        results: list[SubmitResult] = []
        skipped: list[int] = []
        invite_failed = False
        for approval_id, action_type in ordered:
            if action_type == ActionType.SEND_EMAIL.value and invite_failed:
                skipped.append(approval_id)
                continue
            try:
                outcome = await self.approve(approval_id, actor, ignore_conflicts=ignore_conflicts)
            except ApprovalStateError as exc:
                outcome = SubmitResult(approval_id, ApprovalStatus.FAILED, error=str(exc))
            results.append(outcome)
            if action_type == ActionType.CREATE_CALENDAR_EVENT.value and not outcome.executed:
                invite_failed = True
        return GroupResult(email_id, results, skipped)

    async def reject_email(self, email_id: int, actor: str, reason: str | None = None) -> int:
        """Reject every open action of one email. Returns how many were rejected."""
        async with self._db.session() as session:
            ids = [r.id for r in await ApprovalRepository(session).list_for_email(email_id, (ApprovalStatus.PENDING, ApprovalStatus.FAILED))]
        count = 0
        for approval_id in ids:
            try:
                await self.reject(approval_id, actor, reason)
                count += 1
            except ApprovalStateError:
                continue
        return count

    async def reject(self, approval_id: int, actor: str, reason: str | None = None) -> None:
        """Reject a pending action (nothing is executed)."""
        async with self._db.session() as session:
            repo = ApprovalRepository(session)
            row = await repo.get(approval_id)
            if row is None:
                raise ApprovalStateError(f"Approval {approval_id} does not exist")
            if not await repo.reject(approval_id, actor, reason):
                raise ApprovalStateError(f"Approval {approval_id} is '{row.status}' and cannot be rejected")
            await AuditRepository(session).add(
                "approval_rejected", email_id=row.email_id, actor=actor, detail={"approval_id": approval_id, "reason": reason}
            )
        await self._refresh_email_status(row.email_id)

    async def expire_stale(self) -> int:
        """Mark overdue pending approvals as expired."""
        async with self._db.session() as session:
            count = await ApprovalRepository(session).expire_stale(datetime.now(UTC))
        if count:
            logger.info("approvals_expired", count=count)
        return count

    async def reconcile_drafts(self) -> dict[str, int]:
        """Notice what the user did directly in Gmail and keep the ledger / History truthful.

        For every open SEND approval: if the draft was sent from Gmail the approval is closed as executed; if the
        draft was deleted it is closed as rejected. Gmail errors are ignored (checked again next time).
        """
        counts = {"sent": 0, "deleted": 0}
        async with self._db.session() as session:
            rows = [
                r for r in await ApprovalRepository(session).list_by_status((ApprovalStatus.PENDING, ApprovalStatus.FAILED))
                if r.action_type == ActionType.SEND_EMAIL.value and r.arguments.get("draft_id")
            ]
            emails = {e.id: e for e in await EmailRepository(session).get_many({r.email_id for r in rows})}
            todo = [(r.id, r.email_id, r.arguments["draft_id"], r.created_at, emails[r.email_id].thread_id) for r in rows if r.email_id in emails]
        for approval_id, email_id, draft_id, created_at, thread_id in todo:
            try:
                data = await self._registry.call(
                    TOOL_DRAFT_STATUS, {"draft_id": draft_id, "thread_id": thread_id, "since": created_at.isoformat()}
                )
            except Exception as exc:  # noqa: BLE001 - Gmail unavailable / not connected: try again later
                logger.warning("draft_reconcile_skipped", approval_id=approval_id, error_type=type(exc).__name__)
                continue
            status = data.get("status")
            if status == "sent":
                async with self._db.session() as session:
                    repo = ApprovalRepository(session)
                    if await repo.claim(approval_id, "gmail-sync", (ApprovalStatus.PENDING, ApprovalStatus.FAILED)):
                        await repo.finish(approval_id, ApprovalStatus.EXECUTED, result={"via": "sent directly from Gmail"})
                        await AuditRepository(session).add("reply_sent_from_gmail", email_id=email_id, actor="gmail-sync", detail={"approval_id": approval_id})
                        counts["sent"] += 1
                await self._refresh_email_status(email_id)
            elif status == "gone":
                try:
                    await self.reject(approval_id, "gmail-sync", "draft deleted in Gmail")
                    counts["deleted"] += 1
                except ApprovalStateError:
                    continue
        if any(counts.values()):
            logger.info("drafts_reconciled", **counts)
        return counts

    # ------------------------------------------------------------------ queries
    async def list_pending(self) -> list[ApprovalView]:
        await self.expire_stale()
        async with self._db.session() as session:
            rows = await ApprovalRepository(session).list_by_status((ApprovalStatus.PENDING, ApprovalStatus.FAILED))
        return [self._view(r) for r in rows]

    async def list_groups(self) -> list[ApprovalGroup]:
        """Open approvals grouped per email, with review warnings (for the UI / CLI)."""
        await self.expire_stale()
        async with self._db.session() as session:
            rows = await ApprovalRepository(session).list_by_status((ApprovalStatus.PENDING, ApprovalStatus.FAILED))
            emails = {e.id: e for e in await EmailRepository(session).get_many({r.email_id for r in rows})}
        by_email: dict[int, list[ApprovalView]] = {}
        for row in rows:
            by_email.setdefault(row.email_id, []).append(self._view(row))
        groups: list[ApprovalGroup] = []
        for email_id, items in by_email.items():
            items.sort(key=lambda v: (_GROUP_ORDER.get(v.action_type, 9), v.id))
            rec = emails.get(email_id)
            kinds = {i.action_type: i for i in items}
            warnings: list[str] = []
            invite = kinds.get(ActionType.CREATE_CALENDAR_EVENT.value)
            if invite and ActionType.SEND_EMAIL.value in kinds:
                warnings.append("The reply promises a calendar invite - approve both together so the invite is not forgotten.")
            if any(i.error and i.error.startswith("CONFLICT") for i in items):
                warnings.append("Your calendar changed since this was proposed. Reject, or approve anyway to double-book.")
            groups.append(
                ApprovalGroup(
                    email_id=email_id,
                    sender=rec.sender_email if rec else "",
                    subject=rec.subject_redacted if rec else "",
                    category=rec.category if rec else None,
                    priority=rec.priority_score if rec else None,
                    summary=rec.summary if rec else None,
                    items=items,
                    warnings=warnings,
                    oldest=min(i.created_at for i in items),
                )
            )
        return sorted(groups, key=lambda g: (-(g.priority or 0), g.email_id))

    async def get(self, approval_id: int) -> ApprovalView | None:
        async with self._db.session() as session:
            row = await ApprovalRepository(session).get(approval_id)
        return self._view(row) if row else None

    @staticmethod
    def _view(row: Any) -> ApprovalView:
        return ApprovalView(
            id=row.id,
            email_id=row.email_id,
            action_type=row.action_type,
            status=row.status,
            risk=row.risk,
            preview=row.preview,
            created_at=row.created_at,
            expires_at=row.expires_at,
            error=row.error,
        )

    # ------------------------------------------------------------------ execution
    async def _preflight(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Re-validate preconditions right before executing. Returns an error message to block, else ``None``."""
        if tool_name != TOOL_CREATE_EVENT:
            return None
        start = datetime.fromisoformat(str(arguments["start"]))
        end = datetime.fromisoformat(str(arguments["end"]))
        data = await self._registry.call(
            TOOL_CHECK_AVAILABILITY,
            {
                "time_min": start.isoformat(),
                "time_max": end.isoformat(),
                "calendar_ids": [arguments.get("calendar_id") or self._settings.calendar_id],
            },
        )
        if data.get("busy"):
            return f"CONFLICT: your calendar is no longer free on {start:%a %d %b %Y %H:%M}-{end:%H:%M}."
        return None

    async def _fail(self, approval_id: int, email_id: int, actor: str, message: str, event: str = "action_failed") -> SubmitResult:
        message = PiiMasker.redact(message)[:500]
        async with self._db.session() as session:
            await ApprovalRepository(session).finish(approval_id, ApprovalStatus.FAILED, error=message)
            await AuditRepository(session).add(event, email_id=email_id, actor=actor, detail={"approval_id": approval_id, "error": message})
        logger.error(event, approval_id=approval_id, error=message[:120])
        return SubmitResult(approval_id, ApprovalStatus.FAILED, error=message)

    async def _run(
        self,
        approval_id: int,
        email_id: int,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        actor: str,
        skip_preflight: bool = False,
    ) -> SubmitResult:
        try:
            if not skip_preflight:
                conflict = await self._preflight(tool_name, arguments)
                if conflict:
                    return await self._fail(approval_id, email_id, actor, conflict, event="action_blocked_conflict")
            result = await self._registry.call(tool_name, arguments, approved=True)
        except Exception as exc:  # noqa: BLE001 - any failure must be recorded, never lost
            return await self._fail(approval_id, email_id, actor, f"{type(exc).__name__}: {exc}")

        async with self._db.session() as session:
            await ApprovalRepository(session).finish(approval_id, ApprovalStatus.EXECUTED, result=result)
            await AuditRepository(session).add(
                "action_executed", email_id=email_id, actor=actor, detail={"approval_id": approval_id, "tool": tool_name}
            )
        logger.info("action_executed", approval_id=approval_id, tool=tool_name)
        await self._refresh_email_status(email_id)
        return SubmitResult(approval_id, ApprovalStatus.EXECUTED, result=result)

    async def _refresh_email_status(self, email_id: int) -> None:
        """Move an email from AWAITING_APPROVAL to COMPLETED once nothing is pending."""
        async with self._db.session() as session:
            pending = await ApprovalRepository(session).count_pending_for_email(email_id)
            record = await EmailRepository(session).get(email_id)
            if record and pending == 0 and record.status == EmailStatus.AWAITING_APPROVAL.value:
                record.status = EmailStatus.COMPLETED.value
