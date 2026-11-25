"""EmailAgent: the end-to-end workflow for each email.

fetch -> clean -> memory -> classify -> decide response mode -> (schedule) -> draft -> HITL

Response modes (not every email gets a reply):
  none        nothing is sent (FYI, thanks, receipts, automated senders, categories switched off in Reply policy)
  reply       a drafted reply only - no calendar involvement
  schedule    calendar check + reply (+ invite when a slot is free)
  human_only  needs the user's own judgement - flagged, nothing drafted

Replies are always saved as Gmail DRAFTS first. Settings then decide: send_mode="draft" leaves the draft for the
user; "auto" sends it (unless it is risky). A reply that promises an invite is only sent once the invite exists.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest import result

import structlog

from email_assistant.config import Settings
from email_assistant.core.pii import PiiMasker
from email_assistant.core.senders import sender_matches
from email_assistant.db.models import EmailRecord
from email_assistant.db.repositories import AuditRepository, EmailRepository, ThreadRepository
from email_assistant.db.session import Database
from email_assistant.exceptions import (
    AuthenticationRequiredError,
    ConfigurationError,
    GoogleAPIError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from email_assistant.mcp_tools.names import (
    TOOL_CREATE_DRAFT,
    TOOL_CREATE_EVENT,
    TOOL_GET_MESSAGE,
    TOOL_SEARCH,
    TOOL_SEND_DRAFT,
    TOOL_THREAD_REPLIED,
)
from email_assistant.mcp_tools.registry import ToolRegistry
from email_assistant.schemas.enums import ActionType, EmailCategory, EmailStatus
from email_assistant.schemas.llm import ClassificationResult
from email_assistant.schemas.parsed_email import ParsedEmail
from email_assistant.schemas.scheduling import SchedulingDecision
from email_assistant.services.classification import ClassificationService
from email_assistant.services.drafting import DraftingService
from email_assistant.services.hitl import HITLController
from email_assistant.services.knowledge import KnowledgeService
from email_assistant.services.memory import MemoryContext, MemoryService
from email_assistant.services.routing import HUMAN_ONLY, NONE, SCHEDULE, decide_mode
from email_assistant.services.scheduling import SchedulingService

logger = structlog.get_logger(__name__)

_DONE = {EmailStatus.COMPLETED, EmailStatus.IGNORED, EmailStatus.AWAITING_APPROVAL, EmailStatus.NEEDS_REVIEW}
_STALE_PROCESSING = timedelta(minutes=10)

_CALENDAR_DOWN_FACTS = (
    "Your calendar could not be checked right now. Do NOT propose or accept any specific time. "
    "Say you will check your calendar and confirm shortly."
)
_UNREADABLE_TIME_FACTS = (
    "The sender wants to meet, but the proposed time could not be read reliably. Do NOT propose or accept any "
    "specific time. Say you will check your calendar and confirm shortly."
)


@dataclass
class RunSummary:
    """Aggregate result of one polling cycle."""

    counts: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class EmailAgent:
    """Processes inbox messages end-to-end."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        registry: ToolRegistry,
        classifier: ClassificationService,
        drafter: DraftingService,
        scheduler: SchedulingService,
        memory: MemoryService,
        hitl: HITLController,
        knowledge: KnowledgeService,
        own_email_provider: Callable[[], Awaitable[str]],
    ) -> None:
        self._s = settings
        self._db = db
        self._registry = registry
        self._classifier = classifier
        self._drafter = drafter
        self._scheduler = scheduler
        self._memory = memory
        self._hitl = hitl
        self._knowledge = knowledge
        self._own_email = own_email_provider

    # ------------------------------------------------------------------ polling
    async def _deferred_ids(self) -> list[str]:
        """Emails parked after a rate limit / outage whose retry time has come (even if no longer unread)."""
        cutoff = datetime.now(UTC) - timedelta(minutes=self._s.defer_retry_minutes)
        async with self._db.session() as session:
            rows = await EmailRepository(session).ids_by_status(EmailStatus.DEFERRED.value)
        return [gid for gid, updated in rows if updated <= cutoff]

    async def run_once(self, max_messages: int | None = None) -> RunSummary:
        """Process one batch: sync with Gmail, then deferred emails, then new unread mail."""
        try:
            await self._hitl.reconcile_drafts()  # replies the user sent / drafts deleted directly in Gmail
        except Exception:  # noqa: BLE001 - never block processing on the sync
            logger.warning("draft_reconcile_failed")
        deferred = await self._deferred_ids()
        try:
            found = await self._registry.call(
                TOOL_SEARCH, {"query": self._s.gmail_query, "max_results": max_messages or self._s.gmail_max_per_run}
            )
            fresh: list[str] = found["message_ids"]
        except GoogleAPIError:
            if not deferred:
                raise
            logger.error("gmail_search_unavailable_processing_deferred_only")
            fresh = []
        ids = list(dict.fromkeys([*deferred, *fresh]))
        sem = asyncio.Semaphore(self._s.max_concurrent_emails)

        async def _one(message_id: str) -> EmailStatus:
            async with sem:
                return await self.process_message(message_id)

        statuses = await asyncio.gather(*(_one(i) for i in ids))
        summary = RunSummary(Counter(s.value for s in statuses))
        logger.info("run_completed", total=summary.total, **dict(summary.counts))
        return summary

    async def run_forever(self, interval_seconds: int | None = None) -> None:
        """Poll until cancelled. Auth/config errors stop the loop; everything else is logged and retried."""
        interval = interval_seconds or self._s.poll_interval_seconds
        while True:
            try:
                await self._hitl.expire_stale()
                await self.run_once()
            except (AuthenticationRequiredError, ConfigurationError):
                logger.critical("fatal_configuration_or_auth_error_stopping")
                raise
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("poll_cycle_failed")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ single message
    async def process_message(self, message_id: str) -> EmailStatus:
        """Never raises (except cancellation / auth problems). Transient failures DEFER the email."""
        try:
            return await self._process(message_id)
        except asyncio.CancelledError:
            raise
        except (AuthenticationRequiredError, ConfigurationError):
            raise
        except (LLMRateLimitError, LLMUnavailableError) as exc:
            return await self._defer(message_id, f"AI temporarily unavailable ({type(exc).__name__}); will retry automatically")
        except GoogleAPIError as exc:
            if exc.retryable:
                return await self._defer(message_id, f"Google API temporarily unavailable (HTTP {exc.status}); will retry automatically")
            logger.exception("email_processing_failed", message_id=message_id, error_type=type(exc).__name__)
            await self._mark_failed(message_id, exc)
            return EmailStatus.FAILED
        except Exception as exc:  # noqa: BLE001
            logger.exception("email_processing_failed", message_id=message_id, error_type=type(exc).__name__)
            await self._mark_failed(message_id, exc)
            return EmailStatus.FAILED

    def _should_process(self, rec: EmailRecord) -> bool:
        status = EmailStatus(rec.status)
        now = datetime.now(UTC)
        if status in _DONE:
            return False
        if status is EmailStatus.FAILED:
            return rec.attempts < self._s.max_processing_attempts
        if status is EmailStatus.DEFERRED:
            return now - rec.updated_at >= timedelta(minutes=self._s.defer_retry_minutes)
        if status is EmailStatus.PROCESSING:
            return now - rec.updated_at > _STALE_PROCESSING
        return True

    async def _process(self, message_id: str) -> EmailStatus:
        async with self._db.session() as session:
            existing = await EmailRepository(session).get_by_gmail_id(message_id)
        if existing is not None and not self._should_process(existing):
            return EmailStatus(existing.status)
        # After DEFER_GIVE_UP_HOURS of failed retries use rules / templates instead of waiting forever.
        allow_fallback = (
            existing is not None
            and existing.status == EmailStatus.DEFERRED.value
            and datetime.now(UTC) - existing.created_at >= timedelta(hours=self._s.defer_give_up_hours)
        )

        data = await self._registry.call(TOOL_GET_MESSAGE, {"message_id": message_id})
        email = ParsedEmail.model_validate(data)

        async with self._db.session() as session:
            rec = await EmailRepository(session).upsert(
                message_id,
                thread_id=email.thread_id,
                sender_email=email.sender_email,
                subject_redacted=PiiMasker.redact(email.subject)[:500],
                received_at=email.date,
                status=EmailStatus.PROCESSING.value,
                last_error=None,
            )
            email_id = rec.id
            await AuditRepository(session).add("email_ingested", email_id=email_id, detail={"attempts_so_far": rec.attempts})

        if email.is_from_self:
            return await self._finish(email_id, email, EmailStatus.IGNORED, "sent_by_self")
        allowed = self._s.allowed_list
        if allowed:  # authorized senders only - checked BEFORE any AI call so nothing from strangers reaches the model
            if not sender_matches(allowed, email.sender_email):
                return await self._finish(email_id, email, EmailStatus.IGNORED, "sender_not_authorized")
            if self._s.require_authenticated_sender and email.authenticated is not True:
                return await self._finish(email_id, email, EmailStatus.IGNORED, "sender_not_verified")
        if sender_matches(self._s.ignore_list, email.sender_email):
            return await self._finish(email_id, email, EmailStatus.IGNORED, "sender_on_ignore_list")
        if email.has_calendar_invite:  # a calendar invitation is answered with Accept/Decline in Calendar, not by mail
            return await self._finish(email_id, email, EmailStatus.COMPLETED, "calendar_invitation_answer_in_calendar")
        if not email.body_text.strip():
            return await self._finish(email_id, email, EmailStatus.NEEDS_REVIEW, "no_readable_text")

        masker = PiiMasker()
        memory = await self._memory.get_context(email, email_id)
        outcome = await self._classifier.classify(email, masker, memory, allow_fallback=allow_fallback)
        c = self._apply_vip(email, outcome.result)
        await self._store_classification(email_id, c, outcome.source)
        if outcome.source == "rules":
            await self._audit(email_id, "rules_fallback_used", {"reason": "ai_unavailable_or_invalid"})

        if outcome.degraded or c.confidence < self._s.min_classification_confidence:
            return await self._finish(email_id, email, EmailStatus.NEEDS_REVIEW, "low_confidence_or_ai_unavailable")
        if c.category is EmailCategory.SPAM_MARKETING:
            return await self._finish(email_id, email, EmailStatus.IGNORED, "spam_marketing")
        if c.category is EmailCategory.PRIORITY_ESCALATION:
            logger.warning("priority_escalation", email_id=email_id, priority=c.priority_score)

        mode, reason = decide_mode(email, c, self._s)
        await self._audit(email_id, "response_mode_decided", {"mode": mode, "reason": reason})
        if mode == NONE:
            return await self._finish(email_id, email, EmailStatus.COMPLETED, f"no_response:{reason}")
        if mode == HUMAN_ONLY:
            return await self._finish(email_id, email, EmailStatus.NEEDS_REVIEW, reason)

        if await self._already_answered(email):
            return await self._finish(email_id, email, EmailStatus.COMPLETED, "already_answered_by_you")

        knowledge_text = await self._knowledge.retrieve(f"{email.subject}\n{email.body_text}", self._s.kb_max_snippets)
        if mode == SCHEDULE:
            status = await self._handle_meeting(email_id, email, c, masker, memory, knowledge_text, allow_fallback)
        else:
            status = await self._stage_reply(email_id, email, c, masker, memory, knowledge_text=knowledge_text, allow_fallback=allow_fallback)
        return await self._finish(email_id, email, status, f"{c.category.value}:{mode}")

    async def _already_answered(self, email: ParsedEmail) -> bool:
        """True if the user already replied in this thread (e.g. by hand) after this email arrived."""
        try:
            data = await self._registry.call(
                TOOL_THREAD_REPLIED, {"thread_id": email.thread_id, "since": email.date.isoformat()}
            )
            return bool(data.get("replied"))
        except Exception:  # noqa: BLE001 - if Gmail cannot tell us, carry on and draft
            return False

    def _apply_vip(self, email: ParsedEmail, c: ClassificationResult) -> ClassificationResult:
        """VIP senders are never treated as spam and always get priority >= 4."""
        if not sender_matches(self._s.vip_list, email.sender_email):
            return c
        updates: dict[str, object] = {"priority_score": max(c.priority_score, 4)}
        if c.category is EmailCategory.SPAM_MARKETING:
            updates["category"] = EmailCategory.INFORMATION_ONLY
            updates["response_mode"] = "none"
        return c.model_copy(update=updates)

    # ------------------------------------------------------------------ handlers
    async def _handle_meeting(
        self,
        email_id: int,
        email: ParsedEmail,
        c: ClassificationResult,
        masker: PiiMasker,
        memory: MemoryContext,
        knowledge_text: str,
        allow_fallback: bool,
    ) -> EmailStatus:
        extraction = await self._scheduler.extract(email, masker, allow_fallback=allow_fallback)
        if extraction is not None and not extraction.is_meeting_request:
            await self._audit(email_id, "meeting_not_detected_reply_only")
            return await self._stage_reply(email_id, email, c, masker, memory, knowledge_text=knowledge_text, allow_fallback=allow_fallback)
        if extraction is None or extraction.confidence < 0.5:
            await self._audit(email_id, "meeting_times_unreadable_holding_reply")
            decision = SchedulingDecision(kind="no_meeting", facts=_UNREADABLE_TIME_FACTS)
            return await self._stage_reply(
                email_id, email, c, masker, memory, scheduling_facts=decision.facts, decision=decision,
                knowledge_text=knowledge_text, allow_fallback=allow_fallback,
            )

        own = await self._own_email()
        try:
            decision = await self._scheduler.plan(email, extraction, own_email=own)
        except GoogleAPIError as exc:
            logger.error("calendar_unavailable_holding_reply", status=exc.status)
            await self._audit(email_id, "calendar_unavailable_fallback", {"status": exc.status})
            decision = SchedulingDecision(
                kind="no_meeting", facts=_CALENDAR_DOWN_FACTS,
                user_notes=["Your calendar could not be checked, so no time was proposed or accepted."],
            )
        await self._audit(email_id, "scheduling_decision", {"kind": decision.kind, "slots_offered": len(decision.alternatives)})

        prepared = await self._prepare_draft(
            email_id, email, c, masker, memory, scheduling_facts=decision.facts, decision=decision,
            knowledge_text=knowledge_text, allow_fallback=allow_fallback,
        )
        if prepared is None:
            return EmailStatus.NEEDS_REVIEW
        result, draft_id = prepared
        hold = False
        if decision.kind == "accept" and decision.event is not None:
            ev = decision.event
            invite = await self._hitl.submit(  # Settings decide: created now, or queued for approval
                email_id=email_id,
                action_type=ActionType.CREATE_CALENDAR_EVENT,
                tool_name=TOOL_CREATE_EVENT,
                arguments=ev.model_dump(mode="json"),
                preview=(
                    f"Create event '{ev.summary}' {ev.start.isoformat()} -> {ev.end.isoformat()} ({ev.timezone}); "
                    f"invite: {', '.join(ev.attendees)}; Meet link: {ev.add_meet_link}"
                ),
            )
            hold = not invite.executed  # the reply promises an invite: only auto-send if the invite really exists
        return await self._queue_send(email_id, email, c, result, draft_id, hold=hold, notes=decision.user_notes)

    async def _stage_reply(
        self, email_id: int, email: ParsedEmail, c: ClassificationResult, masker: PiiMasker, memory: MemoryContext,
        scheduling_facts: str | None = None, decision: SchedulingDecision | None = None,
        knowledge_text: str = "", allow_fallback: bool = False, notes: list[str] | None = None,
    ) -> EmailStatus:
        prepared = await self._prepare_draft(
            email_id, email, c, masker, memory, scheduling_facts=scheduling_facts, decision=decision,
            knowledge_text=knowledge_text, allow_fallback=allow_fallback,
        )
        if prepared is None:
            return EmailStatus.NEEDS_REVIEW
        result, draft_id = prepared
        return await self._queue_send(email_id, email, c, result, draft_id, notes=notes)

    async def _prepare_draft(
        self, email_id: int, email: ParsedEmail, c: ClassificationResult, masker: PiiMasker, memory: MemoryContext,
        *, scheduling_facts: str | None, decision: SchedulingDecision | None, knowledge_text: str, allow_fallback: bool,
    ):  # -> tuple[DraftResult, str] | None
        """Write the reply and save it as a Gmail DRAFT (low risk). ``None`` = blocked by guardrails."""
        own = await self._own_email()
        result = await self._drafter.draft_reply(
            email, c, masker, memory, own_email=own, scheduling_facts=scheduling_facts, decision=decision,
            knowledge_text=knowledge_text, allow_fallback=allow_fallback,
        )
        if result.fallback:
            await self._audit(email_id, "template_reply_used", {"reason": "ai_unavailable_or_invalid"})
        if result.blocked:
            await self._audit(email_id, "draft_blocked", {"issues": [i.code for i in result.issues]})
            return None

        created = await self._hitl.submit(
            email_id=email_id,
            action_type=ActionType.CREATE_DRAFT,
            tool_name=TOOL_CREATE_DRAFT,
            arguments={
                "thread_id": email.thread_id,
                "to": email.sender_email,
                "subject": result.subject,
                "body": result.body,
                "in_reply_to": email.rfc_message_id,
                "references": email.references,
            },
            preview=f"Draft reply to {email.sender_email}: {result.subject}",
        )
        if not created.executed or not created.result:
            await self._audit(email_id, "draft_creation_failed", {"error": created.error})
            if created.error and created.error.startswith(("GoogleAPIError", "GoogleRateLimitError", "ToolTransportError")):
                raise GoogleAPIError("Gmail draft creation unavailable", retryable=True)  # -> deferred, retried later
            raise RuntimeError(f"Gmail draft creation failed: {created.error}")
        await self._audit(email_id, "draft_saved", {"draft_id": created.result["draft_id"]})
        return result, created.result["draft_id"]

    async def _queue_send(
        self, email_id: int, email: ParsedEmail, c: ClassificationResult, result, draft_id: str, *, hold: bool = False,
        notes: list[str] | None = None,
    ) -> EmailStatus:
        """Send now (Settings: automatic mode) or leave the draft waiting for the user (draft mode / risky reply)."""
        risky = (
            bool(result.warnings)
            or (result.draft is not None and result.draft.makes_commitments)
            or c.category is EmailCategory.PRIORITY_ESCALATION
        )
        force = result.fallback or hold or (self._s.hold_risky_replies and risky)
        preview_body = result.body if len(result.body) <= 600 else result.body[:600] + "..."
        sent = await self._hitl.submit(
            email_id=email_id,
            action_type=ActionType.SEND_EMAIL,
            tool_name=TOOL_SEND_DRAFT,
            arguments={"draft_id": draft_id},
            preview=(
                "".join(f"NOTE: {note}\n" for note in (notes or []))
                + f"SEND to {email.sender_email} | {result.subject}"
                f"{' | WARNINGS: ' + ','.join(result.warnings) if result.warnings else ''}\n{preview_body}"
            ),
            force_review=force,
        )
        return EmailStatus.COMPLETED if sent.executed else EmailStatus.AWAITING_APPROVAL

    # ------------------------------------------------------------------ persistence helpers
    async def _store_classification(self, email_id: int, c: ClassificationResult, source: str) -> None:
        async with self._db.session() as session:
            rec = await EmailRepository(session).get(email_id)
            if rec is not None:
                rec.category = c.category.value
                rec.confidence = c.confidence
                rec.priority_score = c.priority_score
                rec.reasoning = c.reasoning
                rec.summary = c.summary
                rec.action_items = [a.model_dump() for a in c.action_items]
            await AuditRepository(session).add(
                "classified",
                email_id=email_id,
                detail={
                    "category": c.category.value, "confidence": c.confidence, "priority": c.priority_score,
                    "source": source, "response_mode": c.effective_response_mode,
                },
            )

    async def _finish(self, email_id: int, email: ParsedEmail, status: EmailStatus, reason: str) -> EmailStatus:
        async with self._db.session() as session:
            rec = await EmailRepository(session).get(email_id)
            if rec is not None:
                rec.status = status.value
            await ThreadRepository(session).upsert(email.thread_id, status.value, email_id)
            await AuditRepository(session).add("email_finished", email_id=email_id, detail={"status": status.value, "reason": reason})
        logger.info("email_processed", email_id=email_id, status=status.value, reason=reason)
        return status

    async def _audit(self, email_id: int, event: str, detail: dict | None = None) -> None:
        async with self._db.session() as session:
            await AuditRepository(session).add(event, email_id=email_id, detail=detail)

    async def _defer(self, message_id: str, reason: str) -> EmailStatus:
        """Park an email after a transient failure; it is retried automatically (attempts are NOT consumed)."""
        try:
            async with self._db.session() as session:
                rec = await EmailRepository(session).upsert(message_id, status=EmailStatus.DEFERRED.value)
                rec.last_error = PiiMasker.redact(reason)[:500]
                await AuditRepository(session).add("email_deferred", email_id=rec.id, detail={"reason": rec.last_error})
        except Exception:  # noqa: BLE001
            logger.exception("could_not_persist_deferral", message_id=message_id)
        logger.warning("email_deferred", message_id=message_id, reason=reason)
        return EmailStatus.DEFERRED

    async def _mark_failed(self, message_id: str, exc: Exception) -> None:
        try:
            async with self._db.session() as session:
                rec = await EmailRepository(session).upsert(message_id, status=EmailStatus.FAILED.value)
                rec.attempts = (rec.attempts or 0) + 1
                rec.last_error = PiiMasker.redact(f"{type(exc).__name__}: {exc}")[:500]
                await AuditRepository(session).add("email_failed", email_id=rec.id, detail={"error": rec.last_error})
        except Exception:  # noqa: BLE001
            logger.exception("could_not_persist_failure", message_id=message_id)