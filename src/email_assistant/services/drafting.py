"""Reply drafting with deterministic guardrails.

The LLM only writes the *body*. Recipients, subject, threading headers and signature are produced
by code, and every draft passes :class:`DraftGuardrails` before it is allowed into Gmail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from email_assistant.config import Settings
from email_assistant.core.pii import TOKEN_RE, PiiMasker
from email_assistant.exceptions import LLMError, LLMRateLimitError, LLMUnavailableError
from email_assistant.llm.base import StructuredLLM
from email_assistant.llm.prompts import build_email_block, drafting_prompts
from email_assistant.schemas.llm import ClassificationResult, EmailDraft
from email_assistant.schemas.parsed_email import ParsedEmail
from email_assistant.services.memory import MemoryContext
from email_assistant.scheduling.slots import Interval, format_interval
from email_assistant.schemas.scheduling import SchedulingDecision

logger = structlog.get_logger(__name__)

TOOL_NAME = "draft_reply"
TOOL_DESCRIPTION = "Provide the body of the e-mail reply for the user to review."

_PLACEHOLDER_RE = re.compile(
    r"\[(?:[A-Z][A-Z _]{2,}|your name|insert[^\]]*|name|date|time)\]|\{\{[^}]+\}\}|<(?:insert|your)[^>]*>|\bTODO\b|lorem ipsum",
    re.I,
)
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class GuardrailIssue:
    """A guardrail finding. ``blocking`` issues prevent the draft from being created at all."""

    code: str
    message: str
    blocking: bool


class DraftGuardrails:
    """Static safety checks on generated reply bodies."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    def check(self, body: str, *, original: ParsedEmail, own_email: str, draft: EmailDraft | None = None) -> list[GuardrailIssue]:
        issues: list[GuardrailIssue] = []
        stripped = body.strip()
        if len(stripped) < 5:
            issues.append(GuardrailIssue("empty_body", "Draft body is empty or trivially short", True))
        if len(stripped) > self._max_chars:
            issues.append(GuardrailIssue("too_long", f"Draft exceeds {self._max_chars} characters", True))
        if _PLACEHOLDER_RE.search(stripped):
            issues.append(GuardrailIssue("placeholder", "Draft contains unfilled template placeholders", True))
        if TOKEN_RE.search(stripped):
            issues.append(GuardrailIssue("unresolved_pii_token", "Draft contains unresolved redaction placeholders", True))

        quoted = {a.lower() for a in _EMAIL_RE.findall(f"{original.body_text} {original.subject}")}
        allowed_addresses = original.known_addresses | {own_email.lower()} | quoted
        for address in {a.lower() for a in _EMAIL_RE.findall(stripped)}:
            if address not in allowed_addresses:
                issues.append(
                    GuardrailIssue("introduced_address", "Draft mentions an e-mail address not present in the conversation", True)
                )
                break

        source_text = f"{original.body_text} {original.subject}".lower()
        for url in _URL_RE.findall(stripped):
            if url.lower().rstrip(".,;") not in source_text:
                issues.append(GuardrailIssue("introduced_url", "Draft contains a link that was not in the original email", False))
                break

        if draft is not None:
            if draft.needs_user_input:
                issues.append(GuardrailIssue("needs_user_input", "Model flagged missing information; user must review", False))
            if draft.makes_commitments:
                issues.append(GuardrailIssue("makes_commitments", "Draft commits the user to something", False))
            if draft.confidence < 0.5:
                issues.append(GuardrailIssue("low_confidence", f"Draft confidence is low ({draft.confidence:.2f})", False))
        return issues


@dataclass
class DraftResult:
    """Outcome of :meth:`DraftingService.draft_reply`."""

    subject: str
    body: str
    draft: EmailDraft | None
    issues: list[GuardrailIssue] = field(default_factory=list)
    fallback: bool = False

    @property
    def blocked(self) -> bool:
        return (self.draft is None and not self.fallback) or any(i.blocking for i in self.issues)

    @property
    def warnings(self) -> list[str]:
        return [i.code for i in self.issues if not i.blocking]

def reply_subject(subject: str) -> str:
    """``Re:``-prefix a subject exactly once."""
    clean = subject.strip() or "(no subject)"
    return clean if re.match(r"^(re|aw|sv)\s*:", clean, re.I) else f"Re: {clean}"

def deterministic_reply(email: ParsedEmail, decision: SchedulingDecision | None, settings: Settings) -> str:
    """Safe template reply used when the AI cannot write one. Never invents facts or commitments."""
    first = (email.sender_name.split() or ["there"])[0].strip(",") or "there"
    tz = settings.tzinfo
    if decision is not None and decision.kind == "accept" and decision.chosen_start and decision.chosen_end:
        slot = format_interval(Interval(decision.chosen_start, decision.chosen_end), tz)
        return f"Hi {first},\n\nThanks for reaching out. {slot} works for me. A calendar invitation will follow shortly."
    if decision is not None and decision.kind == "counter":
        reasons = set(decision.reasons)
        if "clash" in reasons:
            why = "I already have another meeting scheduled at that time, so the two clash."
        elif reasons & {"unavailable", "time_off"}:
            why = "that time is outside the hours I am available."
        elif "too_soon" in reasons:
            why = "that is too short notice for me."
        else:
            why = "that time does not work for me."
        if decision.availability_lines:
            lines = "\n".join(f"- {line}" for line in decision.availability_lines)
            return f"Hi {first},\n\nThanks for reaching out. Unfortunately {why} I am available at these times:\n{lines}\n\nPlease let me know what suits you."
        if decision.alternatives:
            options = "\n".join(f"- {format_interval(Interval(a, b), tz)}" for a, b in decision.alternatives)
            return (
                f"Hi {first},\n\nThanks for reaching out. Unfortunately {why} "
                f"Would one of these suit you instead?\n{options}\n\nPlease let me know which works best."
            )
        return f"Hi {first},\n\nThanks for reaching out. Unfortunately {why} I will check my calendar and come back to you with some times shortly."
    if decision is not None and decision.kind == "cancel":
        return f"Hi {first},\n\nThanks for letting me know. I have noted the cancellation."
    return f"Hi {first},\n\nThank you for your email. I have received it and will get back to you shortly."

class DraftingService:
    """Generates context-aware reply drafts."""

    def __init__(self, llm: StructuredLLM, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._guardrails = DraftGuardrails(settings.max_draft_chars)

    async def draft_reply(
        self,
        email: ParsedEmail,
        classification: ClassificationResult,
        masker: PiiMasker,
        memory: MemoryContext,
        *,
        own_email: str,
        scheduling_facts: str | None = None,
        decision: SchedulingDecision | None = None,
        knowledge_text: str = "",
        allow_fallback: bool = False,
    ) -> DraftResult:
        """Rate limits / outages raise (caller defers). With ``allow_fallback`` or unusable model output a
        deterministic template reply is used instead, always flagged for human review."""
        subject = reply_subject(email.subject)
        system, user = drafting_prompts(
            user_name=self._settings.user_name,
            now=datetime.now(UTC).astimezone(self._settings.tzinfo),
            email_block=build_email_block(email, masker),
            classification=classification,
            memory_text=memory.render(),
            scheduling_facts=scheduling_facts,
            default_tone=self._settings.default_tone,
            custom_instructions=self._settings.custom_instructions,
            knowledge_text=knowledge_text,
            business_context=self._settings.business_context,
        )
        try:
            draft = await self._llm.structured_call(
                system=system,
                user=user,
                schema=EmailDraft,
                tool_name=TOOL_NAME,
                tool_description=TOOL_DESCRIPTION,
                tier="quality",
            )
        except (LLMRateLimitError, LLMUnavailableError):
            if not allow_fallback:
                raise
            logger.error("drafting_template_fallback_after_outage", message_id=email.message_id)
            return self._template_result(email, subject, decision, own_email)
        except LLMError as exc:
            logger.error("drafting_template_fallback", message_id=email.message_id, error=type(exc).__name__)
            return self._template_result(email, subject, decision, own_email)

        body = masker.unmask(draft.body).strip()
        issues = self._guardrails.check(body, original=email, own_email=own_email, draft=draft)
        return DraftResult(subject=subject, body=f"{body}\n\n{self._settings.signature}".strip(), draft=draft, issues=issues)

    def _template_result(self, email: ParsedEmail, subject: str, decision: SchedulingDecision | None, own_email: str) -> DraftResult:
        body = deterministic_reply(email, decision, self._settings)
        draft = EmailDraft(body=body, tone="formal", confidence=0.4, needs_user_input=True, makes_commitments=False)
        issues = self._guardrails.check(body, original=email, own_email=own_email, draft=draft)
        issues.append(GuardrailIssue("template_reply", "AI unavailable: a generic template reply was prepared", False))
        return DraftResult(subject=subject, body=f"{body}\n\n{self._settings.signature}".strip(), draft=draft, issues=issues, fallback=True)