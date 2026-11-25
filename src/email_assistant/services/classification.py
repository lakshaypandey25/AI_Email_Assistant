"""Email classification: heuristic pre-filter -> LLM tool call -> validated result (with safe fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import structlog
import re

from email_assistant.config import Settings
from email_assistant.core.pii import PiiMasker
from email_assistant.exceptions import LLMError, LLMRateLimitError, LLMUnavailableError
from email_assistant.llm.base import StructuredLLM
from email_assistant.llm.prompts import build_email_block, classification_prompts
from email_assistant.schemas.enums import EmailCategory
from email_assistant.schemas.llm import ClassificationResult
from email_assistant.schemas.parsed_email import ParsedEmail
from email_assistant.services.memory import MemoryContext

logger = structlog.get_logger(__name__)

TOOL_NAME = "classify_email"
TOOL_DESCRIPTION = "Record the classification, priority, summary and action items of the e-mail."


@dataclass(frozen=True)
class ClassificationOutcome:
    """Classification plus provenance."""

    result: ClassificationResult
    source: Literal["llm", "heuristic", "rules"]

    @property
    def degraded(self) -> bool:
        """True when the keyword rules were used (always sent to human review)."""
        return self.source == "rules"


_MEETING_RE = re.compile(r"\b(meeting|call|catch[- ]?up|sync|demo|zoom|teams|google meet|appointment|schedule|reschedule|availability)\b", re.I)
_TIME_RE = re.compile(r"\b(\d{1,2}(:\d{2})?\s?(am|pm)|monday|tuesday|wednesday|thursday|friday|tomorrow|next week)\b", re.I)
_URGENT_RE = re.compile(r"\b(urgent|asap|immediately|critical|outage|emergency|production (is )?down|deadline today)\b", re.I)
_REQUEST_RE = re.compile(r"(\?|\b(please|could you|can you|would you|kindly|need you|let me know|send me)\b)", re.I)


def rules_classify(email: ParsedEmail) -> ClassificationResult:
    """Keyword fallback used when no AI is available. Deliberately low confidence (0.5) -> goes to review."""
    text = f"{email.subject}\n{email.body_text}"
    if email.is_automated or email.has_list_unsubscribe:
        category = EmailCategory.INFORMATION_ONLY
    elif _URGENT_RE.search(text):
        category = EmailCategory.PRIORITY_ESCALATION
    elif _MEETING_RE.search(text) and _TIME_RE.search(text):
        category = EmailCategory.MEETING_REQUEST
    elif _REQUEST_RE.search(text):
        category = EmailCategory.ACTION_REQUIRED
    else:
        category = EmailCategory.INFORMATION_ONLY
    return ClassificationResult(
        category=category,
        confidence=0.5,
        priority_score=4 if category is EmailCategory.PRIORITY_ESCALATION else 3,
        reasoning="Keyword rules were used because the AI model was unavailable; please review.",
        summary=(email.subject or "No subject")[:200],
        requires_reply=category is not EmailCategory.INFORMATION_ONLY,
        action_items=[],
    )


class ClassificationService:
    """Categorises emails with confidence, priority and extracted action items."""

    def __init__(self, llm: StructuredLLM, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    @staticmethod
    def heuristic_bulk_mail(email: ParsedEmail) -> ClassificationResult | None:
        """Cheap, high-precision rule: RFC-compliant bulk mail (List-Unsubscribe + bulk/list headers)."""
        if email.has_list_unsubscribe and email.is_bulk:
            return ClassificationResult(
                category=EmailCategory.SPAM_MARKETING,
                confidence=0.9,
                priority_score=1,
                reasoning="Bulk mailing-list headers (List-Unsubscribe + Precedence/List-Id) detected.",
                summary=(email.subject or "Bulk mail")[:200],
                requires_reply=False,
                action_items=[],
            )
        return None

    async def classify(
        self, email: ParsedEmail, masker: PiiMasker, memory: MemoryContext, *, allow_fallback: bool = False
    ) -> ClassificationOutcome:
        """Rate limits / outages raise so the caller can DEFER and retry; with ``allow_fallback`` (retry window
        exhausted) the keyword rules are used. Invalid model output always falls back to the rules."""
        if self._settings.enable_heuristic_prefilter:
            rule = self.heuristic_bulk_mail(email)
            if rule is not None:
                logger.info("classified_by_heuristic", message_id=email.message_id)
                return ClassificationOutcome(rule, "heuristic")

        system, user = classification_prompts(
            user_name=self._settings.user_name,
            now=datetime.now(UTC).astimezone(self._settings.tzinfo),
            email_block=build_email_block(email, masker),
            memory_text=memory.render(),
            business_context=self._settings.business_context,
        )
        try:
            result = await self._llm.structured_call(
                system=system,
                user=user,
                schema=ClassificationResult,
                tool_name=TOOL_NAME,
                tool_description=TOOL_DESCRIPTION,
                tier="fast",
            )
            return ClassificationOutcome(result, "llm")
        except (LLMRateLimitError, LLMUnavailableError):
            if not allow_fallback:
                raise
            logger.error("classification_rules_fallback_after_outage", message_id=email.message_id)
        except LLMError as exc:
            logger.error("classification_rules_fallback", message_id=email.message_id, error=type(exc).__name__)
        return ClassificationOutcome(rules_classify(email), "rules")
