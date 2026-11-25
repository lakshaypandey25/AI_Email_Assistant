"""Decide how (and whether) to respond to an email.

The classifier proposes a ``response_mode``; this module applies deterministic business rules on top:
* automated / no-reply senders are never answered,
* categories not enabled under *Reply policy* are never answered,
* scheduling can be switched off (reply-only).
"""

from __future__ import annotations

from email_assistant.config import Settings
from email_assistant.schemas.llm import ClassificationResult
from email_assistant.schemas.parsed_email import ParsedEmail

NONE, REPLY, SCHEDULE, HUMAN_ONLY = "none", "reply", "schedule", "human_only"


def decide_mode(email: ParsedEmail, c: ClassificationResult, settings: Settings) -> tuple[str, str]:
    """Return ``(mode, reason)`` where mode is none | reply | schedule | human_only."""
    mode = c.effective_response_mode
    if mode == HUMAN_ONLY:
        return HUMAN_ONLY, "needs_human_judgement"
    if mode in (REPLY, SCHEDULE):
        if email.is_automated:
            return NONE, "automated_sender_never_answered"
        if c.category.value not in settings.reply_categories_set:
            return NONE, "category_disabled_in_reply_policy"
        if mode == SCHEDULE and not settings.enable_scheduling:
            return REPLY, "scheduling_disabled_reply_only"
    return mode, "model_decision"