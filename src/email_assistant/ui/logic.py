"""Framework-free helpers for the UI (unit-testable without Streamlit)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from email_assistant.config import Settings

GMAIL_DRAFTS_URL = "https://mail.google.com/mail/#drafts"

DAYS: list[tuple[str, str]] = [
    ("mon", "Mon"), ("tue", "Tue"), ("wed", "Wed"), ("thu", "Thu"), ("fri", "Fri"), ("sat", "Sat"), ("sun", "Sun"),
]  # fmt: skip

CUSTOM_QUERY = "Custom Gmail search"
GMAIL_PRESETS: dict[str, str] = {
    "Unread in the Primary inbox, last 2 days (recommended)": "in:inbox is:unread newer_than:2d -category:promotions -category:social",
    "Unread in the Primary inbox, last 7 days": "in:inbox is:unread newer_than:7d -category:promotions -category:social",
    "Every unread email in the inbox": "in:inbox is:unread",
    "Starred emails in the inbox": "in:inbox is:starred",
}

STATUS_LABELS: dict[str, str] = {
    "awaiting_approval": "🟡 Awaiting approval",
    "needs_review": "🟠 Needs review",
    "completed": "🟢 Completed",
    "ignored": "⚪ Ignored",
    "failed": "🔴 Failed",
    "processing": "🔵 Processing",
    "received": "🔵 Received",
    "deferred": "⏳ Deferred (retrying)",
}

_NON_CHAT_MARKERS = ("whisper", "orpheus", "guard", "tts", "embed", "safeguard")


def validate_prefs(candidate: dict[str, Any]) -> str | None:
    """Return a human-readable error if ``candidate`` is not a valid set of customizations."""
    try:
        Settings(**candidate)  # type: ignore[arg-type]
    except ValidationError as exc:
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return None


def validate_client_secret(raw: bytes) -> tuple[bool, str]:
    """Check an uploaded Google OAuth client file. Only *Desktop app* clients work with the local sign-in flow."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False, "This is not a valid JSON file."
    if not isinstance(data, dict):
        return False, "Unexpected file format."
    if "installed" in data:
        return True, f"Desktop OAuth client for project '{data['installed'].get('project_id', '?')}'."
    if "web" in data:
        return False, (
            "This is a *Web application* client, which causes 'redirect_uri_mismatch'. In Google Cloud Console create "
            "an OAuth client of type **Desktop app** and upload that file instead."
        )
    return False, "Not a Google OAuth client file (expected an 'installed' section)."


def filter_chat_models(model_ids: Iterable[str]) -> list[str]:
    """Drop speech / safety / embedding models from a provider model list."""
    return sorted(m for m in model_ids if not any(marker in m.lower() for marker in _NON_CHAT_MARKERS))


def lines_to_csv(text: str) -> str:
    """Text area (one entry per line) -> comma list."""
    return ",".join(p.strip() for p in text.replace(",", "\n").splitlines() if p.strip())


def csv_to_lines(csv: str) -> str:
    return "\n".join(p.strip() for p in csv.split(",") if p.strip())


def signature_for_editing(raw: str) -> str:
    """Stored signature uses a literal backslash-n; show real line breaks in the editor."""
    return raw.replace("\\n", "\n")


def signature_for_storage(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\\n")


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)

EVENT_LABELS: dict[str, str] = {
    "email_ingested": "Email read",
    "classified": "Classified",
    "response_mode_decided": "Decided how to respond",
    "scheduling_decision": "Calendar checked",
    "draft_saved": "Reply saved as a Gmail draft",
    "approval_requested": "Waiting for your approval",
    "action_auto_approved": "Allowed automatically by your settings",
    "approval_granted": "You approved",
    "action_executed": "Done (draft saved / reply sent / invite created)",
    "reply_sent_from_gmail": "You sent the reply directly from Gmail",
    "approval_rejected": "Rejected (or draft deleted in Gmail)",
    "action_failed": "Action failed - can be retried",
    "action_blocked_conflict": "Invite blocked: calendar no longer free",
    "email_deferred": "Parked - AI/Google busy, retrying automatically",
    "rules_fallback_used": "AI unavailable - keyword rules used",
    "template_reply_used": "AI unavailable - template reply used",
    "calendar_unavailable_fallback": "Calendar unavailable - holding reply prepared",
    "email_finished": "Finished",
    "email_failed": "Failed",
    "draft_blocked": "Reply blocked by safety checks",
}


def event_label(event: str) -> str:
    return EVENT_LABELS.get(event, event.replace("_", " ").capitalize())

CATEGORY_LABELS: dict[str, str] = {
    "action_required": "Action required",
    "meeting_request": "Meeting request",
    "information_only": "Information only (FYI)",
    "spam_marketing": "Spam / marketing",
    "priority_escalation": "Priority escalation",
}
REPLY_CATEGORY_OPTIONS = ["action_required", "meeting_request", "priority_escalation", "information_only"]

MODE_LABELS: dict[str, str] = {
    "none": "No response needed",
    "reply": "Reply only",
    "schedule": "Reply + calendar",
    "human_only": "Needs your judgement",
}
REASON_LABELS: dict[str, str] = {
    "model_decision": "AI judged no reply was needed (FYI / thanks / notification)",
    "automated_sender_never_answered": "Automated / no-reply sender",
    "category_disabled_in_reply_policy": "Category switched off in Reply policy",
}


def pct(value: float | None, digits: int = 0) -> str:
    """Format a 0..1 rate as a percentage; ``-`` when undefined."""
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 0, suffix: str = "") -> str:
    return "-" if value is None else f"{value:,.{digits}f}{suffix}"

FINISH_REASONS: dict[str, str] = {
    "sender_not_authorized": "Not an authorized sender - skipped (the AI never saw it)",
    "sender_not_verified": "Sender address could not be verified (SPF/DKIM/DMARC) - skipped",
    "sender_on_ignore_list": "Sender is on your ignore list",
    "sent_by_self": "Sent by you",
    "spam_marketing": "Spam / marketing",
    "calendar_invitation_answer_in_calendar": "Calendar invitation - answer it with Accept/Decline in Calendar",
    "no_readable_text": "No readable text (image or attachment only) - please look at it yourself",
    "already_answered_by_you": "You already replied in this thread",
    "low_confidence_or_ai_unavailable": "AI unsure or unavailable - needs your review",
    "needs_human_judgement": "Needs your personal judgement - nothing drafted",
}


def reason_text(reason: str) -> str:
    """Friendly explanation of the machine reason stored when an email finished."""
    if not reason:
        return ""
    if reason in FINISH_REASONS:
        return FINISH_REASONS[reason]
    if reason.startswith("no_response:"):
        return "No reply needed - " + REASON_LABELS.get(reason.split(":", 1)[1], reason.split(":", 1)[1].replace("_", " "))
    if ":" in reason:  # e.g. "meeting_request:schedule"
        category, mode = reason.split(":", 1)
        return f"{CATEGORY_LABELS.get(category, category)} - {MODE_LABELS.get(mode, mode).lower()}"
    return reason.replace("_", " ")