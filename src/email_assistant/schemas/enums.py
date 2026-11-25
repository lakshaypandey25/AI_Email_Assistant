"""Enumerations shared by schemas, persistence and services."""

from __future__ import annotations

from enum import StrEnum


class EmailCategory(StrEnum):
    """Routing buckets produced by the classifier."""

    ACTION_REQUIRED = "action_required"
    MEETING_REQUEST = "meeting_request"
    INFORMATION_ONLY = "information_only"
    SPAM_MARKETING = "spam_marketing"
    PRIORITY_ESCALATION = "priority_escalation"


class EmailStatus(StrEnum):
    """Processing lifecycle of an email."""

    RECEIVED = "received"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    IGNORED = "ignored"
    FAILED = "failed"
    DEFERRED = "deferred"  # transient outage / rate limit: retried automatically later


class RiskLevel(StrEnum):
    """Risk class of a tool / action. Drives the human-in-the-loop policy."""

    READ = "read"  # no side effects
    LOW = "low"  # reversible side effects (e.g. create a Gmail *draft*)
    HIGH = "high"  # irreversible / externally visible (send email, calendar invite)


class ActionType(StrEnum):
    """Actions that can be recorded in the approval ledger."""

    CREATE_DRAFT = "create_draft"
    SEND_EMAIL = "send_email"
    CREATE_CALENDAR_EVENT = "create_calendar_event"


class ApprovalStatus(StrEnum):
    """Approval ledger state machine.

    PENDING -> APPROVED -> EXECUTED | FAILED (FAILED may be re-approved)
    PENDING -> REJECTED | EXPIRED
    AUTO_APPROVED is used for low-risk actions executed without a human.
    """

    PENDING = "pending"
    APPROVED = "approved"
    AUTO_APPROVED = "auto_approved"
    EXECUTED = "executed"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"
