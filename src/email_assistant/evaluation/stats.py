"""Live operational statistics computed from the database (no extra tracking needed)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from email_assistant.config import Settings
from email_assistant.db.repositories import (
    ApprovalRepository,
    AuditRepository,
    EmailRepository,
    FeedbackRepository,
    MetricsRepository,
)
from email_assistant.db.session import Database
from email_assistant.evaluation import metrics as m
from email_assistant.schemas.enums import ApprovalStatus, EmailCategory

LABELS = [c.value for c in EmailCategory]
_DECIDED = {ApprovalStatus.EXECUTED.value, ApprovalStatus.APPROVED.value, ApprovalStatus.FAILED.value}


class EvaluationService:
    """Aggregates throughput, human-in-the-loop, reliability and accuracy metrics for a time window."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self._s = settings

    async def snapshot(self, days: int = 30) -> dict[str, Any]:
        since = datetime.now(UTC) - timedelta(days=days)
        async with self._db.session() as session:
            emails = [e for e in await EmailRepository(session).list_recent(5000) if e.created_at >= since]
            approvals = await ApprovalRepository(session).list_since(since)
            events = await AuditRepository(session).count_events(since)
            modes = await AuditRepository(session).details_for("response_mode_decided", since)
            classified = await AuditRepository(session).details_for("classified", since)
            calls = await MetricsRepository(session).llm_calls_since(since)
            feedback = await FeedbackRepository(session).list_all()
        return {
            "days": days,
            "operations": self._operations(emails, modes, classified),
            "human": self._human(approvals, events),
            "reliability": self._reliability(calls, events, classified),
            "quality": self._quality(feedback),
        }

    # ------------------------------------------------------------------ throughput & routing
    @staticmethod
    def _operations(emails: list[Any], modes: list[dict[str, Any]], classified: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(emails)
        by_status = Counter(e.status for e in emails)
        by_category = Counter(e.category for e in emails if e.category)
        confidences = [e.confidence for e in emails if e.confidence is not None]
        mode_counts = Counter(d.get("mode", "?") for d in modes)
        reasons = Counter(d.get("reason", "?") for d in modes if d.get("mode") == "none")
        answered = mode_counts.get("reply", 0) + mode_counts.get("schedule", 0)
        decided = sum(mode_counts.values())
        handled_without_human = by_status.get("completed", 0) + by_status.get("ignored", 0)
        return {
            "total": total,
            "by_status": dict(by_status),
            "by_category": dict(by_category),
            "avg_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "auto_handled_rate": m.rate(handled_without_human, total),
            "needs_review_rate": m.rate(by_status.get("needs_review", 0), total),
            "response_modes": dict(mode_counts),
            "reply_rate": m.rate(answered, decided),
            "no_reply_reasons": dict(reasons),
            "classifier_sources": dict(Counter(d.get("source", "?") for d in classified)),
        }

    # ------------------------------------------------------------------ human in the loop
    def _human(self, approvals: list[Any], events: dict[str, int]) -> dict[str, Any]:
        sends = [a for a in approvals if a.action_type == "send_email"]
        invites = [a for a in approvals if a.action_type == "create_calendar_event"]

        def split(rows: list[Any]) -> dict[str, int]:
            return dict(Counter(r.status for r in rows))

        decided = [a for a in sends if a.status in _DECIDED | {"rejected"}]
        accepted = sum(1 for a in decided if a.status in _DECIDED)
        latencies = [
            (a.decided_at - a.created_at).total_seconds()
            for a in approvals
            if a.decided_at and a.status in _DECIDED | {"rejected"} and a.decided_by != "policy:auto"
        ]
        now = datetime.now(UTC)
        overdue = [a for a in approvals if a.status in {"pending", "failed"} and now - a.created_at > timedelta(hours=self._s.sla_hours)]
        return {
            "send_status": split(sends),
            "invite_status": split(invites),
            "draft_acceptance_rate": m.rate(accepted, len(decided)),
            "median_time_to_decision_min": (m.median(latencies) / 60.0) if latencies else None,
            "pending": sum(1 for a in approvals if a.status == "pending"),
            "overdue_vs_sla": len(overdue),
            "sla_hours": self._s.sla_hours,
            "conflicts_blocked": events.get("action_blocked_conflict", 0),
            "invites_created": sum(1 for a in invites if a.status == "executed"),
            "emails_sent": sum(1 for a in sends if a.status == "executed"),
        }

    # ------------------------------------------------------------------ reliability & cost
    @staticmethod
    def _reliability(calls: list[Any], events: dict[str, int], classified: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(calls)
        ok_latencies = [c.latency_ms for c in calls if c.ok and c.valid and c.latency_ms]
        per_model: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "ok": 0, "tokens": 0, "latency_sum": 0, "latency_n": 0})
        for c in calls:
            row = per_model[c.model]
            row["calls"] += 1
            row["ok"] += 1 if c.ok and c.valid else 0
            row["tokens"] += c.total_tokens
            if c.latency_ms:
                row["latency_sum"] += c.latency_ms
                row["latency_n"] += 1
        return {
            "llm_calls": total,
            "api_success_rate": m.rate(sum(1 for c in calls if c.ok), total),
            "valid_output_rate": m.rate(sum(1 for c in calls if c.ok and c.valid), total),
            "rate_limit_hits": sum(1 for c in calls if c.error_type == "RateLimit"),
            "fallback_model_calls": sum(1 for c in calls if c.fallback_used and c.ok and c.valid),
            "total_tokens": sum(c.total_tokens for c in calls),
            "latency_p50_ms": m.percentile(ok_latencies, 50),
            "latency_p95_ms": m.percentile(ok_latencies, 95),
            "per_model": {
                name: {
                    "calls": int(r["calls"]),
                    "success_rate": m.rate(r["ok"], r["calls"]),
                    "tokens": int(r["tokens"]),
                    "avg_latency_ms": m.rate(r["latency_sum"], r["latency_n"]),
                }
                for name, r in per_model.items()
            },
            "emails_deferred": events.get("email_deferred", 0),
            "rules_fallbacks": events.get("rules_fallback_used", 0),
            "template_replies": events.get("template_reply_used", 0),
            "calendar_fallbacks": events.get("calendar_unavailable_fallback", 0),
            "drafts_blocked_by_guardrails": events.get("draft_blocked", 0),
        }

    # ------------------------------------------------------------------ accuracy from human verdicts
    @staticmethod
    def _quality(feedback: list[Any]) -> dict[str, Any]:
        pairs = [(f.actual, f.predicted) for f in feedback]
        stats = m.per_class(pairs, LABELS)
        return {
            "labelled": len(pairs),
            "accuracy": m.accuracy(pairs),
            "macro_f1": m.macro_f1(stats),
            "per_class": stats,
            "confusion": m.confusion_matrix(pairs, LABELS),
        }