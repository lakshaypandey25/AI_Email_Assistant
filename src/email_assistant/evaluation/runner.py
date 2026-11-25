"""Benchmark runner: replays the labelled cases through the *real* classifier, routing rules and date resolution.

It measures the same code paths production uses (prompts, tool schemas, weekday verification, reply policy), with the
bulk-mail heuristic switched off so the language model itself is evaluated. No Gmail / Calendar access is needed.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any

from email_assistant.config import Settings
from email_assistant.core.pii import PiiMasker
from email_assistant.evaluation import metrics as m
from email_assistant.evaluation.dataset import CASES, QUICK_IDS, EvalCase
from email_assistant.exceptions import LLMError
from email_assistant.llm.base import StructuredLLM
from email_assistant.preprocessing.mime_parser import parse_raw_email
from email_assistant.scheduling.dates import next_occurrence
from email_assistant.schemas.enums import EmailCategory
from email_assistant.schemas.parsed_email import ParsedEmail
from email_assistant.services.classification import ClassificationService
from email_assistant.services.memory import MemoryContext
from email_assistant.services.routing import decide_mode
from email_assistant.services.scheduling import SchedulingService

LABELS = [c.value for c in EmailCategory]


def build_email(case: EvalCase) -> ParsedEmail:
    """Render the case as a real RFC 822 message and parse it with the production parser."""
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = case.sender, "me@example.com", case.subject
    msg["Date"] = "Mon, 03 Nov 2025 09:00:00 +0000"
    for key, value in case.headers.items():
        msg[key] = value
    msg.set_content(case.body)
    return parse_raw_email(msg.as_bytes(), gmail_id=f"eval-{case.id}", thread_id=f"eval-{case.id}", own_email="me@example.com")


def expected_start(spec: dict[str, Any], now: datetime, settings: Settings) -> datetime:
    """Independent ground truth for the expected meeting start (plain calendar arithmetic)."""
    tz = settings.tzinfo
    local = now.astimezone(tz)
    at = datetime(2000, 1, 1, int(spec.get("hour", 9)), int(spec.get("minute", 0)), tzinfo=tz)
    if spec["kind"] == "tomorrow":
        day = (local + timedelta(days=1)).date()
        return datetime.combine(day, at.timetz().replace(tzinfo=None), tzinfo=tz)
    return next_occurrence(local, int(spec["weekday"]), at)


async def run_benchmark(
    settings: Settings,
    llm: StructuredLLM,
    *,
    quick: bool = False,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run the benchmark and return a JSON-serialisable result."""
    eval_settings = settings.model_copy(update={"enable_heuristic_prefilter": False})
    classifier = ClassificationService(llm, eval_settings)
    scheduler = SchedulingService(llm, None, eval_settings)  # type: ignore[arg-type]  # extraction only, no calendar
    cases = [c for c in CASES if not quick or c.id in QUICK_IDS]
    now = datetime.now(UTC)
    tz = settings.tzinfo
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        if progress:
            progress(index / len(cases), f"{case.id} ({index + 1}/{len(cases)})")
        email = build_email(case)
        row: dict[str, Any] = {"id": case.id, "expected_category": case.accepted_categories[0], "expected_mode": case.expected_mode}
        started = time.monotonic()
        try:
            outcome = await classifier.classify(email, PiiMasker(), MemoryContext(), allow_fallback=False)
            c = outcome.result
            got = c.category.value
            row.update(got_category=got, source=outcome.source, confidence=c.confidence)
            row["category_ok"] = got in case.accepted_categories
            mode, _ = decide_mode(email, c, eval_settings)
            row.update(got_mode=mode, mode_ok=mode == case.expected_mode)
            if case.adversarial_forbidden:
                row["injection_resisted"] = got != case.adversarial_forbidden
            if case.slot and got == EmailCategory.MEETING_REQUEST.value:
                extraction = await scheduler.extract(email, PiiMasker())
                slots = (
                    scheduler._slots_from_extraction(
                        extraction, timedelta(minutes=extraction.duration_minutes or settings.default_meeting_minutes), now, f"{email.subject}\n{email.body_text}"
                    )
                    if extraction
                    else []
                )
                want = expected_start(case.slot, now, settings)
                have = slots[0].start.astimezone(tz) if slots else None
                row["slot_ok"] = bool(have and (have.date(), have.hour, have.minute) == (want.date(), want.hour, want.minute))
                row["slot_expected"] = want.strftime("%a %Y-%m-%d %H:%M")
                row["slot_got"] = have.strftime("%a %Y-%m-%d %H:%M") if have else "none"
                if case.duration_minutes:
                    row["duration_ok"] = bool(extraction and extraction.duration_minutes == case.duration_minutes)
        except LLMError as exc:
            row["error"] = f"{type(exc).__name__}"
        row["latency_s"] = round(time.monotonic() - started, 2)
        rows.append(row)
    if progress:
        progress(1.0, "done")

    scored = [r for r in rows if "category_ok" in r]
    pairs = [(r["expected_category"] if not r["category_ok"] else r["got_category"], r["got_category"]) for r in scored]
    stats = m.per_class(pairs, LABELS)
    slot_rows = [r for r in rows if "slot_ok" in r]
    dur_rows = [r for r in rows if "duration_ok" in r]
    inj_rows = [r for r in rows if "injection_resisted" in r]
    latencies = [r["latency_s"] for r in rows if "error" not in r]
    return {
        "timestamp": now.isoformat(),
        "models": {"fast": settings.groq_model_fast, "quality": settings.groq_model_quality},
        "cases": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "error_types": dict(Counter(r["error"] for r in rows if "error" in r)),
        "classification_accuracy": m.accuracy(pairs),
        "macro_f1": m.macro_f1(stats),
        "per_class": stats,
        "confusion": m.confusion_matrix(pairs, LABELS),
        "response_mode_accuracy": m.rate(sum(1 for r in scored if r.get("mode_ok")), len(scored)),
        "date_time_accuracy": m.rate(sum(1 for r in slot_rows if r["slot_ok"]), len(slot_rows)),
        "duration_accuracy": m.rate(sum(1 for r in dur_rows if r["duration_ok"]), len(dur_rows)),
        "injection_resistance": m.rate(sum(1 for r in inj_rows if r["injection_resisted"]), len(inj_rows)),
        "latency_p50_s": m.percentile(latencies, 50),
        "latency_p95_s": m.percentile(latencies, 95),
        "rows": rows,
    }