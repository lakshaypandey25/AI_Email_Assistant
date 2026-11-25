"""Live statistics, human feedback accuracy and the labelled benchmark."""

from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_settings
from email_assistant.evaluation.dataset import CASES
from email_assistant.evaluation.runner import build_email, expected_start, run_benchmark
from email_assistant.evaluation.store import load_runs, save_run
from email_assistant.exceptions import LLMRateLimitError
from email_assistant.services.knowledge import KnowledgeItem, rank, tokens
from email_assistant.services.telemetry import MetricsRecorder
from fakes import CLASSIFY_ACTION, DRAFT_OK, FakeLLM, make_raw
from test_hitl_and_agent import app_with


# --------------------------------------------------------------------------- live operations
async def test_snapshot_counts_outcomes_modes_and_human_decisions(db):
    spam = {**CLASSIFY_ACTION, "category": "spam_marketing", "requires_reply": False}
    silent = {**CLASSIFY_ACTION, "requires_reply": False, "response_mode": "none"}
    messages = [make_raw(msg_id="a"), make_raw(msg_id="b"), make_raw(msg_id="c")]
    classifications = iter([CLASSIFY_ACTION, silent, spam])

    class Sequenced(FakeLLM):
        async def structured_call(self, **kw):
            if kw["tool_name"] == "classify_email":
                self.responses["classify_email"] = next(classifications)
            return await super().structured_call(**kw)

    app, _, _ = app_with(db, messages=messages, llm=Sequenced({"draft_reply": DRAFT_OK}))
    for mid in ("a", "b", "c"):
        await app.agent.process_message(mid)

    snap = await app.evaluation.snapshot(30)
    ops = snap["operations"]
    assert ops["total"] == 3 and ops["by_status"] == {"awaiting_approval": 1, "completed": 1, "ignored": 1}
    assert ops["response_modes"] == {"reply": 1, "none": 1} and ops["reply_rate"] == pytest.approx(0.5)
    assert ops["auto_handled_rate"] == pytest.approx(2 / 3)
    assert snap["human"]["pending"] == 1 and snap["human"]["draft_acceptance_rate"] is None

    [group] = await app.hitl.list_groups()
    await app.hitl.approve_email(group.email_id, "u")
    human = (await app.evaluation.snapshot(30))["human"]
    assert human["emails_sent"] == 1 and human["draft_acceptance_rate"] == 1.0 and human["pending"] == 0
    assert human["median_time_to_decision_min"] is not None and human["overdue_vs_sla"] == 0


async def test_overdue_items_are_counted_against_the_sla(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, _, _ = app_with(db, llm=llm, sla_hours=0)
    await app.agent.process_message("m1")
    assert (await app.evaluation.snapshot(30))["human"]["overdue_vs_sla"] == 1


# --------------------------------------------------------------------------- reliability
async def test_reliability_metrics_from_recorded_llm_calls(db):
    app, *_ = app_with(db)
    record = MetricsRecorder(db)
    base = {"tool": "classify_email", "valid": True, "fallback_used": False, "latency_ms": 400, "total_tokens": 1000, "error_type": None}
    await record({**base, "model": "a", "ok": True})
    await record({**base, "model": "a", "ok": False, "valid": False, "latency_ms": 0, "total_tokens": 0, "error_type": "RateLimit"})
    await record({**base, "model": "b", "ok": True, "fallback_used": True, "latency_ms": 800, "total_tokens": 1200})
    r = (await app.evaluation.snapshot(30))["reliability"]
    assert r["llm_calls"] == 3 and r["rate_limit_hits"] == 1 and r["fallback_model_calls"] == 1
    assert r["api_success_rate"] == pytest.approx(2 / 3) and r["total_tokens"] == 2200 and r["latency_p50_ms"] == pytest.approx(600)
    assert r["per_model"]["a"]["calls"] == 2 and r["per_model"]["b"]["success_rate"] == 1.0


# --------------------------------------------------------------------------- accuracy from human feedback
async def test_feedback_drives_accuracy_and_confusion(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, *_ = app_with(db, messages=[make_raw(msg_id="a"), make_raw(msg_id="b")], llm=llm)
    await app.agent.process_message("a")
    await app.agent.process_message("b")
    assert await app.inbox.record_feedback(1, True) and await app.inbox.record_feedback(2, False, "information_only")
    assert await app.inbox.feedback_for(2) == "information_only"
    q = (await app.evaluation.snapshot(30))["quality"]
    assert q["labelled"] == 2 and q["accuracy"] == 0.5
    assert q["confusion"]["information_only"]["action_required"] == 1
    assert q["per_class"]["action_required"]["precision"] == 0.5


# --------------------------------------------------------------------------- knowledge ranking
def test_knowledge_ranking_prefers_title_and_content_overlap():
    items = [
        KnowledgeItem(1, "Refund policy", "Refunds within 30 days of purchase", True),
        KnowledgeItem(2, "Office hours", "Open Monday to Friday", True),
        KnowledgeItem(3, "Shipping", "Refund of shipping costs is not possible", False),  # disabled
    ]
    assert [i.id for i in rank(items, "What is your refund policy?")] == [1]
    assert rank(items, "completely unrelated gardening question") == []
    assert "refund" in tokens("Refunds? refund-policy")


# --------------------------------------------------------------------------- benchmark
class OracleLLM:
    """A perfect model: answers every benchmark case with the labelled truth (optionally with wrong dates)."""

    def __init__(self, settings, wrong_dates=False):
        self.settings, self.wrong_dates = settings, wrong_dates

    async def structured_call(self, *, system, user, schema, tool_name, tool_description, tier="fast"):
        case = next(c for c in CASES if f"subject: {c.subject}\n" in user)
        if tool_name == "classify_email":
            return schema.model_validate({
                "category": case.accepted_categories[0], "confidence": 0.9, "priority_score": 3, "reasoning": "oracle",
                "summary": case.subject, "requires_reply": case.expected_mode != "none", "response_mode": case.expected_mode,
            })
        start = expected_start(case.slot, datetime.now(UTC), self.settings)
        if self.wrong_dates:
            start += timedelta(days=1)
        return schema.model_validate({
            "is_meeting_request": True, "confidence": 0.9, "duration_minutes": case.duration_minutes,
            "proposed_slots": [{"start": start.strftime("%Y-%m-%dT%H:%M:%S")}],
        })


def test_benchmark_cases_are_wellformed():
    assert len({c.id for c in CASES}) == len(CASES) >= 16
    assert {cat for c in CASES for cat in c.accepted_categories} == {
        "action_required", "meeting_request", "information_only", "spam_marketing", "priority_escalation",
    }
    assert build_email(CASES[0]).subject == "Quick sync" and build_email(next(c for c in CASES if c.id == "info_ooo")).is_automated


async def test_perfect_model_scores_100_percent_and_results_are_stored(tmp_path):
    settings = make_settings()
    result = await run_benchmark(settings, OracleLLM(settings))
    assert result["errors"] == 0 and result["classification_accuracy"] == 1.0 and result["macro_f1"] == 1.0
    assert result["response_mode_accuracy"] == 1.0 and result["date_time_accuracy"] == 1.0
    assert result["duration_accuracy"] == 1.0 and result["injection_resistance"] == 1.0
    save_run(result, tmp_path / "runs.json")
    assert load_runs(tmp_path / "runs.json")[0]["cases"] == result["cases"]


async def test_weekday_verification_rescues_wrong_model_dates_where_the_email_names_the_day():
    settings = make_settings()
    result = await run_benchmark(settings, OracleLLM(settings, wrong_dates=True))
    rows = {r["id"]: r for r in result["rows"] if "slot_ok" in r}
    assert rows["meet_friday"]["slot_ok"] and rows["meet_monday"]["slot_ok"]  # 'Friday' / 'Monday' written in the email
    assert not rows["meet_tomorrow"]["slot_ok"]  # relative dates cannot be verified from the text alone
    assert result["date_time_accuracy"] == pytest.approx(0.5)


async def test_quick_mode_runs_a_subset_and_llm_errors_are_excluded_not_counted_wrong():
    class Busy:
        async def structured_call(self, **kw):
            raise LLMRateLimitError("busy")

    quick = await run_benchmark(make_settings(), Busy(), quick=True)
    assert quick["cases"] == 8 and quick["errors"] == 8 and quick["classification_accuracy"] is None
    assert quick["error_types"] == {"LLMRateLimitError": 8}