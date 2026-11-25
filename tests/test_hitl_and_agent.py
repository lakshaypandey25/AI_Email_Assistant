import pytest

from conftest import make_settings
from email_assistant.container import build_container
from email_assistant.db.repositories import EmailRepository
from email_assistant.exceptions import ApprovalStateError, LLMOutputError
from email_assistant.schemas.enums import ActionType, ApprovalStatus, EmailStatus
from fakes import (
    CLASSIFY_ACTION, CLASSIFY_MEETING, DRAFT_OK, FakeCalendar, FakeGmail, FakeLLM, make_raw, next_weekday,
)


def app_with(db, *, messages=None, busy=None, llm=None, **settings_kw):
    gmail = FakeGmail(messages if messages is not None else [make_raw()])
    calendar = FakeCalendar(busy)
    app = build_container(make_settings(**settings_kw), gmail=gmail, calendar=calendar, llm=llm or FakeLLM(), db=db)
    return app, gmail, calendar


def meeting_llm(day, hour=10, **extra):
    extraction = {
        "is_meeting_request": True, "confidence": 0.9, "title": "Sync", "duration_minutes": 30,
        "proposed_slots": [{"start": day.replace(hour=hour).strftime("%Y-%m-%dT%H:%M:%S")}],
    }  # fmt: skip
    return FakeLLM({"classify_email": CLASSIFY_MEETING, "extract_meeting_request": extraction, "draft_reply": DRAFT_OK, **extra})


async def status_of(db, gmail_id="m1"):
    async with db.session() as s:
        rec = await EmailRepository(s).get_by_gmail_id(gmail_id)
        return EmailStatus(rec.status)


# ------------------------------------------------------------------------------ end-to-end flows
async def test_meeting_request_stages_draft_and_pending_approvals_then_executes_on_approval(db):
    app, gmail, calendar = app_with(db, llm=meeting_llm(next_weekday()))
    status = await app.agent.process_message("m1")

    assert status is EmailStatus.AWAITING_APPROVAL
    assert len(gmail.drafts) == 1 and gmail.sent == [] and calendar.events == []  # nothing irreversible yet
    pending = await app.hitl.list_pending()
    assert {p.action_type for p in pending} == {ActionType.SEND_EMAIL.value, ActionType.CREATE_CALENDAR_EVENT.value}

    for view in pending:
        result = await app.hitl.approve(view.id, "user:test")
        assert result.status is ApprovalStatus.EXECUTED
    assert gmail.sent == ["draft-1"] and len(calendar.events) == 1
    assert calendar.events[0]["attendees"] == ["alice@acme.com"] and calendar.events[0]["send_updates"] == "all"
    assert await status_of(db) is EmailStatus.COMPLETED


async def test_conflicting_meeting_offers_alternatives_and_creates_no_event(db):
    day = next_weekday()
    busy = [(day.replace(hour=9), day.replace(hour=12))]
    llm = meeting_llm(day)
    app, gmail, calendar = app_with(db, busy=busy, llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    pending = await app.hitl.list_pending()
    assert [p.action_type for p in pending] == [ActionType.SEND_EMAIL.value]
    drafting_prompt = next(c for c in llm.calls if c["tool"] == "draft_reply")["user"]
    assert "do NOT work" in drafting_prompt and "Offer these alternatives" in drafting_prompt


async def test_action_required_email_gets_draft_only(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert gmail.drafts["draft-1"]["to"] == "alice@acme.com" and gmail.drafts["draft-1"]["subject"] == "Re: Hello"
    assert gmail.sent == []


async def test_processing_is_idempotent(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, llm=llm)
    await app.agent.process_message("m1")
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert len(gmail.drafts) == 1 and len([c for c in llm.calls if c["tool"] == "classify_email"]) == 1


async def test_spam_is_ignored_and_info_only_needs_no_reply(db):
    spam = {**CLASSIFY_ACTION, "category": "spam_marketing", "requires_reply": False}
    app, gmail, _ = app_with(db, llm=FakeLLM({"classify_email": spam}))
    assert await app.agent.process_message("m1") is EmailStatus.IGNORED and not gmail.drafts
    info = {**CLASSIFY_ACTION, "category": "information_only", "requires_reply": False}
    app2, gmail2, _ = app_with(db, messages=[make_raw(msg_id="m2")], llm=FakeLLM({"classify_email": info}))
    assert await app2.agent.process_message("m2") is EmailStatus.COMPLETED and not gmail2.drafts


async def test_llm_outage_degrades_to_needs_review_without_side_effects(db):
    app, gmail, _ = app_with(db, llm=FakeLLM({"classify_email": LLMOutputError("down")}))
    assert await app.agent.process_message("m1") is EmailStatus.NEEDS_REVIEW
    assert not gmail.drafts


async def test_low_confidence_goes_to_review(db):
    low = {**CLASSIFY_ACTION, "confidence": 0.3}
    app, gmail, _ = app_with(db, llm=FakeLLM({"classify_email": low, "draft_reply": DRAFT_OK}))
    assert await app.agent.process_message("m1") is EmailStatus.NEEDS_REVIEW and not gmail.drafts


async def test_guardrail_blocked_draft_is_never_created(db):
    bad = {"body": "Hi [YOUR NAME], will do.", "confidence": 0.9}
    app, gmail, _ = app_with(db, llm=FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": bad}))
    assert await app.agent.process_message("m1") is EmailStatus.NEEDS_REVIEW and not gmail.drafts


async def test_self_sent_mail_is_ignored(db):
    msg = make_raw(sender="Dana <dana@example.com>")
    app, gmail, _ = app_with(db, messages=[msg], llm=FakeLLM())
    assert await app.agent.process_message("m1") is EmailStatus.IGNORED


async def test_run_once_summarises(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, _, _ = app_with(db, messages=[make_raw(msg_id="a"), make_raw(msg_id="b")], llm=llm)
    summary = await app.agent.run_once()
    assert summary.counts["awaiting_approval"] == 2


# ------------------------------------------------------------------------------ HITL controller
async def staged_send(db, **kw):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, calendar = app_with(db, llm=llm, **kw)
    await app.agent.process_message("m1")
    return app, gmail, calendar


async def test_double_approval_and_post_reject_approval_are_refused(db):
    app, gmail, _ = await staged_send(db)
    [pending] = await app.hitl.list_pending()
    await app.hitl.approve(pending.id, "u")
    with pytest.raises(ApprovalStateError):
        await app.hitl.approve(pending.id, "u")
    assert gmail.sent == ["draft-1"]  # sent exactly once


async def test_reject_prevents_execution(db):
    app, gmail, _ = await staged_send(db)
    [pending] = await app.hitl.list_pending()
    await app.hitl.reject(pending.id, "u", "too pushy")
    with pytest.raises(ApprovalStateError):
        await app.hitl.approve(pending.id, "u")
    assert gmail.sent == [] and (await app.hitl.get(pending.id)).status == ApprovalStatus.REJECTED.value


async def test_expired_approvals_cannot_be_executed(db):
    app, gmail, _ = await staged_send(db, approval_ttl_hours=-1)
    [pending] = await app.hitl.list_pending() or [None]
    assert pending is None  # already expired -> not listed as pending
    with pytest.raises(ApprovalStateError):
        await app.hitl.approve(2, "u")  # id 1 is the auto-created Gmail draft
    assert gmail.sent == []


async def test_failed_execution_is_recorded_and_retryable(db):
    app, gmail, _ = await staged_send(db)
    [pending] = await app.hitl.list_pending()
    gmail.drafts.clear()  # user deleted the draft in Gmail
    result = await app.hitl.approve(pending.id, "u")
    assert result.status is ApprovalStatus.FAILED and "GoogleAPIError" in result.error
    gmail.drafts["draft-1"] = {}  # draft restored
    assert (await app.hitl.approve(pending.id, "u")).status is ApprovalStatus.EXECUTED

async def test_rate_limited_email_is_deferred_then_retried(db):
    from email_assistant.exceptions import LLMRateLimitError

    llm = FakeLLM({"classify_email": LLMRateLimitError("busy")})
    app, gmail, _ = app_with(db, llm=llm, defer_retry_minutes=0, defer_give_up_hours=99)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED and not gmail.drafts
    llm.responses = {"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK}  # AI is back
    summary = await app.agent.run_once()
    assert summary.counts["awaiting_approval"] == 1 and len(gmail.drafts) == 1