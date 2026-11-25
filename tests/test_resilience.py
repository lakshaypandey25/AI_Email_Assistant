"""Fallbacks for every external API, response modes, knowledge grounding."""

import pytest

from conftest import make_settings
from email_assistant.container import build_container
from email_assistant.exceptions import GoogleAPIError, LLMOutputError, LLMRateLimitError
from email_assistant.schemas.enums import EmailStatus
from fakes import CLASSIFY_ACTION, CLASSIFY_MEETING, DRAFT_OK, FakeCalendar, FakeGmail, FakeLLM, make_raw, next_weekday
from test_hitl_and_agent import app_with, meeting_llm, status_of

FAST_RETRY = {"defer_retry_minutes": 0}


async def audit_events(app, email_id=1):
    return {a["event"] for a in (await app.inbox.details(email_id))["audit"]}


# --------------------------------------------------------------------------- Groq: defer -> retry -> fallback
async def test_rate_limited_email_is_deferred_then_retried_automatically(db):
    llm = FakeLLM({"classify_email": LLMRateLimitError("busy")})
    app, gmail, _ = app_with(db, llm=llm, defer_give_up_hours=99, **FAST_RETRY)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED
    assert not gmail.drafts and await status_of(db) is EmailStatus.DEFERRED
    assert "email_deferred" in await audit_events(app)

    llm.responses = {"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK}  # AI is back
    summary = await app.agent.run_once()
    assert summary.counts["awaiting_approval"] == 1 and len(gmail.drafts) == 1


async def test_deferred_email_is_retried_even_when_gmail_search_is_down(db):
    class SearchDown(FakeGmail):
        async def list_message_ids(self, query, max_results=20):
            raise GoogleAPIError("down", status=503, retryable=True)

    llm = FakeLLM({"classify_email": LLMRateLimitError("busy")})
    gmail = SearchDown([make_raw()])
    app = build_container(make_settings(defer_give_up_hours=99, **FAST_RETRY), gmail=gmail, calendar=FakeCalendar(), llm=llm, db=db)
    await app.agent.process_message("m1")
    llm.responses = {"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK}
    summary = await app.agent.run_once()  # search fails, deferred id still processed
    assert summary.counts["awaiting_approval"] == 1

    empty = build_container(make_settings(), gmail=SearchDown([]), calendar=FakeCalendar(), llm=FakeLLM(), db=db)
    with pytest.raises(GoogleAPIError):  # nothing to fall back on -> surface the error
        await empty.agent.run_once()


async def test_after_the_retry_window_keyword_rules_take_over_and_go_to_review(db):
    llm = FakeLLM({"classify_email": LLMRateLimitError("busy")})
    app, gmail, _ = app_with(db, llm=llm, defer_give_up_hours=0, **FAST_RETRY)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED
    assert await app.agent.process_message("m1") is EmailStatus.NEEDS_REVIEW
    assert "rules_fallback_used" in await audit_events(app) and not gmail.drafts


async def test_drafting_outage_defers_then_template_reply_after_the_window(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": LLMRateLimitError("busy")})
    app, gmail, _ = app_with(db, llm=llm, defer_give_up_hours=0, **FAST_RETRY)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED and not gmail.drafts
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    [group] = await app.hitl.list_groups()
    assert "template_reply" in group.items[0].preview  # flagged: a human must review the generic reply
    assert "get back to you shortly" in next(iter(gmail.drafts.values()))["body"]


# --------------------------------------------------------------------------- Calendar / extraction fallbacks
async def test_calendar_outage_produces_holding_reply_and_no_invite(db):
    class CalendarDown(FakeCalendar):
        async def query_freebusy(self, *args, **kwargs):
            raise GoogleAPIError("down", status=503, retryable=True)

    llm = meeting_llm(next_weekday())
    app = build_container(make_settings(), gmail=FakeGmail([make_raw()]), calendar=CalendarDown(), llm=llm, db=db)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert [i.action_type for g in await app.hitl.list_groups() for i in g.items] == ["send_email"]
    assert "could not be checked" in next(c for c in llm.calls if c["tool"] == "draft_reply")["user"]
    assert "calendar_unavailable_fallback" in await audit_events(app)


async def test_unreadable_meeting_times_get_a_holding_reply_not_a_guess(db):
    llm = FakeLLM({"classify_email": CLASSIFY_MEETING, "extract_meeting_request": LLMOutputError("bad"), "draft_reply": DRAFT_OK})
    app, gmail, calendar = app_with(db, llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert "could not be read reliably" in next(c for c in llm.calls if c["tool"] == "draft_reply")["user"]
    assert [i.action_type for g in await app.hitl.list_groups() for i in g.items] == ["send_email"] and calendar.events == []


# --------------------------------------------------------------------------- Gmail fallbacks
async def test_transient_gmail_read_error_defers_but_permanent_error_fails(db):
    class Flaky(FakeGmail):
        error = GoogleAPIError("busy", status=503, retryable=True)

        async def get_raw_message(self, message_id):
            raise self.error

    gmail = Flaky([make_raw()])
    app = build_container(make_settings(**FAST_RETRY), gmail=gmail, calendar=FakeCalendar(), llm=FakeLLM(), db=db)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED
    Flaky.error = GoogleAPIError("gone", status=404, retryable=False)
    assert await app.agent.process_message("m1") is EmailStatus.FAILED


async def test_gmail_draft_outage_defers_the_whole_email_for_a_later_retry(db):
    class DraftsDown(FakeGmail):
        async def create_draft(self, **kwargs):
            raise GoogleAPIError("down", status=503, retryable=True)

    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app = build_container(make_settings(), gmail=DraftsDown([make_raw()]), calendar=FakeCalendar(), llm=llm, db=db)
    assert await app.agent.process_message("m1") is EmailStatus.DEFERRED


# --------------------------------------------------------------------------- not every email gets a reply
SILENT = {**CLASSIFY_ACTION, "requires_reply": False, "response_mode": "none"}


@pytest.mark.parametrize(
    "classification,kwargs,sender,expected",
    [
        (SILENT, {}, None, EmailStatus.COMPLETED),  # AI: no response needed
        ({**CLASSIFY_ACTION, "response_mode": "human_only"}, {}, None, EmailStatus.NEEDS_REVIEW),  # needs the user's judgement
        (CLASSIFY_ACTION, {"reply_categories": "meeting_request"}, None, EmailStatus.COMPLETED),  # policy switched it off
        (CLASSIFY_ACTION, {}, "Bot <noreply@shop.example>", EmailStatus.COMPLETED),  # never answer automated senders
    ],
)
async def test_emails_that_should_not_be_answered_create_no_draft(db, classification, kwargs, sender, expected):
    message = make_raw(sender=sender) if sender else make_raw()
    llm = FakeLLM({"classify_email": classification, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, messages=[message], llm=llm, **kwargs)
    assert await app.agent.process_message("m1") is expected
    assert not gmail.drafts and all(c["tool"] != "draft_reply" for c in llm.calls)


async def test_reply_only_mode_never_touches_the_calendar(db):
    reply_only = {**CLASSIFY_MEETING, "response_mode": "reply"}
    llm = FakeLLM({"classify_email": reply_only, "draft_reply": DRAFT_OK})
    app, gmail, calendar = app_with(db, llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert all(c["tool"] != "extract_meeting_request" for c in llm.calls) and len(gmail.drafts) == 1
    assert [i.action_type for g in await app.hitl.list_groups() for i in g.items] == ["send_email"]


async def test_scheduling_can_be_switched_off_so_meetings_get_a_plain_reply(db):
    llm = FakeLLM({"classify_email": CLASSIFY_MEETING, "draft_reply": DRAFT_OK})
    app, gmail, calendar = app_with(db, llm=llm, enable_scheduling=False)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert all(c["tool"] != "extract_meeting_request" for c in llm.calls) and calendar.events == []


# --------------------------------------------------------------------------- knowledge base grounding
async def test_relevant_knowledge_reaches_the_drafting_prompt_and_irrelevant_does_not(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, _, _ = app_with(db, messages=[make_raw(subject="Refund question", body="What is your refund policy for annual plans?")], llm=llm)
    await app.knowledge.add("Refund policy", "Refunds are available within 30 days of purchase with proof of payment.")
    await app.knowledge.add("Office hours", "Our office is open Monday to Friday from 9 to 6.")
    await app.agent.process_message("m1")
    prompt = next(c for c in llm.calls if c["tool"] == "draft_reply")["user"]
    assert "KNOWLEDGE BASE" in prompt and "30 days" in prompt and "Office hours" not in prompt