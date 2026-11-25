"""Clash explanations, availability lists, explicit dates, authorized senders and real-life edge cases."""

from datetime import UTC, datetime, timedelta
from email.message import EmailMessage

from conftest import make_settings
from email_assistant.container import build_container
from email_assistant.integrations.gmail_client import RawMessage
from email_assistant.mcp_tools.tools import build_registry
from email_assistant.preprocessing.mime_parser import parse_raw_email
from email_assistant.scheduling.slots import Availability, parse_windows
from email_assistant.schemas.enums import EmailStatus
from email_assistant.schemas.scheduling import MeetingExtraction, SchedulingDecision
from email_assistant.services.drafting import deterministic_reply
from email_assistant.services.fetcher import EmailFetcher
from email_assistant.services.scheduling import SchedulingService
from email_assistant.ui.logic import reason_text
from fakes import CLASSIFY_ACTION, CLASSIFY_MEETING, DRAFT_OK, FakeCalendar, FakeGmail, FakeLLM, make_raw, next_weekday
from test_hitl_and_agent import app_with, meeting_llm
from test_services import parse

NOW = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)  # a Friday


def service(calendar=None, **settings_kw):
    settings = make_settings(**settings_kw)
    gmail = FakeGmail([make_raw()])
    registry = build_registry(gmail=gmail, calendar=calendar or FakeCalendar(), fetcher=EmailFetcher(gmail, settings), settings=settings)
    return SchedulingService(FakeLLM(), registry, settings)


def extraction(start, **kw):
    data = {"is_meeting_request": True, "confidence": 0.9, "duration_minutes": 30, "proposed_slots": [{"start": start}]}
    data.update(kw)
    return MeetingExtraction.model_validate(data)


async def plan(svc, ext, body="Can we meet?", now=NOW):
    return await svc.plan(parse(body=body), ext, own_email="dana@example.com", now=now)


# --------------------------------------------------------------------------- 1. clash with an existing meeting
async def test_clash_is_explained_to_the_sender_without_naming_the_other_meeting_and_to_you_with_its_name():
    busy = [(datetime(2026, 11, 24, 10, 0, tzinfo=UTC), datetime(2026, 11, 24, 10, 30, tzinfo=UTC))]
    standup = {"title": "Team standup", "start": "2026-11-24T10:00:00+00:00", "end": "2026-11-24T10:30:00+00:00", "all_day": False}
    svc = service(FakeCalendar(busy, [standup]))
    decision = await plan(svc, extraction("2026-11-24T10:00:00"))
    assert decision.kind == "counter" and decision.reasons == ["clash"]
    assert "already has a meeting scheduled at that time" in decision.facts and "Offer these alternatives" in decision.facts
    assert "Team standup" not in decision.facts  # never leaked to the sender
    assert any("Clash" in n and "Team standup" in n for n in decision.user_notes)
    assert decision.alternatives and all(not (a < busy[0][1] and busy[0][0] < b) for a, b in decision.alternatives)


async def test_clash_approval_card_shows_the_note_and_the_draft_is_told_about_the_clash(db):
    day = next_weekday()
    busy = [(day.replace(hour=10), day.replace(hour=10, minute=30))]
    standup = {"title": "Team standup", "start": busy[0][0].isoformat(), "end": busy[0][1].isoformat(), "all_day": False}
    llm = meeting_llm(day)
    app = build_container(make_settings(), gmail=FakeGmail([make_raw()]), calendar=FakeCalendar(busy, [standup]), llm=llm, db=db)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    [group] = await app.hitl.list_groups()
    preview = group.items[0].preview
    assert preview.startswith("NOTE: Clash:") and "Team standup" in preview
    assert "CLASHES with a meeting the user already has" in next(c for c in llm.calls if c["tool"] == "draft_reply")["user"]


async def test_back_to_back_meeting_is_reported_as_too_close_not_as_a_clash():
    busy = [(datetime(2026, 11, 24, 9, 30, tzinfo=UTC), datetime(2026, 11, 24, 10, 0, tzinfo=UTC))]
    decision = await plan(service(FakeCalendar(busy)), extraction("2026-11-24T10:00:00"))
    assert decision.reasons == ["buffer"] and "too close to another meeting" in decision.facts


# --------------------------------------------------------------------------- 2. outside availability -> list the free hours
async def test_request_outside_availability_lists_every_free_day_with_hours_from_settings():
    busy = [(datetime(2026, 11, 23, 9, 0, tzinfo=UTC), datetime(2026, 11, 23, 12, 0, tzinfo=UTC))]  # Monday morning taken
    decision = await plan(service(FakeCalendar(busy)), extraction("2026-11-22T14:00:00"), body="Can we meet this Sunday?")
    assert decision.kind == "counter" and "unavailable" in decision.reasons
    assert decision.availability_lines[0] == "Mon 23 Nov: 12:10-17:30"  # settings hours minus the busy block (+10 min buffer)
    assert decision.availability_lines[1] == "Tue 24 Nov: 09:00-17:30" and len(decision.availability_lines) == 5
    assert not any(line.startswith(("Sat", "Sun")) for line in decision.availability_lines)
    assert "List EVERY line below" in decision.facts and "Mon 23 Nov: 12:10-17:30" in decision.facts
    assert any(n.startswith("Outside your availability") for n in decision.user_notes)


async def test_weekend_hours_appear_in_the_list_when_the_user_opens_them_in_settings():
    schedule = Availability({0: parse_windows("09:00-17:30"), 6: parse_windows("14:00-16:00")}).to_json()
    decision = await plan(service(availability=schedule), extraction("2026-11-21T10:00:00"), body="Can we meet Saturday?")
    assert "Sun 22 Nov: 14:00-16:00" in decision.availability_lines


def test_template_reply_mentions_the_clash_or_lists_the_availability():
    first = parse(sender="Priya <p@client.example>")
    clash = SchedulingDecision(kind="counter", facts="", reasons=["clash"], alternatives=[(NOW, NOW + timedelta(minutes=30))])
    assert "already have another meeting scheduled at that time" in deterministic_reply(first, clash, make_settings())
    outside = SchedulingDecision(kind="counter", facts="", reasons=["unavailable"], availability_lines=["Mon 23 Nov: 09:00-17:30"])
    text = deterministic_reply(first, outside, make_settings())
    assert "outside the hours I am available" in text and "- Mon 23 Nov: 09:00-17:30" in text


# --------------------------------------------------------------------------- 3. explicit dates
async def test_a_specific_date_in_the_email_beats_the_models_guess():
    decision = await plan(service(), extraction("2026-11-23T14:00:00"), body="Can we meet on 25 Nov at 2 pm?")
    assert decision.kind == "accept" and decision.chosen_start == datetime(2026, 11, 25, 14, 0, tzinfo=UTC)
    assert any("taken from the text" in n for n in decision.user_notes)


async def test_missing_month_and_year_default_to_the_current_ones():
    decision = await plan(service(), extraction("2026-11-23T15:00:00"), body="How about the 26th at 3 pm?")
    assert decision.kind == "accept" and decision.chosen_start == datetime(2026, 11, 26, 15, 0, tzinfo=UTC)


async def test_a_day_that_already_passed_this_month_rolls_to_next_month_and_is_reported_as_too_far():
    decision = await plan(service(), extraction("2026-11-15T15:00:00"), body="Could we meet on the 15th at 3 pm?")
    assert decision.kind == "counter" and decision.reasons == ["too_far"]  # 15 Dec is beyond the 14-day look-ahead


async def test_day_first_date_format_is_respected_and_can_be_switched():
    day_first = await plan(service(), extraction("2026-11-23T10:00:00"), body="Let's meet on 03/12 at 10")
    month_first = await plan(service(day_first_dates=False), extraction("2026-11-23T10:00:00"), body="Let's meet on 12/03 at 10")
    assert day_first.kind == month_first.kind == "accept"
    assert day_first.chosen_start == month_first.chosen_start == datetime(2026, 12, 3, 10, 0, tzinfo=UTC)  # both = 3 December


async def test_weekday_that_contradicts_the_date_is_not_accepted_but_confirmed():
    ext = extraction("2026-11-27T14:00:00", proposed_slots=[{"start": "2026-11-27T14:00:00", "weekday": "Friday"}])
    decision = await plan(service(), ext, body="Friday 28 Nov at 2pm works")  # 28 Nov 2026 is a Saturday
    assert decision.kind == "no_meeting" and decision.reasons == ["ambiguous_date"] and "confirm which date" in decision.facts


async def test_time_ranges_are_not_mistaken_for_dates():
    decision = await plan(service(), extraction("2026-11-24T10:00:00"), body="I am free between 10-11 am on Tuesday")
    assert decision.kind == "accept" and decision.chosen_start.day == 24


# --------------------------------------------------------------------------- 4. authorized senders only
AUTH_OK = {"Authentication-Results": "mx.google.com; dkim=pass header.i=@vitstudent.ac.in; spf=pass smtp.mailfrom=s@vitstudent.ac.in; dmarc=pass header.from=vitstudent.ac.in"}


async def run_one(db, sender, headers=None, **kw):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, messages=[make_raw(sender=sender, headers=headers or {})], llm=llm, allowed_senders="vitstudent.ac.in", **kw)
    status = await app.agent.process_message("m1")
    return status, llm, gmail, (await app.inbox.recent())[0]


async def test_only_authorized_senders_reach_the_ai_and_approvals(db):
    status, llm, gmail, row = await run_one(db, "Stranger <x@gmail.com>", AUTH_OK)
    assert status is EmailStatus.IGNORED and llm.calls == [] and not gmail.drafts
    assert row.reason == "sender_not_authorized" and "never saw it" in reason_text(row.reason)


async def test_lookalike_domains_are_not_authorized(db):
    status, llm, *_ = await run_one(db, "Fake <x@notvitstudent.ac.in>", AUTH_OK)
    assert status is EmailStatus.IGNORED and llm.calls == []


async def test_authorized_but_unverified_sender_is_skipped_because_the_from_address_can_be_forged(db):
    status, llm, _, row = await run_one(db, "Student <s@vitstudent.ac.in>")  # no Authentication-Results header
    assert status is EmailStatus.IGNORED and llm.calls == [] and row.reason == "sender_not_verified"


async def test_verified_authorized_sender_is_processed(db):
    status, llm, gmail, _ = await run_one(db, "Student <s@vitstudent.ac.in>", AUTH_OK)
    assert status is EmailStatus.AWAITING_APPROVAL and len(gmail.drafts) == 1


async def test_verification_can_be_switched_off(db):
    status, *_ = await run_one(db, "Student <s@vitstudent.ac.in>", require_authenticated_sender=False)
    assert status is EmailStatus.AWAITING_APPROVAL


def test_authentication_results_are_parsed_by_the_mime_parser():
    verified = make_raw(sender="S <s@vitstudent.ac.in>", headers=AUTH_OK)
    assert parse_raw_email(verified.raw, gmail_id="m", thread_id="t").authenticated is True
    forged = make_raw(sender="S <s@vitstudent.ac.in>", headers={"Authentication-Results": "mx.google.com; dkim=pass header.d=evil.com; spf=fail smtp.mailfrom=evil.com; dmarc=fail"})
    assert parse_raw_email(forged.raw, gmail_id="m", thread_id="t").authenticated is False
    assert parse_raw_email(make_raw().raw, gmail_id="m", thread_id="t").authenticated is None


# --------------------------------------------------------------------------- 5. other real-life edge cases
def ics_message() -> RawMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "Priya <priya@client.example>", "dana@example.com", "Invitation: Sync @ Fri"
    msg["Date"] = "Mon, 03 Mar 2025 09:00:00 +0000"
    msg.set_content("You have been invited to Sync.")
    msg.add_alternative("BEGIN:VCALENDAR\nEND:VCALENDAR", subtype="calendar")
    return RawMessage(id="m1", thread_id="t1", raw=msg.as_bytes())


async def test_calendar_invitation_emails_are_not_answered_by_mail(db):
    llm = FakeLLM({"classify_email": CLASSIFY_MEETING, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, messages=[ics_message()], llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.COMPLETED
    assert llm.calls == [] and not gmail.drafts and (await app.inbox.recent())[0].reason == "calendar_invitation_answer_in_calendar"


async def test_email_without_readable_text_goes_to_review_instead_of_the_ai(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION})
    app, *_ = app_with(db, messages=[make_raw(body="   ")], llm=llm)
    assert await app.agent.process_message("m1") is EmailStatus.NEEDS_REVIEW and llm.calls == []


async def test_a_thread_you_already_answered_by_hand_gets_no_second_reply(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, llm=llm)
    gmail.replied_threads.add("t1")
    assert await app.agent.process_message("m1") is EmailStatus.COMPLETED
    assert not gmail.drafts and (await app.inbox.recent())[0].reason == "already_answered_by_you"


async def test_recurring_requests_are_held_and_reschedules_and_cancellations_warn_you():
    svc = service()
    recurring = await plan(svc, extraction("2026-11-24T10:00:00", is_recurring=True))
    assert recurring.kind == "no_meeting" and recurring.reasons == ["recurring"] and "RECURRING" in recurring.facts
    resched = await plan(svc, extraction("2026-11-24T10:00:00", request_type="reschedule"))
    assert resched.kind == "accept" and any("OLD event" in n for n in resched.user_notes)
    cancel = await plan(svc, extraction("2026-11-24T10:00:00", request_type="cancel"))
    assert cancel.kind == "cancel" and any("never deletes" in n for n in cancel.user_notes)


async def test_slot_too_soon_or_in_the_past_is_explained_not_silently_dropped():
    soon = await plan(service(), extraction("2026-11-20T11:00:00"), body="Quick call at 11?")  # 1 hour away, notice is 2 h
    assert soon.reasons == ["too_soon"] and "less than 2 hours away" in soon.facts
    past = await plan(service(), extraction("2026-11-18T11:00:00"), body="Call at 11?")
    assert past.reasons == ["past"]


def test_finish_reasons_are_translated_for_history():
    assert reason_text("no_response:automated_sender_never_answered").startswith("No reply needed - Automated")
    assert reason_text("meeting_request:schedule") == "Meeting request - reply + calendar"
    assert "already replied" in reason_text("already_answered_by_you")