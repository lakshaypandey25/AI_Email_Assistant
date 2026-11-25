"""Draft vs automatic sending, availability editing, Gmail sync, and History accuracy."""

from datetime import UTC, datetime

from conftest import make_settings
from email_assistant.container import build_container
from email_assistant.schemas.enums import EmailStatus
from email_assistant.schemas.scheduling import MeetingExtraction
from email_assistant.scheduling.slots import Availability, parse_windows
from fakes import CLASSIFY_ACTION, DRAFT_OK, FakeCalendar, FakeGmail, FakeLLM, make_raw, next_weekday
from test_hitl_and_agent import app_with, meeting_llm, staged_send
from test_services import parse, scheduler


# --------------------------------------------------------------------------- availability (the 'Sunday 2pm' case)
async def test_sunday_is_accepted_only_when_the_user_marks_it_available():
    now = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)
    ext = MeetingExtraction.model_validate({
        "is_meeting_request": True, "confidence": 0.9, "duration_minutes": 30,
        "proposed_slots": [{"start": "2026-11-08T14:00:00", "weekday": "Sunday"}],
    })
    email = parse(body="Can we meet on Sunday at 2 pm?")
    default_svc, _ = scheduler()  # Mon-Fri 09:00-17:30
    refused = await default_svc.plan(email, ext, own_email="dana@example.com", now=now)
    assert refused.kind == "counter" and "availability" in refused.facts

    sunday = Availability({6: parse_windows("14:00-16:00")}).to_json()
    svc, _ = scheduler(availability=sunday)
    accepted = await svc.plan(email, ext, own_email="dana@example.com", now=now)
    assert accepted.kind == "accept" and accepted.chosen_start == datetime(2026, 11, 8, 14, 0, tzinfo=UTC)


async def test_time_off_blocks_an_otherwise_available_day():
    now = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)
    ext = MeetingExtraction.model_validate({"is_meeting_request": True, "confidence": 0.9, "proposed_slots": [{"start": "2026-11-05T10:00:00"}]})
    off = Availability.from_json('{"windows": {"thu": ["09:00-17:00"]}, "time_off": ["2026-11-05"]}').to_json()
    svc, _ = scheduler(availability=off)
    decision = await svc.plan(parse(), ext, own_email="dana@example.com", now=now)
    assert decision.kind == "counter" and "time off" in decision.facts


# --------------------------------------------------------------------------- draft vs automatic
async def test_draft_mode_saves_a_draft_and_sends_nothing(db):
    app, gmail, _ = await staged_send(db)
    assert gmail.sent == [] and len(gmail.drafts) == 1
    [row] = await app.inbox.recent()
    assert row.status == "awaiting_approval" and row.reply == "Draft saved - waiting for you"


async def test_automatic_mode_sends_plain_replies_immediately_and_history_says_so(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, gmail, _ = app_with(db, llm=llm, send_mode="auto")
    assert await app.agent.process_message("m1") is EmailStatus.COMPLETED
    assert gmail.sent == ["draft-1"]
    [row] = await app.inbox.recent()
    assert row.status == "completed" and row.reply == "Sent (automatically)" and row.updated_at is not None


async def test_automatic_mode_still_holds_risky_replies_unless_switched_off(db):
    risky = {**DRAFT_OK, "makes_commitments": True}
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": risky})
    app, gmail, _ = app_with(db, llm=llm, send_mode="auto")
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL and gmail.sent == []
    app2, gmail2, _ = app_with(db, messages=[make_raw(msg_id="m2")], llm=llm, send_mode="auto", hold_risky_replies=False)
    assert await app2.agent.process_message("m2") is EmailStatus.COMPLETED and gmail2.sent == ["draft-1"]


async def test_reply_promising_an_invite_waits_when_events_are_not_automatic(db):
    app, gmail, calendar = app_with(db, llm=meeting_llm(next_weekday()), send_mode="auto", auto_create_events=False)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert gmail.sent == [] and calendar.events == []
    assert len(await app.hitl.list_groups()) == 1


async def test_everything_automatic_creates_the_invite_then_sends_the_reply(db):
    app, gmail, calendar = app_with(db, llm=meeting_llm(next_weekday()), send_mode="auto", auto_create_events=True, hold_risky_replies=False)
    assert await app.agent.process_message("m1") is EmailStatus.COMPLETED
    assert len(calendar.events) == 1 and gmail.sent == ["draft-1"]
    [row] = await app.inbox.recent()
    assert row.reply == "Sent (automatically)" and row.invite == "Created (automatically)"


async def test_invite_that_hits_a_conflict_holds_back_the_automatic_reply(db):
    day = next_weekday()

    class Racy(FakeCalendar):
        calls = 0

        async def query_freebusy(self, tmin, tmax, ids=None):
            self.calls += 1  # free when the assistant plans, taken by the time the invite is created
            return [] if self.calls == 1 else [(day.replace(hour=10), day.replace(hour=10, minute=30))]

    calendar = Racy()
    app = build_container(
        make_settings(send_mode="auto", auto_create_events=True, hold_risky_replies=False),
        gmail=FakeGmail([make_raw()]), calendar=calendar, llm=meeting_llm(day), db=db,
    )
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    assert calendar.events == [] and (await app.inbox.recent())[0].reply == "Draft saved - waiting for you"
    [group] = await app.hitl.list_groups()
    assert any(i.error and i.error.startswith("CONFLICT") for i in group.items)


# --------------------------------------------------------------------------- Gmail sync + History accuracy
async def test_reply_sent_by_hand_in_gmail_closes_the_approval(db):
    app, gmail, _ = await staged_send(db)
    gmail.sent_externally.add("draft-1")
    del gmail.drafts["draft-1"]
    assert await app.hitl.reconcile_drafts() == {"sent": 1, "deleted": 0}
    [row] = await app.inbox.recent()
    assert row.status == "completed" and row.reply == "Sent (from Gmail)"
    assert await app.hitl.list_groups() == []


async def test_draft_deleted_in_gmail_is_recorded_and_untouched_drafts_stay_pending(db):
    app, gmail, _ = await staged_send(db)
    assert await app.hitl.reconcile_drafts() == {"sent": 0, "deleted": 0}  # draft still there
    del gmail.drafts["draft-1"]
    assert await app.hitl.reconcile_drafts() == {"sent": 0, "deleted": 1}
    [row] = await app.inbox.recent()
    assert row.reply == "Draft deleted in Gmail" and row.status == "completed"


async def test_history_is_updated_immediately_when_you_approve(db):
    app, gmail, _ = await staged_send(db)
    before = (await app.inbox.recent())[0]
    await app.hitl.approve_email(1, "u")
    after = (await app.inbox.recent())[0]
    assert (before.status, before.reply) == ("awaiting_approval", "Draft saved - waiting for you")
    assert (after.status, after.reply) == ("completed", "Sent (by you)") and after.updated_at >= before.updated_at
    events = [a["event"] for a in (await app.inbox.details(1))["audit"]]
    last_executed = max(i for i, e in enumerate(events) if e == "action_executed")
    assert events.index("draft_saved") < events.index("approval_requested") < events.index("approval_granted") < last_executed