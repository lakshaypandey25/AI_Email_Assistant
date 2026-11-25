"""Grouped approvals, approval-time conflict re-check, customizations, ignore/VIP lists, inbox queries."""

import json

from conftest import make_settings
from email_assistant.config import Settings, load_preferences, save_preferences, set_env_value
from email_assistant.core.senders import parse_list, sender_matches
from email_assistant.schemas.enums import EmailStatus
from email_assistant.ui.logic import (
    csv_to_lines, filter_chat_models, lines_to_csv, signature_for_editing, signature_for_storage,
    validate_client_secret, validate_prefs,
)
from fakes import CLASSIFY_ACTION, DRAFT_OK, FakeLLM, make_raw, next_weekday
from test_hitl_and_agent import app_with, meeting_llm


# ----------------------------------------------------------------------------- pure helpers
def test_sender_matching_addresses_and_domains():
    patterns = parse_list("Boss@Corp.com, @acme.com\nnews.example.org")
    assert sender_matches(patterns, "boss@corp.com")
    assert sender_matches(patterns, "anyone@acme.com") and sender_matches(patterns, "x@mail.acme.com")
    assert sender_matches(patterns, "a@news.example.org")
    assert not sender_matches(patterns, "other@corp.com") and not sender_matches(patterns, "a@notacme.com")


def test_preferences_roundtrip_is_whitelisted_and_applied(tmp_path):
    path = tmp_path / "prefs.json"
    save_preferences({"user_name": "Zed", "default_tone": "friendly", "groq_api_key": "leak", "bogus": 1}, path)
    saved = load_preferences(path)
    assert saved == {"user_name": "Zed", "default_tone": "friendly"}  # secrets / unknown keys never persisted
    settings = Settings(_env_file=None, **saved)
    assert settings.user_name == "Zed" and settings.default_tone == "friendly"


def test_set_env_value_updates_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nGROQ_API_KEY=old\n# comment\n")
    set_env_value("GROQ_API_KEY", "gsk_new", env)
    set_env_value("NEW_KEY", "with space", env)
    text = env.read_text()
    assert "GROQ_API_KEY=gsk_new" in text and "old" not in text and 'NEW_KEY="with space"' in text and "A=1" in text


def test_validate_prefs_and_client_secret_and_models():
    assert validate_prefs({"user_name": "Ok"}) is None
    assert "user_timezone" in validate_prefs({"user_timezone": "Mars/Base"})
    assert "work_days" in validate_prefs({"work_days": "someday"})
    assert validate_client_secret(json.dumps({"installed": {"project_id": "p"}}).encode())[0]
    ok, msg = validate_client_secret(json.dumps({"web": {}}).encode())
    assert not ok and "Desktop" in msg
    assert not validate_client_secret(b"not json")[0]
    assert filter_chat_models(["whisper-large-v3", "openai/gpt-oss-20b", "meta-llama/llama-prompt-guard-2-22m"]) == ["openai/gpt-oss-20b"]


def test_list_and_signature_conversions():
    assert lines_to_csv("a@x.com\n\n @b.com ") == "a@x.com,@b.com" and csv_to_lines("a@x.com,@b.com") == "a@x.com\n@b.com"
    assert signature_for_storage(signature_for_editing("Best,\\n{user_name}")) == "Best,\\n{user_name}"


# ----------------------------------------------------------------------------- grouped approvals
async def staged_meeting(db, **kw):
    day = next_weekday()
    app, gmail, calendar = app_with(db, llm=meeting_llm(day), **kw)
    assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
    return app, gmail, calendar, day


async def test_groups_show_context_and_warn_about_paired_invite(db):
    app, *_ = await staged_meeting(db)
    [group] = await app.hitl.list_groups()
    assert group.sender == "alice@acme.com" and group.category == "meeting_request"
    assert [i.action_type for i in group.items] == ["create_calendar_event", "send_email"]  # invite listed first
    assert any("invite" in w for w in group.warnings)


async def test_approve_email_runs_invite_then_reply(db):
    app, gmail, calendar, _ = await staged_meeting(db)
    result = await app.hitl.approve_email(1, "u")
    assert result.all_executed and len(calendar.events) == 1 and gmail.sent == ["draft-1"]
    assert await app.hitl.list_groups() == []


async def test_conflict_at_approval_blocks_invite_and_withholds_reply(db):
    app, gmail, calendar, day = await staged_meeting(db)
    calendar.busy.append((day.replace(hour=10), day.replace(hour=10, minute=30)))  # slot taken after drafting
    result = await app.hitl.approve_email(1, "u")
    assert result.results[0].is_conflict and len(result.skipped) == 1
    assert calendar.events == [] and gmail.sent == []  # no invite AND no reply promising one
    [group] = await app.hitl.list_groups()
    assert any("calendar changed" in w for w in group.warnings)

    forced = await app.hitl.approve_email(1, "u", ignore_conflicts=True)
    assert forced.all_executed and len(calendar.events) == 1 and gmail.sent == ["draft-1"]


async def test_reject_email_rejects_every_open_action(db):
    app, gmail, calendar, _ = await staged_meeting(db)
    assert await app.hitl.reject_email(1, "u", "not now") == 2
    assert await app.hitl.list_groups() == [] and gmail.sent == [] and calendar.events == []


# ----------------------------------------------------------------------------- customizations in the pipeline
async def test_ignore_list_skips_processing_without_llm(db):
    llm = FakeLLM()
    app, gmail, _ = app_with(db, llm=llm, ignore_senders="@acme.com")
    assert await app.agent.process_message("m1") is EmailStatus.IGNORED
    assert llm.calls == [] and not gmail.drafts


async def test_vip_is_never_spam_and_gets_priority_boost(db):
    spam = {**CLASSIFY_ACTION, "category": "spam_marketing", "requires_reply": False, "priority_score": 1}
    app, _, _ = app_with(db, llm=FakeLLM({"classify_email": spam}), vip_senders="alice@acme.com")
    assert await app.agent.process_message("m1") is EmailStatus.COMPLETED
    [row] = await app.inbox.recent()
    assert row.category == "information_only" and row.priority >= 4


async def test_custom_instructions_and_tone_reach_the_drafting_prompt(db):
    llm = FakeLLM({"classify_email": CLASSIFY_ACTION, "draft_reply": DRAFT_OK})
    app, _, _ = app_with(db, llm=llm, custom_instructions="Never discuss pricing by email.", default_tone="friendly")
    await app.agent.process_message("m1")
    system = next(c for c in llm.calls if c["tool"] == "draft_reply")["system"]
    assert "Never discuss pricing by email." in system and "friendly" in system


# ----------------------------------------------------------------------------- inbox + memory services
async def test_inbox_details_search_and_reprocess(db):
    llm = FakeLLM({"classify_email": {**CLASSIFY_ACTION, "category": "spam_marketing", "requires_reply": False}})
    app, _, _ = app_with(db, messages=[make_raw(subject="Quarterly budget")], llm=llm)
    await app.agent.process_message("m1")
    assert [r.id for r in await app.inbox.recent(search="budget")] == [1]
    assert await app.inbox.recent(search="nothing-matches") == []
    details = await app.inbox.details(1)
    assert details["reasoning"] and any(a["event"] == "classified" for a in details["audit"])
    assert (await app.inbox.status_counts()) == {"ignored": 1}

    assert await app.inbox.reprocess(1) is True
    assert (await app.inbox.recent())[0].status == "failed"
    await app.agent.process_message("m1")  # failed + attempts 0 -> processed again
    assert len([c for c in llm.calls if c["tool"] == "classify_email"]) == 2


async def test_memory_add_list_delete(db):
    app, *_ = app_with(db)
    await app.memory.add("global", "preference", "Be brief")
    [entry] = await app.memory.list_entries()
    assert entry["content"] == "Be brief"
    assert await app.memory.delete(entry["id"]) is True and await app.memory.list_entries() == []
