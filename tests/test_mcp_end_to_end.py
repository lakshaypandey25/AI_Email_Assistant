"""The agent talks to Gmail and Calendar ONLY through MCP (in-memory transport: the same JSON-RPC as stdio)."""

from contextlib import asynccontextmanager

import pytest

from conftest import make_settings
from email_assistant.container import build_container
from email_assistant.core.approval import ApprovalSigner
from email_assistant.exceptions import ApprovalRequiredError, GoogleAPIError, ToolInputError, ToolTransportError
from email_assistant.mcp_tools.client import McpToolClient
from email_assistant.mcp_tools.server import create_mcp_server
from email_assistant.mcp_tools.tools import build_registry
from email_assistant.schemas.enums import EmailStatus
from email_assistant.services.fetcher import EmailFetcher
from fakes import FakeCalendar, FakeGmail, make_raw, next_weekday
from test_hitl_and_agent import meeting_llm


def stack(expose_high=True):
    settings = make_settings()
    gmail, calendar = FakeGmail([make_raw()]), FakeCalendar()
    registry = build_registry(gmail=gmail, calendar=calendar, fetcher=EmailFetcher(gmail, settings), settings=settings)
    signer = ApprovalSigner(b"s" * 32)
    server = create_mcp_server(registry, signer=signer, expose_high_risk=expose_high)
    return settings, gmail, calendar, server, signer


def test_approval_tokens_are_bound_to_tool_arguments_expiry_and_single_use():
    now = {"t": 1000.0}
    signer = ApprovalSigner(b"k" * 32, ttl_seconds=60, clock=lambda: now["t"])
    args = {"draft_id": "d1"}
    token = signer.issue("gmail_send_draft", args)
    assert signer.verify(token, "gmail_send_draft", args) is True
    assert signer.verify(token, "gmail_send_draft", args) is False  # replay
    assert signer.verify(signer.issue("gmail_send_draft", args), "calendar_create_event", args) is False  # other tool
    assert signer.verify(signer.issue("gmail_send_draft", args), "gmail_send_draft", {"draft_id": "d2"}) is False  # other args
    assert signer.verify(ApprovalSigner(b"x" * 32, clock=lambda: now["t"]).issue("gmail_send_draft", args), "gmail_send_draft", args) is False
    late = signer.issue("gmail_send_draft", args)
    now["t"] += 61
    assert signer.verify(late, "gmail_send_draft", args) is False  # expired


async def test_read_tools_round_trip_through_the_mcp_protocol():
    _, _, _, server, signer = stack()
    client = McpToolClient.in_memory(server, signer)
    try:
        assert (await client.call("gmail_get_message", {"message_id": "m1"}))["sender_email"] == "alice@acme.com"
        assert (await client.call("gmail_get_profile", {}))["email"] == "dana@example.com"
        assert "free" in await client.call("calendar_free_windows", {"days": 3})
    finally:
        await client.aclose()


async def test_high_risk_tools_need_approval_and_then_run_through_a_signed_token():
    _, gmail, _, server, signer = stack()
    gmail.drafts["d1"] = {}
    client = McpToolClient.in_memory(server, signer)
    try:
        with pytest.raises(ApprovalRequiredError):
            await client.call("gmail_send_draft", {"draft_id": "d1"})
        assert gmail.sent == []
        assert (await client.call("gmail_send_draft", {"draft_id": "d1"}, approved=True))["message_id"] == "sent-1"
        assert gmail.sent == ["d1"]
        with pytest.raises(ToolInputError):
            await client.call("gmail_get_message", {"unexpected": 1})
    finally:
        await client.aclose()


async def test_server_rejects_unsigned_forged_tampered_and_replayed_calls():
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    _, gmail, _, server, signer = stack()
    gmail.drafts["d1"] = {}
    async with connect(server) as session:
        assert (await session.call_tool("gmail_send_draft", {"draft_id": "d1"})).isError  # no token
        forged = ApprovalSigner(b"x" * 32).issue("gmail_send_draft", {"draft_id": "d1"})
        assert (await session.call_tool("gmail_send_draft", {"draft_id": "d1", "approval_token": forged})).isError
        other_args = signer.issue("gmail_send_draft", {"draft_id": "d2"})
        assert (await session.call_tool("gmail_send_draft", {"draft_id": "d1", "approval_token": other_args})).isError
        good = signer.issue("gmail_send_draft", {"draft_id": "d1"})
        assert not (await session.call_tool("gmail_send_draft", {"draft_id": "d1", "approval_token": good})).isError
        assert (await session.call_tool("gmail_send_draft", {"draft_id": "d1", "approval_token": good})).isError  # replay
    assert gmail.sent == ["d1"]  # exactly once


async def test_standalone_server_exposes_only_read_and_draft_tools():
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    _, _, _, server, _ = stack(expose_high=False)
    async with connect(server) as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert "gmail_send_draft" not in names and "calendar_create_event" not in names
    assert {"gmail_search_messages", "calendar_free_windows", "gmail_create_reply_draft_for_message"} <= names


async def test_google_errors_keep_their_type_and_retry_flags_across_the_protocol():
    _, _, _, server, signer = stack()
    client = McpToolClient.in_memory(server, signer)
    try:
        with pytest.raises(GoogleAPIError) as caught:
            await client.call("gmail_send_draft", {"draft_id": "missing"}, approved=True)
        assert caught.value.status == 404 and caught.value.retryable is False
    finally:
        await client.aclose()


async def test_a_tool_server_that_cannot_start_is_a_retryable_transport_error():
    @asynccontextmanager
    async def broken():
        raise RuntimeError("cannot spawn")
        yield  # pragma: no cover

    client = McpToolClient(lambda: broken(), ApprovalSigner(b"s" * 32))
    with pytest.raises(ToolTransportError) as caught:
        await client.call("gmail_get_profile", {})
    assert caught.value.retryable


async def test_the_whole_agent_pipeline_runs_through_mcp(db):
    settings, gmail, calendar, server, signer = stack()
    client = McpToolClient.in_memory(server, signer)
    app = build_container(settings, tools=client, llm=meeting_llm(next_weekday()), db=db)
    try:
        assert await app.agent.process_message("m1") is EmailStatus.AWAITING_APPROVAL
        assert len(gmail.drafts) == 1 and gmail.sent == [] and calendar.events == []  # nothing irreversible yet
        [group] = await app.hitl.list_groups()
        result = await app.hitl.approve_email(group.email_id, "user:test")
        assert result.all_executed
        assert gmail.sent == ["draft-1"] and len(calendar.events) == 1  # sent and created over MCP, with signed tokens
    finally:
        await client.aclose()