import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import groq
import httpx
import pytest

from conftest import make_settings
from email_assistant.core.rate_limit import AsyncRateLimiter
from email_assistant.core.retry import RetryPolicy
from email_assistant.exceptions import LLMOutputError, LLMRateLimitError
from email_assistant.llm.groq_client import GroqLLMClient
from email_assistant.llm.tooling import pydantic_to_tool
from email_assistant.schemas.llm import ClassificationResult
from email_assistant.schemas.scheduling import MeetingExtraction

VALID = {
    "category": "action_required", "confidence": 0.8, "priority_score": 4,
    "reasoning": "Asks for review.", "summary": "Review doc.", "requires_reply": True,
}  # fmt: skip


def tool_response(args, name="classify_email"):
    payload = args if isinstance(args, str) else json.dumps(args)
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=payload))
    msg = SimpleNamespace(tool_calls=[call], content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(total_tokens=42))


def content_response(text):
    msg = SimpleNamespace(tool_calls=None, content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def make_client(outputs, sleeps=None, **settings_kw):
    create = AsyncMock(side_effect=outputs)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def fake_sleep(d):
        if sleeps is not None:
            sleeps.append(d)

    client = GroqLLMClient(
        make_settings(**{"groq_model_fast": "model-a", "groq_model_quality": "model-a", "groq_tokens_per_minute": 0, **settings_kw}),
        client=fake,
        rate_limiter=AsyncRateLimiter(10_000, burst=10_000),
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False, sleep=fake_sleep),
    )
    return client, create


async def call(client, tier="fast"):
    return await client.structured_call(
        system="s", user="u", schema=ClassificationResult, tool_name="classify_email", tool_description="d", tier=tier
    )


async def test_valid_tool_call_is_parsed_and_tool_choice_forced():
    client, create = make_client([tool_response(VALID)])
    result = await call(client)
    assert result.category.value == "action_required"
    kwargs = create.call_args.kwargs
    assert kwargs["tool_choice"]["function"]["name"] == "classify_email"
    assert kwargs["model"] == "model-a"


# async def test_quality_tier_uses_quality_model():
#     client, create = make_client([tool_response(VALID)])
#     await call(client, tier="quality")
#     assert create.call_args.kwargs["model"] == make_settings().groq_model_quality


async def test_tolerant_validators_normalise_model_quirks():
    quirky = {**VALID, "category": "Action Required", "confidence": "85%", "priority_score": "4"}
    client, _ = make_client([tool_response(quirky)])
    result = await call(client)
    assert result.confidence == pytest.approx(0.85) and result.priority_score == 4


async def test_invalid_output_triggers_repair_then_succeeds():
    client, create = make_client([tool_response({"category": "nonsense"}), tool_response(VALID)])
    result = await call(client)
    assert result.requires_reply is True
    assert create.call_count == 2
    assert "invalid" in create.call_args.kwargs["messages"][-1]["content"]


async def test_gives_up_with_llm_output_error():
    client, create = make_client([tool_response("{not json")] * 3)
    with pytest.raises(LLMOutputError):
        await call(client)
    assert create.call_count == 3


async def test_json_in_content_fallback():
    client, _ = make_client([content_response("```json\n" + json.dumps(VALID) + "\n```")])
    assert (await call(client)).priority_score == 4


def _rate_limit_error(retry_after="1"):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"), headers={"retry-after": retry_after})
    return groq.RateLimitError("rate limited", response=response, body=None)


async def test_rate_limit_is_retried_with_retry_after():
    sleeps = []
    client, create = make_client([_rate_limit_error("2"), tool_response(VALID)], sleeps)
    await call(client)
    assert create.call_count == 2 and sleeps and sleeps[0] >= 2


async def test_persistent_rate_limit_raises_domain_error():
    client, _ = make_client([_rate_limit_error()] * 3)
    with pytest.raises(LLMRateLimitError):
        await call(client)


def test_tool_schema_is_flat_and_inlined():
    tool = pydantic_to_tool(ClassificationResult, "classify_email", "desc")
    text = json.dumps(tool)
    assert "$ref" not in text and "$defs" not in text
    params = tool["function"]["parameters"]
    assert "category" in params["properties"] and "action_items" in params["properties"]
    assert params["properties"]["action_items"]["items"]["properties"]["description"]


def test_optional_fields_collapse_anyof_null():
    tool = pydantic_to_tool(MeetingExtraction, "extract_meeting_request", "d")
    props = tool["function"]["parameters"]["properties"]
    assert "title" in props  # a field named "title" must survive schema cleanup
    assert props["duration_minutes"]["type"] == ["integer", "null"]  # nullable, not required to be set

def _error(cls, status, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"), headers=headers)
    return cls("boom", response=response, body=None)


async def test_rate_limited_model_fails_over_to_next_model_immediately():
    client, create = make_client(
        [_error(groq.RateLimitError, 429, "600"), tool_response(VALID)],
        groq_model_fast="model-a", groq_model_quality="model-b",
    )
    assert (await call(client)).priority_score == 4
    assert [c.kwargs["model"] for c in create.call_args_list] == ["model-a", "model-b"]


async def test_cooldown_skips_a_rate_limited_model_and_all_limited_raises_for_deferral():
    client, create = make_client(
        [_error(groq.RateLimitError, 429, "600")] * 2, groq_model_fast="a", groq_model_quality="b"
    )
    with pytest.raises(LLMRateLimitError):
        await call(client)
    with pytest.raises(LLMRateLimitError):  # both cooling down -> no API call at all
        await call(client)
    assert create.call_count == 2


async def test_retired_model_is_skipped():
    client, create = make_client(
        [_error(groq.NotFoundError, 404), tool_response(VALID)], groq_model_fast="old", groq_model_quality="new"
    )
    await call(client)
    assert create.call_args.kwargs["model"] == "new" and client._cooling("old")