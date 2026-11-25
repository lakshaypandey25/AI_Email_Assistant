"""Shared fixtures: deterministic settings and an in-memory async database."""

from __future__ import annotations

import pytest_asyncio

from email_assistant.config import Settings
from email_assistant.db.session import Database


def make_settings(**overrides) -> Settings:
    base = dict(
        groq_api_key="test-key",
        user_name="Dana Test",
        user_email="dana@example.com",
        user_timezone="UTC",
        database_url="sqlite+aiosqlite:///:memory:",
        google_requests_per_second=1000.0,
        groq_requests_per_minute=60000,
        max_concurrent_emails=1,
        enable_heuristic_prefilter=True,
        # tests must not depend on anything configured on this machine (.env, environment variables, defaults)
        allowed_senders="", ignore_senders="", vip_senders="",
        require_authenticated_sender=True, day_first_dates=True,
        send_mode="draft", auto_create_events=False, hold_risky_replies=True,
        reply_categories="action_required,meeting_request,priority_escalation", enable_scheduling=True,
        availability="", custom_instructions="", business_context="", default_tone="formal",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest_asyncio.fixture
async def settings() -> Settings:
    return make_settings()


@pytest_asyncio.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()
