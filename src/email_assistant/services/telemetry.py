"""Persists LLM call metrics (no prompt content) for the Evaluation page."""

from __future__ import annotations

from typing import Any

from email_assistant.db.repositories import MetricsRepository
from email_assistant.db.session import Database


class MetricsRecorder:
    """Callable passed to the LLM client as ``recorder``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def __call__(self, fields: dict[str, Any]) -> None:
        async with self._db.session() as session:
            await MetricsRepository(session).add_llm_call(**fields)