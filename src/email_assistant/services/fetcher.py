"""Email fetcher: Gmail -> raw MIME -> cleaned :class:`ParsedEmail`."""

from __future__ import annotations

import structlog

from email_assistant.config import Settings
from email_assistant.integrations.gmail_client import GmailClient
from email_assistant.preprocessing.mime_parser import parse_raw_email
from email_assistant.schemas.parsed_email import ParsedEmail

logger = structlog.get_logger(__name__)


class EmailFetcher:
    """Lists new messages and returns cleaned, structured emails."""

    def __init__(self, gmail: GmailClient, settings: Settings) -> None:
        self._gmail = gmail
        self._settings = settings

    async def list_new_ids(self, query: str | None = None, max_results: int | None = None) -> list[str]:
        """Message ids matching ``query`` (defaults to ``GMAIL_QUERY``)."""
        return await self._gmail.list_message_ids(
            query or self._settings.gmail_query, max_results or self._settings.gmail_max_per_run
        )

    async def fetch(self, message_id: str) -> ParsedEmail:
        """Fetch, parse and clean one message."""
        raw = await self._gmail.get_raw_message(message_id)
        own = await self._gmail.own_email()
        parsed = parse_raw_email(
            raw.raw,
            gmail_id=raw.id,
            thread_id=raw.thread_id,
            label_ids=raw.label_ids,
            own_email=own,
            fallback_date=raw.internal_date,
            snippet=raw.snippet,
            max_chars=self._settings.llm_max_body_chars,
        )
        logger.info(
            "email_fetched",
            message_id=message_id,
            body_chars=len(parsed.body_text),
            raw_chars=parsed.raw_body_chars,
            automated=parsed.is_automated,
        )
        return parsed
