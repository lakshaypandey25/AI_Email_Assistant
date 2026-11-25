"""Normalised representation of an inbound email after MIME parsing and cleaning."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ParsedEmail(BaseModel):
    """A cleaned, LLM-ready email plus sender context and routing signals."""

    model_config = ConfigDict(frozen=False)

    message_id: str = Field(description="Gmail message id")
    thread_id: str
    rfc_message_id: str = ""
    in_reply_to: str = ""
    references: str = ""
    subject: str = ""
    sender_name: str = ""
    sender_email: str
    sender_domain: str = ""
    reply_to: str | None = None
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    date: datetime
    body_text: str = Field(default="", description="Cleaned body: no quotes, signature or boilerplate")
    raw_body_chars: int = 0
    attachments: list[str] = Field(default_factory=list)
    label_ids: list[str] = Field(default_factory=list)
    snippet: str = ""

    # --- sender-context signals -------------------------------------------
    is_automated: bool = False
    has_list_unsubscribe: bool = False
    is_bulk: bool = False
    is_internal: bool = False
    is_from_self: bool = False
    is_reply: bool = False
    authenticated: bool | None = Field(default=None, description="SPF/DKIM/DMARC verdict for the sender domain; None = unknown")
    has_calendar_invite: bool = False  # carries a text/calendar part or .ics attachment

    @property
    def known_addresses(self) -> set[str]:
        """All addresses visible in the headers (used to constrain LLM-proposed recipients)."""
        found = {self.sender_email.lower()}
        found.update(a.lower() for a in self.to)
        found.update(a.lower() for a in self.cc)
        return found
