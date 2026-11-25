"""Async Gmail client (read inbox, create drafts, send approved drafts)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import parseaddr

import structlog
from email_assistant.exceptions import GoogleAPIError
from email_assistant.integrations.base import GoogleApiClient

logger = structlog.get_logger(__name__)

_CTRL_RE = re.compile(r"[\r\n]+")
_ADDR_RE = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")


@dataclass(frozen=True)
class RawMessage:
    """Gmail message in ``format=raw`` plus metadata."""

    id: str
    thread_id: str
    raw: bytes
    label_ids: list[str] = field(default_factory=list)
    internal_date: datetime | None = None
    snippet: str = ""


def _safe_header(value: str) -> str:
    """Strip CR/LF so values can never inject additional headers."""
    return _CTRL_RE.sub(" ", value).strip()


def _validate_address(address: str) -> str:
    _, addr = parseaddr(address)
    if not addr or not _ADDR_RE.match(addr):
        raise ValueError(f"Invalid e-mail address: {address!r}")
    return addr


def build_reply_message(
    *,
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
    references: str = "",
    cc: list[str] | None = None,
) -> EmailMessage:
    """Build an RFC 5322 reply with correct threading headers."""
    msg = EmailMessage()
    msg["To"] = _validate_address(to)
    if cc:
        msg["Cc"] = ", ".join(_validate_address(c) for c in cc)
    msg["Subject"] = _safe_header(subject)
    if in_reply_to:
        msg["In-Reply-To"] = _safe_header(in_reply_to)
        msg["References"] = _safe_header(f"{references} {in_reply_to}".strip())
    msg.set_content(body)
    return msg


class GmailClient(GoogleApiClient):
    """Gmail API v1 operations used by the assistant."""

    service_name = "gmail"
    service_version = "v1"
    _own_email: str | None = None

    async def own_email(self) -> str:
        """The authenticated mailbox address (cached; ``USER_EMAIL`` overrides)."""
        if self._settings.user_email:
            return self._settings.user_email.lower()
        if self._own_email is None:
            profile = await self._execute(lambda s: s.users().getProfile(userId="me"))
            self._own_email = str(profile["emailAddress"]).lower()
        return self._own_email

    async def list_message_ids(self, query: str, max_results: int = 20) -> list[str]:
        """Return ids of messages matching a Gmail search ``query`` (newest first)."""
        resp = await self._execute(
            lambda s: s.users().messages().list(userId="me", q=query, maxResults=max_results, labelIds=["INBOX"])
        )
        return [m["id"] for m in resp.get("messages", [])]

    async def get_raw_message(self, message_id: str) -> RawMessage:
        """Fetch one message as raw RFC 822 bytes."""
        resp = await self._execute(lambda s: s.users().messages().get(userId="me", id=message_id, format="raw"))
        raw_b64: str = resp["raw"]
        raw = base64.urlsafe_b64decode(raw_b64 + "=" * (-len(raw_b64) % 4))
        internal = resp.get("internalDate")
        return RawMessage(
            id=resp["id"],
            thread_id=resp.get("threadId", ""),
            raw=raw,
            label_ids=list(resp.get("labelIds", [])),
            internal_date=datetime.fromtimestamp(int(internal) / 1000, tz=UTC) if internal else None,
            snippet=resp.get("snippet", ""),
        )

    async def create_draft(
        self,
        *,
        thread_id: str,
        to: str,
        subject: str,
        body: str,
        in_reply_to: str = "",
        references: str = "",
        cc: list[str] | None = None,
    ) -> str:
        """Create a reply draft in the user's mailbox. Nothing is sent. Returns the draft id."""
        msg = build_reply_message(
            to=to, subject=subject, body=body, in_reply_to=in_reply_to, references=references, cc=cc
        )
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        payload = {"message": {"raw": raw, "threadId": thread_id}}
        resp = await self._execute(lambda s: s.users().drafts().create(userId="me", body=payload))
        return str(resp["id"])

    async def draft_status(self, draft_id: str, thread_id: str, since: datetime) -> str:
        """'draft' (still there), 'sent' (a reply left this thread after ``since``) or 'gone' (deleted)."""
        try:
            await self._execute(lambda s: s.users().drafts().get(userId="me", id=draft_id, format="minimal"))
            return "draft"
        except GoogleAPIError as exc:
            if exc.status != 404:
                raise
        try:
            thread = await self._execute(
                lambda s: s.users().threads().get(userId="me", id=thread_id, format="metadata", metadataHeaders=["From"])
            )
        except GoogleAPIError as exc:
            if exc.status == 404:
                return "gone"
            raise
        since_ms = int(since.timestamp() * 1000)
        for message in thread.get("messages", []):
            if "SENT" in message.get("labelIds", []) and int(message.get("internalDate", 0)) >= since_ms:
                return "sent"
        return "gone"

    async def thread_has_reply_since(self, thread_id: str, since: datetime) -> bool:
        """True if a message from the mailbox owner (label SENT) exists in the thread after ``since``."""
        try:
            thread = await self._execute(
                lambda s: s.users().threads().get(userId="me", id=thread_id, format="metadata", metadataHeaders=["From"])
            )
        except GoogleAPIError as exc:
            if exc.status == 404:
                return False
            raise
        since_ms = int(since.timestamp() * 1000)
        return any(
            "SENT" in m.get("labelIds", []) and int(m.get("internalDate", 0)) >= since_ms for m in thread.get("messages", [])
        )

    async def send_draft(self, draft_id: str) -> dict[str, str]:
        """Send an existing draft (HIGH risk - only invoked after human approval)."""
        resp = await self._execute(lambda s: s.users().drafts().send(userId="me", body={"id": draft_id}))
        return {"message_id": str(resp.get("id", "")), "thread_id": str(resp.get("threadId", ""))}
