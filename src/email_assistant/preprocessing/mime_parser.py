"""Parse RFC 5322 / MIME bytes (Gmail ``format=raw``) into a :class:`ParsedEmail`."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from email_assistant.preprocessing.auth_results import sender_authenticated
from email_assistant.preprocessing.cleaner import clean_body
from email_assistant.schemas.parsed_email import ParsedEmail

_AUTOMATED_LOCALPARTS = {
    "noreply", "no-reply", "donotreply", "do-not-reply", "notifications", "notification",
    "mailer-daemon", "postmaster", "bounce", "bounces", "alerts",
}  # fmt: skip
_CTRL_RE = re.compile(r"[\r\n\t]+")


def _header(msg: EmailMessage, name: str) -> str:
    """Return a decoded header with control characters removed (defends against header injection)."""
    try:
        value = msg.get(name, "")
    except Exception:  # malformed header objects
        return ""
    return _CTRL_RE.sub(" ", str(value or "")).strip()


def _addresses(msg: EmailMessage, name: str) -> list[str]:
    raw = [str(h) for h in (msg.get_all(name, []) or [])]
    return [addr.lower() for _, addr in getaddresses(raw) if addr]


def _extract_body(msg: EmailMessage) -> tuple[str, bool]:
    """Return ``(content, is_html)`` preferring text/plain over text/html."""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return "", False
    try:
        content = part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace")
    return str(content), part.get_content_type() == "text/html"


def _parse_date(msg: EmailMessage, fallback: datetime | None) -> datetime:
    raw = _header(msg, "Date")
    if raw:
        try:
            parsed = parsedate_to_datetime(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return fallback or datetime.now(UTC)


def parse_raw_email(
    raw: bytes,
    *,
    gmail_id: str,
    thread_id: str,
    label_ids: Sequence[str] = (),
    own_email: str = "",
    fallback_date: datetime | None = None,
    snippet: str = "",
    max_chars: int = 4000,
) -> ParsedEmail:
    """Convert raw message bytes into a cleaned :class:`ParsedEmail`.

    Args:
        raw: The RFC 822 message.
        gmail_id: Gmail message id.
        thread_id: Gmail thread id.
        label_ids: Gmail labels on the message.
        own_email: Authenticated user's address (for ``is_internal`` / ``is_from_self``).
        fallback_date: Used when the ``Date`` header is missing or invalid.
        snippet: Gmail-provided snippet.
        max_chars: Maximum cleaned body length.
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    from_header = _header(msg, "From")
    sender_name, sender_addr = parseaddr(from_header)
    sender_email = sender_addr.lower()
    sender_domain = sender_email.rsplit("@", 1)[-1] if "@" in sender_email else ""
    own = own_email.lower()
    own_domain = own.rsplit("@", 1)[-1] if "@" in own else ""

    body_raw, is_html = _extract_body(msg)
    body_clean = clean_body(body_raw, is_html=is_html, max_chars=max_chars)

    attachments = [p.get_filename() or "unnamed" for p in msg.iter_attachments()]

    local_part = sender_email.split("@", 1)[0]
    auto_submitted = _header(msg, "Auto-Submitted").lower()
    precedence = _header(msg, "Precedence").lower()
    has_unsub = bool(_header(msg, "List-Unsubscribe"))
    has_list_id = bool(_header(msg, "List-Id"))
    is_bulk = precedence in {"bulk", "list", "junk"} or has_list_id
    is_automated = (
        local_part in _AUTOMATED_LOCALPARTS
        or (auto_submitted not in {"", "no"})
        or is_bulk
    )

    subject = _header(msg, "Subject")
    in_reply_to = _header(msg, "In-Reply-To")

    auth_headers = [str(h) for h in (msg.get_all("Authentication-Results", []) or [])]
    has_invite = any(part.get_content_type() == "text/calendar" for part in msg.walk()) or any(
        name.lower().endswith(".ics") for name in attachments
    )

    return ParsedEmail(
        message_id=gmail_id,
        thread_id=thread_id,
        rfc_message_id=_header(msg, "Message-ID"),
        in_reply_to=in_reply_to,
        references=_header(msg, "References"),
        subject=subject,
        sender_name=sender_name.strip(),
        sender_email=sender_email,
        sender_domain=sender_domain,
        reply_to=_header(msg, "Reply-To") or None,
        to=_addresses(msg, "To"),
        cc=_addresses(msg, "Cc"),
        date=_parse_date(msg, fallback_date),
        body_text=body_clean,
        raw_body_chars=len(body_raw),
        attachments=attachments,
        label_ids=list(label_ids),
        snippet=snippet,
        is_automated=is_automated,
        has_list_unsubscribe=has_unsub,
        is_bulk=is_bulk,
        is_internal=bool(own_domain) and sender_domain == own_domain,
        is_from_self=bool(own) and sender_email == own,
        is_reply=bool(in_reply_to) or subject.lower().startswith(("re:", "aw:", "sv:")),
        authenticated=sender_authenticated(auth_headers, sender_domain),
        has_calendar_invite=has_invite,
    )
