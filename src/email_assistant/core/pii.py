"""PII detection, reversible masking (for LLM prompts) and irreversible redaction (for logs).

Two usage modes:

* :class:`PiiMasker` instance - replaces PII with stable placeholders such as
  ``<EMAIL_1>`` before text is sent to the LLM, and restores originals in the
  model's output via :meth:`PiiMasker.unmask`.
* :meth:`PiiMasker.redact` - static, one-way replacement used for logs / audit rows.

Detected: e-mail addresses, payment-card numbers (Luhn-validated), US SSNs,
Indian Aadhaar and PAN identifiers, IPv4 addresses and phone numbers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR_RE = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{8,}\d(?!\w)")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _luhn_ok(value: str) -> bool:
    digits = [int(c) for c in _digits(value)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for idx, digit in enumerate(reversed(digits)):
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _phone_ok(value: str) -> bool:
    if _ISO_DATE_RE.fullmatch(value.strip()):
        return False
    return 10 <= len(_digits(value)) <= 15


def _ipv4_ok(value: str) -> bool:
    return all(0 <= int(part) <= 255 for part in value.split("."))


def _always(_: str) -> bool:
    return True


# (kind, pattern, validator) - order matters: most specific first.
_RULES: tuple[tuple[str, re.Pattern[str], Callable[[str], bool]], ...] = (
    ("EMAIL", _EMAIL_RE, _always),
    ("CARD", _CARD_RE, _luhn_ok),
    ("SSN", _SSN_RE, _always),
    ("AADHAAR", _AADHAAR_RE, _always),
    ("PAN", _PAN_RE, _always),
    ("IP", _IPV4_RE, _ipv4_ok),
    ("PHONE", _PHONE_RE, _phone_ok),
)

#: Matches any placeholder issued by :class:`PiiMasker`.
TOKEN_RE = re.compile(r"<(?:EMAIL|CARD|SSN|AADHAAR|PAN|IP|PHONE)_\d+>")


@dataclass
class PiiMasker:
    """Reversible PII masker. Create one instance per email being processed."""

    _forward: dict[tuple[str, str], str] = field(default_factory=dict)
    _reverse: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)

    def _token_for(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._forward:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            token = f"<{kind}_{self._counters[kind]}>"
            self._forward[key] = token
            self._reverse[token] = value
        return self._forward[key]

    def mask(self, text: str) -> str:
        """Replace PII in ``text`` with stable placeholders."""
        if not text:
            return text
        for kind, pattern, validator in _RULES:
            text = pattern.sub(
                lambda m, k=kind, v=validator: self._token_for(k, m.group(0)) if v(m.group(0)) else m.group(0),
                text,
            )
        return text

    def unmask(self, text: str) -> str:
        """Restore original values for every placeholder this masker issued."""
        for token, original in self._reverse.items():
            text = text.replace(token, original)
        return text

    def lookup(self, token: str) -> str | None:
        """Return the original value for a placeholder, or ``None`` if unknown."""
        return self._reverse.get(token)

    @staticmethod
    def redact(text: str) -> str:
        """One-way redaction for logs and audit trails."""
        if not text:
            return text
        for kind, pattern, validator in _RULES:
            text = pattern.sub(
                lambda m, k=kind, v=validator: f"[REDACTED_{k}]" if v(m.group(0)) else m.group(0),
                text,
            )
        return text


def scrub_structure(value: Any, _depth: int = 0) -> Any:
    """Recursively redact PII from strings inside dicts / lists (for logs and audit rows)."""
    if isinstance(value, str):
        return PiiMasker.redact(value)
    if _depth > 5:
        return value
    if isinstance(value, dict):
        return {k: scrub_structure(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_structure(v, _depth + 1) for v in value]
    return value
