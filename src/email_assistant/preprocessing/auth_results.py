"""Sender authentication from the ``Authentication-Results`` header that Gmail adds to every received message.

The ``From:`` address of an email can be forged by anyone. Gmail records whether SPF / DKIM / DMARC checks passed
for the claimed domain; we use that to be sure "someone@vitstudent.ac.in" really came from that domain.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_DKIM_RE = re.compile(r"dkim=pass[^;]*?header\.(?:d|i)=(?:[^@\s;]*@)?([a-z0-9.\-]+)")
_SPF_RE = re.compile(r"spf=pass[^;]*?smtp\.mailfrom=(?:[^@\s;]*@)?([a-z0-9.\-]+)")


def _aligned(a: str, b: str) -> bool:
    a, b = a.strip(".").lower(), b.strip(".").lower()
    return bool(a) and bool(b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def sender_authenticated(headers: Sequence[str], sender_domain: str) -> bool | None:
    """True if DMARC passed, or DKIM / SPF passed for a domain aligned with the sender's; False if checks ran but
    none did; ``None`` when the message carries no authentication header at all."""
    if not headers:
        return None
    text = " ".join(headers).lower()
    if "dmarc=pass" in text:
        return True
    for pattern in (_DKIM_RE, _SPF_RE):
        if any(_aligned(match.group(1), sender_domain) for match in pattern.finditer(text)):
            return True
    return False