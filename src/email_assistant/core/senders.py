"""Sender allow/deny list matching (addresses, ``@domain`` and bare domains)."""

from __future__ import annotations

import re
from collections.abc import Iterable


def parse_list(text: str) -> list[str]:
    """Split a comma / semicolon / newline separated list into lower-case entries."""
    return [p.strip().lower() for p in re.split(r"[,;\n]", text or "") if p.strip()]


def sender_matches(patterns: Iterable[str], address: str) -> bool:
    """True if ``address`` equals an entry, or belongs to an entry's domain (``@acme.com`` / ``acme.com``)."""
    addr = address.lower().strip()
    domain = addr.rsplit("@", 1)[-1]
    for pattern in patterns:
        if "@" in pattern and not pattern.startswith("@"):
            if addr == pattern:
                return True
        else:
            dom = pattern.lstrip("@")
            if domain == dom or domain.endswith("." + dom):
                return True
    return False
