"""Signed, single-use approval tokens.

The MCP server exposes HIGH-risk tools (send a draft, create a calendar event), but it only runs one when the call
carries a token proving that a human approval happened. Only the application's approval controller can mint such a
token: it shares a random secret with the MCP server process it spawns. An outside MCP client (Inspector, Claude
Desktop) never learns the secret, so it can never send mail or create invites.

A token is bound to the tool name and to the exact arguments (SHA-256), expires quickly, and works once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Any

#: Environment variable through which the agent hands the secret to the MCP server process it spawns.
SECRET_ENV = "EMAIL_ASSISTANT_MCP_SECRET"


class ApprovalSigner:
    """Issues and verifies approval tokens with an HMAC-SHA256 signature."""

    def __init__(self, secret: bytes, ttl_seconds: int = 300, clock: Callable[[], float] = time.time) -> None:
        if len(secret) < 16:
            raise ValueError("approval secret must be at least 16 bytes")
        self._secret = secret
        self._ttl = ttl_seconds
        self._clock = clock
        self._used: dict[str, int] = {}  # nonce -> expiry (replay protection, server side)

    @classmethod
    def random(cls, **kwargs: Any) -> ApprovalSigner:
        return cls(secrets.token_bytes(32), **kwargs)

    @staticmethod
    def digest(arguments: Mapping[str, Any]) -> str:
        canonical = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _sign(self, body: str) -> str:
        return hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).hexdigest()

    def issue(self, tool: str, arguments: Mapping[str, Any]) -> str:
        payload = {"t": tool, "h": self.digest(arguments), "e": int(self._clock() + self._ttl), "n": secrets.token_hex(8)}
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str, tool: str, arguments: Mapping[str, Any]) -> bool:
        """True exactly once for a genuine, unexpired token issued for this tool and these arguments."""
        try:
            body, signature = token.rsplit(".", 1)
            if not hmac.compare_digest(signature, self._sign(body)):
                return False
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        except (ValueError, TypeError, AttributeError):
            return False
        now = self._clock()
        self._used = {n: e for n, e in self._used.items() if e >= now}  # forget expired nonces
        if payload.get("t") != tool or payload.get("h") != self.digest(arguments) or payload.get("e", 0) < now:
            return False
        nonce = str(payload.get("n", ""))
        if not nonce or nonce in self._used:
            return False  # replay
        self._used[nonce] = int(payload["e"])
        return True