"""Memory service: per-sender interaction history, notes and user feedback fed back into prompts."""

from __future__ import annotations

from dataclasses import dataclass, field

from email_assistant.db.repositories import EmailRepository, MemoryRepository
from email_assistant.db.session import Database
from email_assistant.schemas.parsed_email import ParsedEmail

GLOBAL_SCOPE = "global"


def sender_scope(email: str) -> str:
    return f"sender:{email.lower()}"


@dataclass
class MemoryContext:
    """What the assistant remembers that is relevant to the current email."""

    history: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Compact text block for prompts (already PII-light: stored summaries are masked)."""
        parts: list[str] = []
        if self.history:
            parts.append("Previous interactions with this sender (newest first):\n" + "\n".join(f"- {h}" for h in self.history))
        if self.notes:
            parts.append("Notes:\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.feedback:
            parts.append("User feedback / preferences to respect:\n" + "\n".join(f"- {f}" for f in self.feedback))
        return "\n".join(parts)


class MemoryService:
    """Reads and writes long-term memory."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_context(self, email: ParsedEmail, current_email_id: int | None = None) -> MemoryContext:
        async with self._db.session() as session:
            recent = await EmailRepository(session).recent_for_sender(email.sender_email, current_email_id, limit=5)
            entries = await MemoryRepository(session).for_scopes([GLOBAL_SCOPE, sender_scope(email.sender_email)], limit=10)
        ctx = MemoryContext()
        for rec in recent:
            day = rec.received_at.date().isoformat() if rec.received_at else "unknown date"
            ctx.history.append(f"{day} [{rec.category or 'unclassified'}] {rec.summary or rec.subject_redacted}"[:240])
        for entry in entries:
            (ctx.feedback if entry.kind in {"feedback", "preference"} else ctx.notes).append(entry.content[:300])
        return ctx

    async def add(self, scope: str, kind: str, content: str) -> None:
        """Persist a memory entry (``kind``: note | feedback | preference)."""
        async with self._db.session() as session:
            await MemoryRepository(session).add(scope, kind, content.strip())

    async def list_entries(self, limit: int = 200) -> list[dict[str, object]]:
        """All memory entries (newest first) for display / management."""
        async with self._db.session() as session:
            rows = await MemoryRepository(session).list_recent(limit)
        return [
            {"id": r.id, "scope": r.scope, "kind": r.kind, "content": r.content, "created_at": r.created_at}
            for r in rows
        ]

    async def delete(self, entry_id: int) -> bool:
        async with self._db.session() as session:
            return await MemoryRepository(session).delete(entry_id)
