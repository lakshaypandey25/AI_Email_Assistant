"""Knowledge base: company facts / FAQs / policies used to ground replies (retrieval by keyword overlap)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from email_assistant.db.repositories import KnowledgeRepository
from email_assistant.db.session import Database

_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_STOP = frozenset(
    "that this with from have your will would could should please thanks thank there their about which when what "
    "where been were they them then than also just into some more very much only over such like need want".split()
)


@dataclass(frozen=True)
class KnowledgeItem:
    id: int
    title: str
    content: str
    enabled: bool


def tokens(text: str) -> set[str]:
    """Lower-case content words (>=4 chars, minus common stop words)."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def rank(items: Sequence[KnowledgeItem], query: str, k: int = 3, max_chars: int = 1500) -> list[KnowledgeItem]:
    """Top ``k`` entries by keyword overlap (title words weigh double). Entries with no overlap are dropped."""
    q = tokens(query)
    scored: list[tuple[float, KnowledgeItem]] = []
    for item in items:
        if not item.enabled:
            continue
        score = len(q & tokens(item.content)) + 2 * len(q & tokens(item.title))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    chosen: list[KnowledgeItem] = []
    used = 0
    for _, item in scored[:k]:
        if used + len(item.content) > max_chars and chosen:
            break
        chosen.append(item)
        used += len(item.content)
    return chosen


def render(items: Sequence[KnowledgeItem], max_chars: int = 1500) -> str:
    return "\n".join(f"[{i.title}] {i.content[:max_chars]}" for i in items)


class KnowledgeService:
    """CRUD + retrieval over knowledge entries."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, title: str, content: str) -> None:
        async with self._db.session() as session:
            await KnowledgeRepository(session).add(title, content)

    async def list_all(self) -> list[KnowledgeItem]:
        async with self._db.session() as session:
            rows = await KnowledgeRepository(session).list_all()
        return [KnowledgeItem(r.id, r.title, r.content, r.enabled) for r in rows]

    async def set_enabled(self, entry_id: int, enabled: bool) -> bool:
        async with self._db.session() as session:
            return await KnowledgeRepository(session).set_enabled(entry_id, enabled)

    async def delete(self, entry_id: int) -> bool:
        async with self._db.session() as session:
            return await KnowledgeRepository(session).delete(entry_id)

    async def retrieve(self, query: str, k: int = 3) -> str:
        """Relevant entries rendered for the drafting prompt ('' if nothing matches)."""
        async with self._db.session() as session:
            rows = await KnowledgeRepository(session).list_enabled()
        items = [KnowledgeItem(r.id, r.title, r.content, r.enabled) for r in rows]
        return render(rank(items, query, k))