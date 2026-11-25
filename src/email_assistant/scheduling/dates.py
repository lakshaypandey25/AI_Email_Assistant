"""Calendar arithmetic that must NOT be left to the language model."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

from email_assistant.scheduling.slots import Interval, resolve_datetime

_WEEKDAY_WORDS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "tues": 1, "wednesday": 2, "weds": 2, "thursday": 3, "thurs": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_SHORT: dict[str, int] = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_TEXT_RE = re.compile(r"\b(" + "|".join(_WEEKDAY_WORDS) + r")\b", re.I)


def parse_weekday(value: str | None) -> int | None:
    """'Friday' / 'fri' / 'FRI.' -> 4; anything else -> None."""
    if not value:
        return None
    word = value.strip().lower().rstrip(".,")
    return _WEEKDAY_WORDS.get(word, _SHORT.get(word[:3]) if len(word) >= 3 and word[:3] in _SHORT else None)


def weekdays_in_text(text: str) -> set[int]:
    """Weekdays named in free text (full names only, so 'sat'/'sun'/'mon' inside prose do not match)."""
    return {_WEEKDAY_WORDS[m.group(1).lower()] for m in _TEXT_RE.finditer(text or "")}


def next_occurrence(now_local: datetime, weekday: int, at: datetime) -> datetime:
    """Next date with ``weekday`` at the wall-clock time of ``at`` (today only if that time is still ahead)."""
    delta = (weekday - now_local.weekday()) % 7
    target = now_local.date() + timedelta(days=delta)
    candidate = datetime.combine(target, at.timetz().replace(tzinfo=None), tzinfo=at.tzinfo)
    if candidate <= now_local.astimezone(at.tzinfo):
        candidate += timedelta(days=7)
    return candidate


def fix_weekday(start: datetime, expected: int, now: datetime) -> datetime:
    """Return ``start`` unchanged if it already falls on ``expected``; else the next such weekday, same time."""
    if start.weekday() == expected:
        return start
    return next_occurrence(now.astimezone(start.tzinfo), expected, start)


def upcoming_calendar(now: datetime, tz: tzinfo, days: int = 14) -> str:
    """Plain-text calendar for the prompt: models look dates up instead of computing them."""
    local = now.astimezone(tz)
    lines = []
    for i in range(days):
        day = local + timedelta(days=i)
        tag = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        lines.append(f"{day:%a %Y-%m-%d}{tag}")
    return "\n".join(lines)


def resolve_proposed_slots(
    slots: Iterable[Any], *, hint: str | None, default_tz: tzinfo, duration: timedelta, now: datetime, text: str = ""
) -> list[Interval]:
    """Turn model-proposed slots into verified UTC intervals.

    The weekday the model reports for a slot (or, with a single slot, the only weekday named in the email)
    is treated as ground truth: a date on a different weekday is moved to the next matching weekday.
    """
    slot_list: Sequence[Any] = list(slots)
    named = weekdays_in_text(text)
    resolved: list[Interval] = []
    for slot in slot_list:
        try:
            start = resolve_datetime(slot.start, hint, default_tz)
            end = resolve_datetime(slot.end, hint, default_tz) if slot.end else start + duration
        except ValueError:
            continue
        if end <= start:
            end = start + duration
        expected = parse_weekday(getattr(slot, "weekday", None))
        if expected is None and len(slot_list) == 1 and len(named) == 1:
            expected = next(iter(named))
        if expected is not None:
            fixed = fix_weekday(start, expected, now)
            if fixed != start:
                end += fixed - start
                start = fixed
        resolved.append(Interval(start, end))
    return resolved

# --------------------------------------------------------------------------- explicit dates in the text
_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_TEXT_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}\b\.?(?:,?\s+(\d{{4}}))?", re.I)
_MDY_TEXT_RE = re.compile(rf"\b{_MON}\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_NUM_RE = re.compile(r"(?<![\d/.-])(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?![\d/])")
_ORD_RE = re.compile(rf"\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?{_MON})", re.I)
_TIME_AFTER_RE = re.compile(r"^\s*(?:am|pm|a\.m|p\.m|:|h\b|hrs?\b)", re.I)
_RANGE_BEFORE_RE = re.compile(r"(?:between|from|at|around|until|till)\s*$", re.I)


def _complete(day: int, month: int | None, year: int | None, today: date) -> date | None:
    """Fill in a missing month / year with the CURRENT one; if that lands in the past roll forward.

    * year given            -> used as is (a past date is reported later, not silently changed)
    * month missing         -> this month, else the next month that has that day
    * month given, no year  -> this year, else next year
    """
    if year is not None:
        try:
            return date(year, month or today.month, day)
        except ValueError:
            return None
    if month is not None:
        for y in (today.year, today.year + 1):
            try:
                candidate = date(y, month, day)
            except ValueError:
                continue
            if candidate >= today:
                return candidate
        return None
    y, mo = today.year, today.month
    for _ in range(13):
        try:
            candidate = date(y, mo, day)
        except ValueError:
            candidate = None
        if candidate is not None and candidate >= today:
            return candidate
        mo += 1
        if mo > 12:
            y, mo = y + 1, 1
    return None


def _year(value: str | None) -> int | None:
    if not value:
        return None
    number = int(value)
    return 2000 + number if number < 100 else number


def extract_dates(text: str, today: date, day_first: bool = True) -> list[date]:
    """Calendar dates written in free text, in order of appearance, de-duplicated.

    Understands ``2026-11-06``, ``6 Nov``, ``6th of November 2026``, ``Nov 6``, ``06/11``, ``06.11.2026`` and a bare
    ordinal (``on the 15th``). Missing month/year default to the current ones (see :func:`_complete`).
    ``day_first`` decides ``03/04`` (3 April vs 4 March); a number above 12 always decides it.
    """
    spans: list[tuple[int, int]] = []
    found: list[tuple[int, date]] = []

    def free(match: re.Match[str]) -> bool:
        return not any(match.start() < b and a < match.end() for a, b in spans)

    def take(match: re.Match[str], day: int, month: int | None, year: int | None) -> None:
        if not free(match):
            return
        completed = _complete(day, month, year, today)
        spans.append((match.start(), match.end()))
        if completed is not None:
            found.append((match.start(), completed))

    for m in _ISO_RE.finditer(text):
        take(m, int(m.group(3)), int(m.group(2)), int(m.group(1)))
    for m in _DMY_TEXT_RE.finditer(text):
        take(m, int(m.group(1)), _MONTHS[m.group(2).lower()], _year(m.group(3)))
    for m in _MDY_TEXT_RE.finditer(text):
        take(m, int(m.group(2)), _MONTHS[m.group(1).lower()], _year(m.group(3)))
    for m in _DOT_RE.finditer(text):
        take(m, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _NUM_RE.finditer(text):
        if _TIME_AFTER_RE.match(text[m.end() : m.end() + 8]) or _RANGE_BEFORE_RE.search(text[max(0, m.start() - 12) : m.start()]):
            continue  # '10-11 am', 'between 10-11' are times, not dates
        a, b = int(m.group(1)), int(m.group(2))
        day, month = (a, b) if (day_first or a > 12) and not b > 12 else (b, a)
        if 1 <= month <= 12 and 1 <= day <= 31:
            take(m, day, month, _year(m.group(3)))
    for m in _ORD_RE.finditer(text):
        if 1 <= int(m.group(1)) <= 31:
            take(m, int(m.group(1)), None, None)

    ordered: list[date] = []
    for _, value in sorted(found):
        if value not in ordered:
            ordered.append(value)
    return ordered


@dataclass
class SlotResolution:
    """Verified slots plus human-readable notes about what was corrected or looks inconsistent."""

    intervals: list[Interval] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # shown to the user in Approvals
    confirm: list[str] = field(default_factory=list)  # inconsistencies the reply should ask the sender to confirm


def resolve_slots_with_notes(
    slots: Iterable[Any],
    *,
    hint: str | None,
    default_tz: tzinfo,
    duration: timedelta,
    now: datetime,
    text: str = "",
    day_first: bool = True,
) -> SlotResolution:
    """Like :func:`resolve_proposed_slots`, but also trusts explicit dates written in the email.

    Order of trust: an explicit date in the text (assigned to slots in order when the counts match) > the weekday the
    model reports / the only weekday named in the text > the model's own date.
    """
    slot_list: Sequence[Any] = list(slots)
    named = weekdays_in_text(text)
    today = now.astimezone(default_tz).date()
    dates = extract_dates(text, today, day_first)
    targets = dates if dates and len(dates) == len(slot_list) else None
    result = SlotResolution()
    for index, slot in enumerate(slot_list):
        try:
            start = resolve_datetime(slot.start, hint, default_tz)
            end = resolve_datetime(slot.end, hint, default_tz) if slot.end else start + duration
        except ValueError:
            continue
        if end <= start:
            end = start + duration
        expected = parse_weekday(getattr(slot, "weekday", None))
        if expected is None and len(slot_list) == 1 and len(named) == 1:
            expected = next(iter(named))
        target = targets[index] if targets else None
        if target is not None:
            if start.date() != target:
                shifted = datetime.combine(target, start.timetz())
                end += shifted - start
                start = shifted
                result.notes.append(f"Date taken from the text of the email: {target:%A %d %b %Y}.")
            if expected is not None and start.weekday() != expected:
                names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                problem = f"The email says {names[expected]} but {target:%d %b %Y} is a {start:%A}."
                result.notes.append(problem + " Please confirm the date with the sender.")
                result.confirm.append(problem)
        elif expected is not None:
            fixed = fix_weekday(start, expected, now)
            if fixed != start:
                end += fixed - start
                start = fixed
        result.intervals.append(Interval(start, end))
    return result

