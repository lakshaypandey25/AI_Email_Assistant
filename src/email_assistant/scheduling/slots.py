"""Timezone resolution and free-slot computation.

Pure, deterministic and dependency-free (stdlib only) so it can be unit-tested thoroughly.
All :class:`Interval` instances are normalised to UTC; wall-clock logic (working hours,
day boundaries) is evaluated in the user's timezone so DST transitions are handled correctly.
"""

from __future__ import annotations

import re
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

#: Common abbreviations -> IANA zone. Ambiguous ones (CST, IST) use the most common reading.
TZ_ABBREVIATIONS: dict[str, str] = {
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles", "PT": "America/Los_Angeles",
    "MST": "America/Denver", "MDT": "America/Denver", "MT": "America/Denver",
    "CST": "America/Chicago", "CDT": "America/Chicago", "CT": "America/Chicago",
    "EST": "America/New_York", "EDT": "America/New_York", "ET": "America/New_York",
    "UTC": "UTC", "GMT": "UTC", "Z": "UTC",
    "BST": "Europe/London", "CET": "Europe/Paris", "CEST": "Europe/Paris",
    "EET": "Europe/Athens", "EEST": "Europe/Athens",
    "IST": "Asia/Kolkata", "SGT": "Asia/Singapore", "HKT": "Asia/Hong_Kong",
    "JST": "Asia/Tokyo", "KST": "Asia/Seoul", "AEST": "Australia/Sydney", "AEDT": "Australia/Sydney",
    "NZST": "Pacific/Auckland", "NZDT": "Pacific/Auckland",
}  # fmt: skip

_OFFSET_RE = re.compile(r"^(?:UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.I)


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open time interval ``[start, end)`` stored in UTC."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Interval requires timezone-aware datetimes")
        object.__setattr__(self, "start", self.start.astimezone(UTC))
        object.__setattr__(self, "end", self.end.astimezone(UTC))
        if self.end <= self.start:
            raise ValueError("Interval end must be after start")

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def overlaps(self, other: Interval, buffer: timedelta = timedelta(0)) -> bool:
        """True if the intervals overlap once ``other`` is padded by ``buffer`` on both sides."""
        return self.start < other.end + buffer and other.start - buffer < self.end


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping / touching intervals."""
    merged: list[Interval] = []
    for current in sorted(intervals):
        if merged and current.start <= merged[-1].end:
            if current.end > merged[-1].end:
                merged[-1] = Interval(merged[-1].start, current.end)
        else:
            merged.append(current)
    return merged


def is_free(busy: Sequence[Interval], candidate: Interval, buffer: timedelta = timedelta(0)) -> bool:
    """True if ``candidate`` (padded by ``buffer``) does not collide with any busy interval."""
    return not any(candidate.overlaps(b, buffer) for b in busy)


def within_working_hours(
    candidate: Interval,
    tz: tzinfo,
    work_start: time,
    work_end: time,
    work_days: Iterable[int],
) -> bool:
    """True if the whole interval lies inside working hours on a working day (in ``tz``)."""
    local_start = candidate.start.astimezone(tz)
    local_end = candidate.end.astimezone(tz)
    if local_start.date() != local_end.date():
        return False
    if local_start.weekday() not in set(work_days):
        return False
    return work_start <= local_start.time() and local_end.time() <= work_end

DAY_CODES: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WINDOW_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(?:-|–|—|to)\s*(\d{1,2}):(\d{2})\s*$", re.I)


def parse_windows(text: str) -> tuple[tuple[time, time], ...]:
    """'09:00-12:30, 14:00-17:30' -> sorted, non-overlapping windows. Raises ValueError with a friendly message."""
    windows: list[tuple[time, time]] = []
    for part in re.split(r"[;,]", text or ""):
        if not part.strip():
            continue
        match = _WINDOW_RE.match(part)
        if not match:
            raise ValueError(f"Cannot read '{part.strip()}' - use HH:MM-HH:MM, e.g. 09:00-12:30")
        h1, m1, h2, m2 = (int(g) for g in match.groups())
        if not (0 <= h1 <= 23 and 0 <= h2 <= 23 and 0 <= m1 <= 59 and 0 <= m2 <= 59):
            raise ValueError(f"'{part.strip()}' is not a valid time range")
        start, end = time(h1, m1), time(h2, m2)
        if end <= start:
            raise ValueError(f"'{part.strip()}': the end must be after the start")
        windows.append((start, end))
    windows.sort()
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
        if next_start < prev_end:
            raise ValueError("Availability windows on the same day must not overlap")
    return tuple(windows)


def parse_time_off(text: str) -> tuple[tuple[date, date], ...]:
    """One entry per line / comma: 'YYYY-MM-DD' or 'YYYY-MM-DD..YYYY-MM-DD'."""
    ranges: list[tuple[date, date]] = []
    for part in re.split(r"[\n;,]", text or ""):
        part = part.strip()
        if not part:
            continue
        try:
            first, _, last = part.partition("..")
            start = date.fromisoformat(first.strip())
            end = date.fromisoformat(last.strip()) if last else start
        except ValueError as exc:
            raise ValueError(f"Cannot read time off '{part}' - use YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD") from exc
        if end < start:
            raise ValueError(f"Time off '{part}': the end date is before the start date")
        ranges.append((start, end))
    return tuple(ranges)


@dataclass(frozen=True)
class Availability:
    """When the user can take meetings: per-weekday windows plus days off. Evaluated in the user's timezone."""

    windows: dict[int, tuple[tuple[time, time], ...]]
    time_off: tuple[tuple[date, date], ...] = ()

    def windows_on(self, day: date) -> tuple[tuple[time, time], ...]:
        if any(first <= day <= last for first, last in self.time_off):
            return ()
        return self.windows.get(day.weekday(), ())

    def contains(self, candidate: Interval, tz: tzinfo) -> bool:
        """True if the whole interval fits inside one availability window (and not on a day off)."""
        start, end = candidate.start.astimezone(tz), candidate.end.astimezone(tz)
        if start.date() != end.date():
            return False
        return any(ws <= start.time() and end.time() <= we for ws, we in self.windows_on(start.date()))

    def explain(self, candidate: Interval, tz: tzinfo) -> str:
        """Why a time is (not) available - shown in the Settings tester and given to the reply writer."""
        start = candidate.start.astimezone(tz)
        if any(first <= start.date() <= last for first, last in self.time_off):
            return "that day is marked as time off"
        day_windows = self.windows.get(start.weekday(), ())
        if not day_windows:
            return f"you are not available on {start:%A}s"
        if not self.contains(candidate, tz):
            hours = ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in day_windows)
            return f"outside your {start:%A} hours ({hours})"
        return "inside your availability"

    def to_dict(self) -> dict[str, object]:
        return {
            "windows": {
                DAY_CODES[d]: [f"{a:%H:%M}-{b:%H:%M}" for a, b in ws] for d, ws in sorted(self.windows.items()) if ws
            },
            "time_off": [f"{a}..{b}" if a != b else str(a) for a, b in self.time_off],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Availability:
        """Parse the stored JSON. Raises ValueError on any problem."""
        try:
            data = json.loads(text)
            raw_windows = data.get("windows", {})
        except (ValueError, AttributeError) as exc:
            raise ValueError("Availability is not valid JSON") from exc
        windows: dict[int, tuple[tuple[time, time], ...]] = {}
        for code, spans in raw_windows.items():
            if code not in DAY_CODES:
                raise ValueError(f"Unknown day '{code}'")
            parsed = parse_windows(",".join(spans))
            if parsed:
                windows[DAY_CODES.index(code)] = parsed
        return cls(windows, parse_time_off("\n".join(data.get("time_off", []))))

    def describe(self) -> str:
        """Compact summary, e.g. 'Mon-Fri 09:00-17:30 · Sun 14:00-16:00'."""
        groups: list[tuple[int, int, tuple[tuple[time, time], ...]]] = []
        for day in range(7):
            ws = self.windows.get(day, ())
            if not ws:
                continue
            if groups and groups[-1][1] == day - 1 and groups[-1][2] == ws:
                groups[-1] = (groups[-1][0], day, ws)
            else:
                groups.append((day, day, ws))

        def label(a: int, b: int) -> str:
            return DAY_CODES[a].title() if a == b else f"{DAY_CODES[a].title()}-{DAY_CODES[b].title()}"

        parts = [f"{label(a, b)} " + ", ".join(f"{x:%H:%M}-{y:%H:%M}" for x, y in ws) for a, b, ws in groups]
        text = " · ".join(parts) or "never available"
        return text + (f" · {len(self.time_off)} time-off period(s)" if self.time_off else "")


def default_availability(start: time, end: time, days: Iterable[int]) -> Availability:
    """Availability equivalent to the old 'one window on these weekdays' settings."""
    return Availability({d: ((start, end),) for d in sorted(set(days))})

def find_free_slots(
    *,
    busy: Sequence[Interval],
    search_start: datetime,
    search_end: datetime,
    duration: timedelta,
    tz: tzinfo,
    availability: Availability | None = None,
    work_start: time | None = None,
    work_end: time | None = None,
    work_days: Iterable[int] | None = None,
    step: timedelta = timedelta(minutes=30),
    buffer: timedelta = timedelta(0),
    max_slots: int = 3,
    max_per_day: int = 2,
    min_gap: timedelta = timedelta(hours=2),
) -> list[Interval]:
    """Up to ``max_slots`` conflict-free slots inside the user's availability.

    ``availability`` replaces the older work_start/work_end/work_days arguments (still accepted).
    """
    if availability is None:
        availability = default_availability(work_start or time(9), work_end or time(17), work_days or range(5))
    slots: list[Interval] = []
    per_day: dict[object, int] = defaultdict(int)
    day = search_start.astimezone(tz).date()
    last_day = search_end.astimezone(tz).date()
    lower, upper = search_start.astimezone(UTC), search_end.astimezone(UTC)

    while day <= last_day and len(slots) < max_slots:
        for window_start, window_end in availability.windows_on(day):
            cursor = datetime.combine(day, window_start, tzinfo=tz)
            limit = datetime.combine(day, window_end, tzinfo=tz)
            while cursor + duration <= limit and len(slots) < max_slots and per_day[day] < max_per_day:
                candidate = Interval(cursor, cursor + duration)
                if candidate.start >= lower and candidate.end <= upper and is_free(busy, candidate, buffer):
                    slots.append(candidate)
                    per_day[day] += 1
                    cursor = cursor + duration + min_gap
                else:
                    cursor = cursor + step
        day += timedelta(days=1)
    return slots

def _subtract(start: datetime, end: datetime, busy: Sequence[Interval], buffer: timedelta) -> list[tuple[datetime, datetime]]:
    """Parts of ``[start, end)`` not covered by any busy interval (each padded by ``buffer``)."""
    free: list[tuple[datetime, datetime]] = []
    cursor = start
    for item in sorted(busy):
        block_start, block_end = item.start - buffer, item.end + buffer
        if block_end <= cursor or block_start >= end:
            continue
        if block_start > cursor:
            free.append((cursor, block_start))
        cursor = max(cursor, block_end)
        if cursor >= end:
            break
    if cursor < end:
        free.append((cursor, end))
    return free


def free_windows(
    *,
    availability: Availability,
    busy: Sequence[Interval],
    start: datetime,
    end: datetime,
    tz: tzinfo,
    min_length: timedelta,
    buffer: timedelta = timedelta(0),
) -> list[tuple[date, list[tuple[datetime, datetime]]]]:
    """Per day: the user's availability windows minus calendar-busy time, clipped to ``[start, end]``.

    Only ranges at least ``min_length`` long are kept; days with nothing free are omitted.
    """
    lower, upper = start.astimezone(UTC), end.astimezone(UTC)
    days: list[tuple[date, list[tuple[datetime, datetime]]]] = []
    day = start.astimezone(tz).date()
    last_day = end.astimezone(tz).date()
    while day <= last_day:
        ranges: list[tuple[datetime, datetime]] = []
        for window_start, window_end in availability.windows_on(day):
            seg_start = max(datetime.combine(day, window_start, tzinfo=tz).astimezone(UTC), lower)
            seg_end = min(datetime.combine(day, window_end, tzinfo=tz).astimezone(UTC), upper)
            if seg_end - seg_start < min_length:
                continue
            for a, b in _subtract(seg_start, seg_end, busy, buffer):
                if b - a >= min_length:
                    ranges.append((a.astimezone(tz), b.astimezone(tz)))
        if ranges:
            days.append((day, ranges))
        day += timedelta(days=1)
    return days


def format_free_windows(days: Sequence[tuple[date, Sequence[tuple[datetime, datetime]]]]) -> list[str]:
    """One line per day, e.g. ``Mon 09 Nov: 09:00-11:00, 14:00-17:30``."""
    return [f"{day:%a %d %b}: " + ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in ranges) for day, ranges in days]

def zone_from_hint(hint: str | None, default: tzinfo) -> tzinfo:
    """Resolve a timezone hint (abbreviation, IANA name, or ``UTC+5:30``) to a ``tzinfo``."""
    if not hint or not hint.strip():
        return default
    text = hint.strip()
    abbr = TZ_ABBREVIATIONS.get(text.upper().replace(".", ""))
    if abbr:
        return ZoneInfo(abbr)
    offset = _OFFSET_RE.match(text)
    if offset:
        sign = -1 if offset.group(1) == "-" else 1
        delta = timedelta(hours=int(offset.group(2)), minutes=int(offset.group(3) or 0))
        return timezone(sign * delta)
    try:
        return ZoneInfo(text)
    except Exception:  # ZoneInfoNotFoundError, ValueError, OSError
        return default


def resolve_datetime(value: str, hint: str | None, default: tzinfo) -> datetime:
    """Parse an ISO 8601 string into an aware datetime.

    Naive values are interpreted in the zone named by ``hint`` (or ``default`` if absent/unknown).
    Raises ``ValueError`` on unparseable input.
    """
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone_from_hint(hint, default))
    return parsed


def format_interval(interval: Interval, tz: tzinfo, tz_label: str | None = None) -> str:
    """Human-readable range, e.g. ``Tue 04 Mar 2025, 15:00-15:30 (Europe/Berlin)``."""
    start = interval.start.astimezone(tz)
    end = interval.end.astimezone(tz)
    label = tz_label or getattr(tz, "key", None) or start.tzname() or "UTC"
    return f"{start:%a %d %b %Y}, {start:%H:%M}-{end:%H:%M} ({label})"
