from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from email_assistant.scheduling.slots import (
    Interval, find_free_slots, format_interval, is_free, merge_intervals, resolve_datetime,
    within_working_hours, zone_from_hint,
)

IST = ZoneInfo("Asia/Kolkata")
WEEKDAYS = {0, 1, 2, 3, 4}


def iv(h1, m1, h2, m2, day=3, tz=IST):
    return Interval(datetime(2025, 3, day, h1, m1, tzinfo=tz), datetime(2025, 3, day, h2, m2, tzinfo=tz))


def test_interval_validation_and_overlap_with_buffer():
    with pytest.raises(ValueError):
        Interval(datetime(2025, 1, 1), datetime(2025, 1, 2))  # naive
    a, b = iv(9, 0, 10, 0), iv(10, 5, 10, 30)
    assert not a.overlaps(b)
    assert a.overlaps(b, buffer=timedelta(minutes=10))


def test_merge_intervals():
    merged = merge_intervals([iv(9, 0, 10, 0), iv(10, 0, 11, 0), iv(13, 0, 14, 0)])
    assert len(merged) == 2 and merged[0].duration == timedelta(hours=2)


def test_find_free_slots_respects_busy_buffer_and_hours():
    busy = [iv(9, 0, 11, 0)]
    start = datetime(2025, 3, 3, 0, 0, tzinfo=IST)
    slots = find_free_slots(
        busy=busy, search_start=start, search_end=start + timedelta(days=3), duration=timedelta(minutes=30),
        tz=IST, work_start=time(9), work_end=time(17, 30), work_days=WEEKDAYS, buffer=timedelta(minutes=10),
    )
    assert len(slots) == 3
    assert all(is_free(busy, s, timedelta(minutes=10)) for s in slots)
    assert all(within_working_hours(s, IST, time(9), time(17, 30), WEEKDAYS) for s in slots)
    assert slots[0].start.astimezone(IST).time() == time(11, 30)


def test_weekends_are_skipped():
    sat = datetime(2025, 3, 8, 0, 0, tzinfo=IST)
    assert not find_free_slots(
        busy=[], search_start=sat, search_end=sat + timedelta(days=1), duration=timedelta(minutes=30),
        tz=IST, work_start=time(9), work_end=time(17), work_days=WEEKDAYS,
    )


def test_working_hours_check():
    assert within_working_hours(iv(10, 0, 10, 30, day=4), IST, time(9), time(17), WEEKDAYS)
    assert not within_working_hours(iv(18, 0, 18, 30, day=4), IST, time(9), time(17), WEEKDAYS)
    assert not within_working_hours(iv(10, 0, 10, 30, day=8), IST, time(9), time(17), WEEKDAYS)  # Saturday


@pytest.mark.parametrize(
    "value,hint,offset_hours",
    [
        ("2025-03-04T15:00:00", "PST", -8),
        ("2025-07-04T15:00:00", "EST", -4),  # abbreviation maps to the DST-aware zone
        ("2025-03-04T15:00:00", "Europe/Berlin", 1),
        ("2025-03-04T15:00:00", "UTC+5:30", 5.5),
        ("2025-03-04T15:00:00", None, 5.5),  # falls back to default (IST)
        ("2025-03-04T15:00:00Z", "PST", 0),  # explicit offset wins
    ],
)
def test_resolve_datetime(value, hint, offset_hours):
    assert resolve_datetime(value, hint, IST).utcoffset() == timedelta(hours=offset_hours)


def test_unknown_hint_falls_back_and_bad_value_raises():
    assert zone_from_hint("Mars/Olympus", IST) is IST
    with pytest.raises(ValueError):
        resolve_datetime("not a date", None, IST)


def test_format_interval():
    assert format_interval(iv(15, 0, 15, 30), IST) == "Mon 03 Mar 2025, 15:00-15:30 (Asia/Kolkata)"
