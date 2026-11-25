from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from email_assistant.core.rate_limit import TokenWindow
from email_assistant.evaluation import metrics as m
from email_assistant.scheduling.dates import (
    fix_weekday, parse_weekday, resolve_proposed_slots, upcoming_calendar, weekdays_in_text,
)

TUESDAY = datetime(2026, 11, 3, 10, 0, tzinfo=UTC)


def slot(start, weekday=None, end=None):
    return NS(start=start, end=end, weekday=weekday)


def resolve(slots, text="", now=TUESDAY):
    return resolve_proposed_slots(slots, hint=None, default_tz=UTC, duration=timedelta(minutes=30), now=now, text=text)


def test_wrong_weekday_from_the_model_is_corrected_using_its_own_weekday_field():
    [fixed] = resolve([slot("2026-11-04T11:00:00", weekday="Friday")])
    assert fixed.start == datetime(2026, 11, 6, 11, 0, tzinfo=UTC) and fixed.duration == timedelta(minutes=30)


def test_weekday_is_taken_from_the_email_text_when_the_model_gives_none():
    [fixed] = resolve([slot("2026-11-04T11:00:00")], text="Let's have a meeting on Friday at 11 am.")
    assert fixed.start.date().isoformat() == "2026-11-06"


def test_correct_dates_are_left_alone_and_explicit_end_is_shifted_with_the_start():
    [ok] = resolve([slot("2026-11-06T11:00:00", weekday="fri")])
    assert ok.start.day == 6
    [moved] = resolve([slot("2026-11-04T11:00:00", weekday="Friday", end="2026-11-04T12:30:00")])
    assert moved.start.day == 6 and moved.duration == timedelta(minutes=90)


def test_two_slots_each_use_their_own_weekday_and_past_times_roll_to_next_week():
    slots = [slot("2026-11-04T10:00:00", "Tuesday"), slot("2026-11-04T10:00:00", "Thursday")]
    assert [s.start.day for s in resolve(slots)] == [10, 5]  # Tue 10:00 already passed today -> next Tuesday


def test_today_is_used_only_if_the_time_is_still_ahead():
    friday_morning = datetime(2026, 11, 6, 9, 0, tzinfo=UTC)
    assert fix_weekday(datetime(2026, 11, 5, 11, 0, tzinfo=UTC), 4, friday_morning).day == 6
    assert fix_weekday(datetime(2026, 11, 5, 8, 0, tzinfo=UTC), 4, friday_morning).day == 13


def test_weekday_parsing_helpers():
    assert parse_weekday("FRI.") == 4 and parse_weekday("Thursday") == 3 and parse_weekday("nonsense") is None
    assert weekdays_in_text("we sat down on Friday while the sun was out") == {4}


def test_calendar_table_lists_real_dates_for_the_prompt():
    table = upcoming_calendar(TUESDAY, UTC, 5).splitlines()
    assert table[0] == "Tue 2026-11-03 (today)" and table[1].endswith("(tomorrow)") and table[3].startswith("Fri 2026-11-06")


def test_token_window_blocks_until_old_usage_expires():
    clock = {"t": 0.0}
    window = TokenWindow(1000, clock=lambda: clock["t"])
    assert window.wait_needed(600) == 0
    window.reserve(600)
    assert window.wait_needed(600) == pytest.approx(60)
    clock["t"] = 61
    assert window.wait_needed(600) == 0 and TokenWindow(0).wait_needed(10**6) == 0


def test_accuracy_confusion_and_per_class_scores():
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b"), ("c", "a")]
    assert m.accuracy(pairs) == pytest.approx(0.6) and m.accuracy([]) is None
    matrix = m.confusion_matrix(pairs, ["a", "b", "c"])
    assert matrix["a"] == {"a": 1, "b": 1, "c": 0} and matrix["c"]["a"] == 1
    stats = m.per_class(pairs, ["a", "b", "c"])
    assert stats["a"]["precision"] == pytest.approx(0.5) and stats["a"]["recall"] == pytest.approx(0.5)
    assert stats["b"]["precision"] == pytest.approx(2 / 3) and stats["b"]["recall"] == 1.0
    assert stats["c"]["recall"] == 0.0 and stats["c"]["f1"] is None
    assert m.macro_f1(stats) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_percentiles_and_rates():
    assert m.percentile([1, 2, 3, 4, 5], 50) == 3 and m.percentile([10], 95) == 10 and m.percentile([], 50) is None
    assert m.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5) and m.median([5, 1, 3]) == 3
    assert m.rate(1, 4) == 0.25 and m.rate(1, 0) is None