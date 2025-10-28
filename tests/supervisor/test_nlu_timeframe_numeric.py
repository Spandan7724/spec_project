from __future__ import annotations

from src.supervisor.nlu_extractor import NLUExtractor


def test_extract_numeric_days():
    ex = NLUExtractor(use_llm=False)
    p = ex.extract("convert 1000 usd to eur in 21 days")
    assert p.timeframe_days == 21
    assert p.timeframe in (None, "1_day", "1_week", "1_month")  # categorical may be None


def test_extract_weeks():
    ex = NLUExtractor(use_llm=False)
    p = ex.extract("convert to euros in 2 weeks")
    assert p.timeframe_days == 14
    # ensure not misparsed as '1_day'
    assert p.timeframe != "1_day"


def test_extract_range_days():
    ex = NLUExtractor(use_llm=False)
    p = ex.extract("convert 1000 usd to eur in 3-5 days")
    assert p.window_days is not None
    assert p.window_days["start"] == 3 and p.window_days["end"] == 5
    assert p.timeframe_mode == "duration"


def test_extract_hours():
    ex = NLUExtractor(use_llm=False)
    p = ex.extract("convert in 12 hours")
    assert p.timeframe_hours == 12
    assert p.time_unit == "hours"
    assert p.timeframe_days == 0


def test_extract_deadline_absolute():
    ex = NLUExtractor(use_llm=False)
    p = ex.extract("convert by 2025-11-15")
    assert p.deadline_utc is not None
    assert p.timeframe_mode in ("deadline", "duration")
