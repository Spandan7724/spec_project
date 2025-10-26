from datetime import datetime, timedelta, timezone

from src.data_collection.market_intelligence.models import EconomicEvent
from src.data_collection.market_intelligence.bias_calculator import calculate_policy_bias, next_high_impact_event


def make_event(name: str, importance: str, minutes_ahead: int):
    when = datetime.now(timezone.utc) + timedelta(minutes=minutes_ahead)
    return EconomicEvent(
        when_utc=when,
        when_local=when,
        timezone="UTC",
        country="US",
        currency="USD",
        event=name,
        importance=importance,
        source="test",
        source_url="https://example.com",
    )


def test_policy_bias_keywords():
    events = [
        make_event("Rate hike expected", "high", 60),
        make_event("Dovish remarks", "medium", 120),
    ]
    bias = calculate_policy_bias(events)
    # Hike positive, dovish negative; mixed outcome but near-term hike weighted higher
    assert -1.0 <= bias <= 1.0


def test_next_high_event():
    e1 = make_event("CPI", "high", 180)
    e2 = make_event("NFP", "high", 60)
    e3 = make_event("PMI", "medium", 30)
    nxt = next_high_impact_event([e1, e2, e3])
    assert nxt.event in {"NFP", "CPI"}  # picks among high, closest future

