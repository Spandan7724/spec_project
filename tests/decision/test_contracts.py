import pytest

from src.decision.contracts import (
    prediction_timeframe_bucket,
    select_prediction,
    upcoming_events,
)
from src.decision.models import DecisionRequest


def _request(**overrides):
    return DecisionRequest(
        amount=1000,
        currency_pair=overrides.get("currency_pair", "USD/EUR"),
        source_currency=overrides.get("source_currency"),
        target_currency=overrides.get("target_currency"),
        risk_tolerance="moderate",
        urgency="normal",
        timeframe="1_week",
        timeframe_days=7,
    )


def test_pair_order_is_the_default_conversion_direction():
    request = _request()
    assert request.source_currency == "USD"
    assert request.target_currency == "EUR"


def test_explicit_inverse_direction_is_valid():
    request = _request(source_currency="EUR", target_currency="USD")
    assert request.currency_pair == "USD/EUR"
    assert request.source_currency == "EUR"


def test_direction_must_use_the_market_pair_currencies():
    with pytest.raises(ValueError):
        _request(source_currency="GBP", target_currency="USD")


def test_horizon_tie_prefers_bucket_covering_full_window():
    prediction = {
        "predictions": {
            "7": {"mean_change_pct": 0.7},
            "21": {"mean_change_pct": 2.1},
        }
    }
    key, item = select_prediction(prediction, 14)
    assert key == "21"
    assert item["mean_change_pct"] == 2.1


def test_past_calendar_events_are_not_upcoming():
    intelligence = {
        "calendar": {
            "events_extracted": [
                {"event": "Old CPI", "importance": "high", "proximity_minutes": -5}
            ]
        }
    }
    assert upcoming_events(intelligence) == []


def test_numeric_window_overrides_categorical_prediction_default():
    assert prediction_timeframe_bucket("1_day", 21) == "1_week"
