from datetime import datetime, timedelta, timezone

from src.decision.models import DecisionRequest
from src.decision.confidence_aggregator import ConfidenceAggregator


def _agg():
    return ConfidenceAggregator()


def _req(**kw):
    return DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance=kw.get("risk_tolerance", "moderate"),
        urgency=kw.get("urgency", "normal"),
        timeframe=kw.get("timeframe", "1_week"),
        timeframe_days=kw.get("timeframe_days", 7),
        market=kw.get("market"),
        intelligence=kw.get("intelligence"),
        prediction=kw.get("prediction"),
        warnings=kw.get("warnings", []),
    )


def test_all_components_available_high_confidence():
    agg = _agg()
    market = {"quality": {"fresh": True, "dispersion_bps": 10.0}, "indicators": {"rsi_14": 65, "macd": 0.01}}
    intelligence = {"news": {"confidence": "high"}, "calendar": {"next_high_event": {}}}
    prediction = {"confidence": 0.8}
    req = _req(market=market, intelligence=intelligence, prediction=prediction)
    result = agg.aggregate_confidence(req, utility_scores={"convert_now": 0.6, "wait": 0.2, "staged": 0.3})
    assert 0.6 <= result["overall_confidence"] <= 1.0


def test_prediction_confidence_uses_selected_horizon_score():
    prediction = {
        "confidence": 0.9,
        "predictions": {
            "1": {"mean_change_pct": 0.1},
            "7": {"mean_change_pct": 0.2},
        },
        "model_info": {"confidence_by_horizon": {"1": 0.6, "7": 0.3}},
    }
    req = _req(timeframe_days=7, prediction=prediction)

    result = _agg().aggregate_confidence(req)

    assert result["component_confidences"]["prediction"] == 0.3


def test_missing_prediction_penalty():
    agg = _agg()
    market = {"quality": {"fresh": True, "dispersion_bps": 10.0}, "indicators": {"rsi_14": 65, "macd": 0.01}}
    intelligence = {"news": {"confidence": "high"}, "calendar": {"next_high_event": {}}}
    req = _req(market=market, intelligence=intelligence, prediction=None)
    result = agg.aggregate_confidence(req, utility_scores={"convert_now": 0.55, "wait": 0.5})
    assert result["overall_confidence"] < 0.8
    assert any("missing_components" in p for p in result["penalties_applied"])


def test_stale_data_penalty_if_timestamp_old():
    agg = _agg()
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    req = _req(market=None, intelligence=None, prediction={"confidence": 0.7, "timestamp": old_ts})
    result = agg.aggregate_confidence(req)
    # Missing components penalties + stale prediction
    assert any("stale_prediction" in p for p in result["penalties_applied"])


def test_staleness_uses_configured_age_threshold():
    agg = ConfidenceAggregator(max_prediction_age_hours=1)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    req = _req(prediction={"confidence": 0.7, "timestamp": old_ts})
    result = agg.aggregate_confidence(req)
    assert "stale_prediction" in result["penalties_applied"]


def test_low_utility_spread_penalty():
    agg = _agg()
    req = _req(market=None, intelligence=None, prediction=None)
    result = agg.aggregate_confidence(req, utility_scores={"a": 0.31, "b": 0.3})
    assert any("low_utility_spread" in p for p in result["penalties_applied"])


def test_heuristic_penalty_applied():
    agg = _agg()
    req = _req(market=None, intelligence=None, prediction=None)
    result = agg.aggregate_confidence(req, is_heuristic=True)
    assert any("heuristic_penalty" in p for p in result["penalties_applied"])
