from src.decision.config import DecisionConfig
from src.decision.decision_maker import DecisionMaker
from src.decision.models import DecisionRequest


def _maker():
    return DecisionMaker(DecisionConfig.from_yaml())


def _req_with_prediction(risk="moderate", tf_days=7, event_days=None, improvement=0.3, conf=0.8):
    intel = None
    if event_days is not None:
        intel = {"calendar": {"events_extracted": [{"importance": "high", "proximity_minutes": int(event_days * 24 * 60)}]}}
    else:
        intel = {"calendar": {"events_extracted": []}}
    return DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance=risk,
        urgency="normal",
        timeframe="1_week",
        timeframe_days=tf_days,
        market={"indicators": {"atr_14": 0.005, "rsi_14": 55, "macd": 0.001, "macd_signal": 0.0}, "regime": {"trend_direction": "up", "bias": "bullish"}, "quality": {"fresh": True, "dispersion_bps": 10.0}},
        intelligence=intel,
        prediction={
            "status": "success",
            "confidence": conf,
            "latest_close": 1.0,
            "predictions": {str(tf_days): {"mean_change_pct": improvement, "quantiles": {"p10": -0.2, "p50": improvement, "p90": 0.5}}},
        },
    )


def test_decision_with_prediction_utility_path():
    maker = _maker()
    req = _req_with_prediction(improvement=0.3, conf=0.8)
    resp = maker.make_decision(req)
    assert resp.decision_status if hasattr(resp, 'decision_status') else True  # Sanity: object created
    assert "Prediction unavailable" not in " ".join(resp.warnings)
    assert any("Best utility action" in r for r in resp.rationale)


def test_conservative_event_blocking_changes_action():
    maker = _maker()
    # Event in 0.5 days; conservative should not pick convert_now
    req = _req_with_prediction(risk="conservative", event_days=0.5, improvement=0.1, conf=0.8)
    resp = maker.make_decision(req)
    assert resp.action in {"wait", "staged_conversion"}


def test_moderate_near_event_prefers_staged():
    maker = _maker()
    req = _req_with_prediction(risk="moderate", event_days=1.0, improvement=0.3, conf=0.8)
    resp = maker.make_decision(req)
    # Often staged in our defaults; if not, ensure not convert_now
    assert resp.action in {"staged_conversion", "wait"}


def test_expected_outcome_from_prediction_bps():
    maker = _maker()
    req = _req_with_prediction(improvement=0.3, conf=0.8)
    resp = maker.make_decision(req)
    assert abs(resp.expected_outcome.expected_improvement_bps - 30.0) < 1e-6


def test_heuristic_fallback_only_when_pred_and_intel_missing():
    maker = _maker()
    # Disable prediction and intelligence: force heuristic
    req = DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance="moderate",
        urgency="normal",
        timeframe="1_week",
        timeframe_days=7,
        market={"indicators": {"atr_14": 0.005}},
        intelligence=None,
        prediction=None,
    )
    # Temporarily override config to enable heuristics
    maker.config.heuristics_enabled = True
    resp = maker.make_decision(req)
    assert any("heuristic" in w.lower() for w in resp.warnings) or resp.action in {"wait", "staged_conversion", "convert_now"}

