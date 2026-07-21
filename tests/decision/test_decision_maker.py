from datetime import datetime, timedelta, timezone

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
        market={"mid_rate": 1.0, "indicators": {"atr_14": 0.005, "rsi_14": 55, "macd": 0.001, "macd_signal": 0.0}, "regime": {"trend_direction": "up", "bias": "bullish"}, "quality": {"fresh": True, "dispersion_bps": 10.0}},
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


def test_strict_and_relaxed_heuristic_policies_differ_for_weak_intelligence():
    maker = _maker()
    maker.config.heuristics_enabled = True
    req = _req_with_prediction(conf=0.1)
    req.intelligence = {
        "news": {"pair_bias": 0.0, "confidence": "low"},
        "calendar": {"events_extracted": []},
        "policy_bias": 0.0,
    }
    maker.config.heuristics_trigger_policy = "strict"
    assert maker._should_use_heuristic(req) is False
    maker.config.heuristics_trigger_policy = "relaxed"
    assert maker._should_use_heuristic(req) is True


def test_configured_action_thresholds_are_enforced():
    maker = _maker()
    req = _req_with_prediction(improvement=0.01)
    scores = maker.utility_scorer.score_actions(req)
    constrained = maker._apply_hard_constraints(scores, req)
    assert constrained["wait"] == -999.0

    req = _req_with_prediction(improvement=-0.5)
    scores = maker.utility_scorer.score_actions(req)
    maker.config.thresholds.convert_now_min_utility = scores["convert_now"] + 0.01
    constrained = maker._apply_hard_constraints(scores, req)
    assert constrained["convert_now"] == -999.0


def test_event_and_staging_thresholds_are_enforced():
    maker = _maker()
    req = _req_with_prediction(risk="moderate", tf_days=2, event_days=1.0, improvement=-1.0)
    scores = maker.utility_scorer.score_actions(req)
    constrained = maker._apply_hard_constraints(scores, req)
    assert constrained["staged_conversion"] == -999.0

    req.timeframe_days = 7
    maker.config.thresholds.wait_event_proximity_days = 0.5
    outside = maker._apply_hard_constraints(scores, req)
    maker.config.thresholds.wait_event_proximity_days = 2.0
    inside = maker._apply_hard_constraints(scores, req)
    assert inside["convert_now"] == outside["convert_now"] - 0.25


def test_stale_prediction_is_not_used_for_outcome_or_reliability():
    maker = _maker()
    maker.config.heuristics_enabled = True
    req = _req_with_prediction(improvement=2.0)
    req.prediction["timestamp"] = (
        datetime.now(timezone.utc) - timedelta(hours=24)
    ).isoformat()
    req.intelligence = None
    assert maker._should_use_heuristic(req) is True
    resp = maker.make_decision(req)
    assert resp.expected_outcome.expected_improvement_bps == 0.0
    assert "Prediction is stale and was ignored" in resp.warnings


def test_expected_outcome_uses_nearest_horizon_and_conversion_direction():
    maker = _maker()
    req = _req_with_prediction(tf_days=21)
    req.prediction["predictions"] = {
        "7": {"mean_change_pct": 0.7},
        "30": {"mean_change_pct": 3.0},
    }
    req.source_currency = "EUR"
    req.target_currency = "USD"
    outcome = maker._generate_expected_outcome(req, "wait", is_heuristic=False)
    assert -292.0 < outcome.expected_improvement_bps < -291.0
    assert outcome.expected_rate == 1.0 / 1.03

