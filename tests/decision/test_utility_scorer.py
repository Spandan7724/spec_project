from datetime import datetime, timedelta, timezone

from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest
from src.decision.utility_scorer import UtilityScorer


def _scorer():
    return UtilityScorer(DecisionConfig.from_yaml())


def _base_request(**overrides):
    req = DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance=overrides.get("risk_tolerance", "moderate"),
        urgency=overrides.get("urgency", "normal"),
        timeframe=overrides.get("timeframe", "1_week"),
        timeframe_days=overrides.get("timeframe_days", 7),
        market=overrides.get(
            "market",
            {"mid_rate": 1.0, "indicators": {"rsi_14": 50, "atr_14": 0.002, "macd": 0.0}},
        ),
        intelligence=overrides.get("intelligence", {"upcoming_events": []}),
        prediction=overrides.get("prediction"),
    )
    return req


def test_score_actions_basic():
    scorer = _scorer()
    req = _base_request()
    scores = scorer.score_actions(req)
    assert set(scores.keys()) == {"convert_now", "staged_conversion", "wait"}


def test_urgent_favors_convert_now():
    scorer = _scorer()
    # No predicted improvement so urgent bonus should lift convert_now above others
    req = _base_request(urgency="urgent")
    scores = scorer.score_actions(req)
    assert scores["convert_now"] > scores["wait"]
    assert scores["convert_now"] > scores["staged_conversion"]


def test_flexible_favors_wait_with_positive_improvement():
    scorer = _scorer()
    # Provide positive predicted improvement so wait + flexible should be favored
    req = _base_request(
        urgency="flexible",
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.3}}},
    )
    scores = scorer.score_actions(req)
    assert scores["wait"] > scores["convert_now"]
    assert scores["wait"] >= scores["staged_conversion"]


def test_expected_improvement_from_prediction_overrides_technical():
    scorer = _scorer()
    # RSI neutral; prediction provides improvement
    req_pred = _base_request(
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.25}}}
    )
    req_none = _base_request(prediction=None)
    scores_pred = scorer.score_actions(req_pred)
    scores_none = scorer.score_actions(req_none)
    # Wait should be better when prediction says +0.25%
    assert scores_pred["wait"] > scores_none["wait"]


def test_risk_penalty_increases_near_event():
    scorer = _scorer()
    base = _base_request(
        intelligence={"upcoming_events": []},
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.2}}},
    )
    near_event = _base_request(
        intelligence={
            "upcoming_events": [
                {"importance": "high", "days_until": 0.5, "event_name": "CPI"}
            ]
        },
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.2}}},
    )
    s_base = scorer.score_actions(base)
    s_near = scorer.score_actions(near_event)
    # Higher penalty near event should reduce utilities across the board
    for k in s_base:
        assert s_near[k] <= s_base[k]


def test_conservative_penalizes_risk_more_than_aggressive():
    scorer = _scorer()
    # High ATR to emphasize risk
    market = {"mid_rate": 1.0, "indicators": {"rsi_14": 50, "atr_14": 0.01, "macd": 0.0}}
    cons = _base_request(risk_tolerance="conservative", market=market)
    aggr = _base_request(risk_tolerance="aggressive", market=market)
    s_cons = scorer.score_actions(cons)
    s_aggr = scorer.score_actions(aggr)
    # For same inputs, conservative should score lower due to higher risk weight/multiplier
    assert s_cons["wait"] < s_aggr["wait"]


def test_staging_reduces_risk_and_can_beat_convert_now_under_risk():
    scorer = _scorer()
    # Positive improvement with significant risk -> staged may beat convert_now
    req = _base_request(
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.5}}},
        market={"mid_rate": 1.0, "indicators": {"rsi_14": 50, "atr_14": 0.01, "macd": 0.0}},
    )
    scores = scorer.score_actions(req)
    assert scores["staged_conversion"] > scores["convert_now"]


def test_current_intelligence_bias_contract_is_used():
    scorer = _scorer()
    req = _base_request(
        market={},
        intelligence={
            "news": {"pair_bias": 1.0, "confidence": "high"},
            "policy_bias": 0.5,
        },
    )
    assert scorer._get_expected_improvement(req) > 0


def test_nested_calendar_events_add_risk_penalty():
    scorer = _scorer()
    req = _base_request(
        market={},
        intelligence={
            "calendar": {
                "events_extracted": [
                    {"importance": "high", "proximity_minutes": 60, "event": "CPI"}
                ]
            }
        },
    )
    profile = scorer._get_profile("moderate")
    assert scorer._calculate_risk_penalty(req, profile) > 0.4


def test_nearest_prediction_horizon_is_used():
    scorer = _scorer()
    req = _base_request(
        timeframe_days=21,
        prediction={
            "status": "success",
            "predictions": {
                "1": {"mean_change_pct": 0.1},
                "7": {"mean_change_pct": 0.7},
                "30": {"mean_change_pct": 3.0},
            },
        },
    )
    assert scorer._get_expected_improvement(req) == 3.0


def test_stale_prediction_is_ignored():
    scorer = _scorer()
    old = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    req = _base_request(
        market={"mid_rate": 1.0, "indicators": {"rsi_14": 50, "macd": 0.0}},
        intelligence=None,
        prediction={
            "status": "success",
            "timestamp": old,
            "predictions": {"7": {"mean_change_pct": 5.0}},
        },
    )
    assert scorer._get_expected_improvement(req) == 0.0


def test_inverse_conversion_flips_prediction_benefit():
    scorer = _scorer()
    req = _base_request(
        prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.5}}}
    )
    req.source_currency = "EUR"
    req.target_currency = "USD"
    assert -0.5 < scorer._get_expected_improvement(req) < -0.49

