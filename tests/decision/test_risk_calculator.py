from src.decision.risk_calculator import RiskCalculator


def test_calculate_risk_summary_low_risk():
    calc = RiskCalculator()
    market = {"mid_rate": 1.0, "indicators": {"atr_14": 0.001}}
    intelligence = {"upcoming_events": []}
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.event_risk == "none"
    assert r.risk_level == "low"
    assert r.realized_vol_30d > 0
    assert r.var_95 > 0


def test_calculate_risk_summary_high_risk_event():
    calc = RiskCalculator()
    market = {"mid_rate": 1.0, "indicators": {"atr_14": 0.002}}
    intelligence = {
        "upcoming_events": [
            {"importance": "high", "days_until": 0.5, "event_name": "FOMC"}
        ]
    }
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.event_risk == "high"
    assert r.risk_level in ("moderate", "high")


def test_calculate_risk_summary_high_volatility():
    calc = RiskCalculator()
    market = {"mid_rate": 1.0, "indicators": {"atr_14": 0.03}}  # very high
    intelligence = {"upcoming_events": []}
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.realized_vol_30d > 15.0
    assert r.risk_level in ("moderate", "high")


def test_event_risk_classification():
    calc = RiskCalculator()
    market = {"mid_rate": 1.0, "indicators": {"atr_14": 0.001}}
    intelligence = {
        "upcoming_events": [
            {"importance": "high", "days_until": 2.0, "event_name": "CPI"},
            {"importance": "high", "days_until": 0.8, "event_name": "Jobs"},
        ]
    }
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.event_risk == "high"
    assert r.event_details in ("CPI", "Jobs")


def test_var_calculation_matches_formula():
    calc = RiskCalculator()
    market = {"mid_rate": 1.0, "indicators": {"atr_14": 0.001}}  # 0.1%
    r = calc.calculate_risk_summary(market, None)
    # var_95 approx 1.65 * 0.1% = 0.165%
    assert 0.16 <= r.var_95 <= 0.17


def test_atr_is_normalized_by_pair_rate_for_jpy_pair():
    calc = RiskCalculator()
    r = calc.calculate_risk_summary(
        {"mid_rate": 150.0, "indicators": {"atr_14": 1.0}}, None
    )
    assert 0.66 <= r.var_95 / 1.65 <= 0.67
    assert r.realized_vol_30d < 5.0


def test_nested_calendar_event_contract_is_used():
    calc = RiskCalculator()
    intelligence = {
        "calendar": {
            "events_extracted": [
                {
                    "importance": "high",
                    "proximity_minutes": 12 * 60,
                    "event": "CPI",
                }
            ]
        }
    }
    r = calc.calculate_risk_summary(
        {"mid_rate": 1.0, "indicators": {"atr_14": 0.001}}, intelligence
    )
    assert r.event_risk == "high"
    assert r.event_details == "CPI"

