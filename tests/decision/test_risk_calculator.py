from src.decision.risk_calculator import RiskCalculator


def test_calculate_risk_summary_low_risk():
    calc = RiskCalculator()
    market = {"indicators": {"atr_14": 0.001}}
    intelligence = {"upcoming_events": []}
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.event_risk == "none"
    assert r.risk_level == "low"
    assert r.realized_vol_30d > 0
    assert r.var_95 > 0


def test_calculate_risk_summary_high_risk_event():
    calc = RiskCalculator()
    market = {"indicators": {"atr_14": 0.002}}
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
    market = {"indicators": {"atr_14": 0.02}}  # very high
    intelligence = {"upcoming_events": []}
    r = calc.calculate_risk_summary(market, intelligence)
    assert r.realized_vol_30d > 15.0
    assert r.risk_level in ("moderate", "high")


def test_event_risk_classification():
    calc = RiskCalculator()
    market = {"indicators": {"atr_14": 0.001}}
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
    market = {"indicators": {"atr_14": 0.001}}  # 0.1%
    r = calc.calculate_risk_summary(market, None)
    # var_95 approx 1.65 * 0.1% = 0.165%
    assert 0.16 <= r.var_95 <= 0.17

