from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest
from src.decision.cost_calculator import CostCalculator


def _calc():
    cfg = DecisionConfig.from_yaml()
    return CostCalculator(cfg.costs)


def _req(spread_bps=None, fee_bps=None):
    return DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance="moderate",
        urgency="normal",
        timeframe="1_week",
        timeframe_days=7,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
    )


def test_single_conversion_cost():
    calc = _calc()
    req = _req(spread_bps=5.0, fee_bps=0.0)
    c = calc.calculate_cost_estimate(req, is_staged=False)
    assert c.staged_multiplier == 1.0
    assert c.total_bps == 5.0


def test_staged_conversion_cost():
    calc = _calc()
    req = _req(spread_bps=5.0, fee_bps=0.0)
    c = calc.calculate_cost_estimate(req, is_staged=True)
    assert c.staged_multiplier > 1.0
    assert c.total_bps > 5.0


def test_use_default_spread_when_missing():
    calc = _calc()
    req = _req(spread_bps=None, fee_bps=0.0)
    c = calc.calculate_cost_estimate(req, is_staged=False)
    assert c.spread_bps == calc.cfg.default_spread_bps


def test_use_request_spread():
    calc = _calc()
    req = _req(spread_bps=7.0, fee_bps=0.0)
    c = calc.calculate_cost_estimate(req, is_staged=False)
    assert c.spread_bps == 7.0
    assert c.total_bps == 7.0


def test_zero_fee_and_custom_fee():
    calc = _calc()
    req0 = _req(spread_bps=5.0, fee_bps=0.0)
    c0 = calc.calculate_cost_estimate(req0, is_staged=False)
    assert c0.total_bps == 5.0
    reqc = _req(spread_bps=5.0, fee_bps=2.0)
    cc = calc.calculate_cost_estimate(reqc, is_staged=False)
    assert cc.total_bps == 7.0

