from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest
from src.decision.heuristics import HeuristicDecisionMaker


def _maker():
    return HeuristicDecisionMaker(DecisionConfig.from_yaml())


def _req(**kw):
    return DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance=kw.get("risk_tolerance", "moderate"),
        urgency=kw.get("urgency", "normal"),
        timeframe=kw.get("timeframe", "1_week"),
        timeframe_days=kw.get("timeframe_days", 7),
        market=kw.get("market", {"indicators": {"rsi_14": 50, "macd": 0.0, "macd_signal": 0.0}}),
        intelligence=kw.get("intelligence", {"calendar": {"events_extracted": []}}),
    )


def test_event_gating_conservative_blocks():
    maker = _maker()
    req = _req(
        risk_tolerance="conservative",
        intelligence={"calendar": {"events_extracted": [{"importance": "high", "proximity_minutes": 12 * 60}]}},
    )
    d = maker.make_heuristic_decision(req)
    assert d["action"] == "wait"


def test_event_gating_aggressive_allows():
    maker = _maker()
    req = _req(
        risk_tolerance="aggressive",
        intelligence={"calendar": {"events_extracted": [{"importance": "high", "proximity_minutes": 12 * 60}]}},
    )
    d = maker.make_heuristic_decision(req)
    assert d["action"] in {"convert_now", "staged_conversion"}


def test_rsi_oversold_wait_and_overbought_convert():
    maker = _maker()
    req_os = _req(market={"indicators": {"rsi_14": 25}})
    d_os = maker.make_heuristic_decision(req_os)
    assert d_os["action"] == "wait"
    req_ob = _req(market={"indicators": {"rsi_14": 75}})
    d_ob = maker.make_heuristic_decision(req_ob)
    assert d_ob["action"] == "convert_now"


def test_macd_crossovers():
    maker = _maker()
    req_bull = _req(market={"indicators": {"macd": 0.001, "macd_signal": 0.0}})
    d_bull = maker.make_heuristic_decision(req_bull)
    assert d_bull["action"] == "wait"
    req_bear = _req(market={"indicators": {"macd": -0.001, "macd_signal": 0.0}})
    d_bear = maker.make_heuristic_decision(req_bear)
    assert d_bear["action"] == "convert_now"


def test_trend_based_rules():
    maker = _maker()
    req_up = _req(market={"indicators": {}, "regime": {"trend_direction": "up", "bias": "bullish"}})
    d_up = maker.make_heuristic_decision(req_up)
    assert d_up["action"] == "wait"
    req_down = _req(market={"indicators": {}, "regime": {"trend_direction": "down", "bias": "bearish"}})
    d_down = maker.make_heuristic_decision(req_down)
    assert d_down["action"] == "convert_now"


def test_urgent_override():
    maker = _maker()
    req = _req(urgency="urgent", market={"indicators": {"rsi_14": 50}})
    d = maker.make_heuristic_decision(req)
    assert d["action"] == "convert_now"
    # Very negative (RSI >= 75) forces staged
    req_neg = _req(urgency="urgent", market={"indicators": {"rsi_14": 80}})
    d_neg = maker.make_heuristic_decision(req_neg)
    assert d_neg["action"] == "staged_conversion"


def test_neutral_default_by_profile():
    maker = _maker()
    # Mixed signals path returns staged by default
    req_cons = _req(risk_tolerance="conservative", market={"indicators": {}})
    d_cons = maker.make_heuristic_decision(req_cons)
    assert d_cons["action"] in {"wait", "staged_conversion"}
    req_mod = _req(risk_tolerance="moderate", market={"indicators": {}})
    d_mod = maker.make_heuristic_decision(req_mod)
    assert d_mod["action"] in {"staged_conversion", "wait"}
    req_aggr = _req(risk_tolerance="aggressive", market={"indicators": {}})
    d_aggr = maker.make_heuristic_decision(req_aggr)
    assert d_aggr["confidence"] <= 0.6  # heuristic conf always lower


def test_mixed_signals_prefers_staged():
    maker = _maker()
    req = _req(market={"indicators": {"rsi_14": 50, "macd": 0.001, "macd_signal": 0.001}})
    d = maker.make_heuristic_decision(req)
    assert d["action"] == "staged_conversion"

