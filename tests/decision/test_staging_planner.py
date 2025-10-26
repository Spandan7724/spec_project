from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest
from src.decision.staging_planner import StagingPlanner


def _planner():
    cfg = DecisionConfig.from_yaml()
    return StagingPlanner(cfg.staging, cfg.costs)


def _req(timeframe_days=7, urgency="normal", events=None, spread_bps=None):
    return DecisionRequest(
        amount=5000,
        currency_pair="USD/EUR",
        risk_tolerance="moderate",
        urgency=urgency,
        timeframe="1_week" if timeframe_days >= 6 else "1_day",
        timeframe_days=timeframe_days,
        market={"indicators": {"atr_14": 0.005}},
        intelligence={"upcoming_events": events or []},
        spread_bps=spread_bps,
    )


def test_short_timeframe_two_tranches():
    planner = _planner()
    req = _req(timeframe_days=5)
    plan = planner.create_staged_plan(req)
    assert plan.num_tranches == 2


def test_long_timeframe_three_tranches():
    planner = _planner()
    req = _req(timeframe_days=7)
    plan = planner.create_staged_plan(req)
    assert plan.num_tranches == 3


def test_urgent_front_loaded_three_tranches():
    planner = _planner()
    req = _req(timeframe_days=7, urgency="urgent")
    plan = planner.create_staged_plan(req)
    pcts = [t.percentage for t in plan.tranches]
    assert pcts[0] >= pcts[1] >= pcts[2]


def test_normal_equal_split_two_tranches():
    planner = _planner()
    req = _req(timeframe_days=4, urgency="normal")
    plan = planner.create_staged_plan(req)
    pcts = [round(t.percentage) for t in plan.tranches]
    assert plan.num_tranches == 2
    assert sum(pcts) == 100
    # Roughly equal
    assert abs(pcts[0] - pcts[1]) <= 5


def test_flexible_equal_split_three_tranches():
    planner = _planner()
    req = _req(timeframe_days=9, urgency="flexible")
    plan = planner.create_staged_plan(req)
    pcts_f = [t.percentage for t in plan.tranches]
    assert plan.num_tranches == 3
    assert abs(sum(pcts_f) - 100.0) < 0.01
    pcts = [round(x) for x in pcts_f]
    assert max(pcts) - min(pcts) <= 2


def test_avoid_high_impact_event_shift():
    planner = _planner()
    events = [{"importance": "high", "days_until": 3, "event_name": "Fed"}]
    req = _req(timeframe_days=7, events=events)
    plan = planner.create_staged_plan(req)
    exec_days = [t.execute_day for t in plan.tranches]
    # none should be within 0.5 days of 3 (integer schedule → not equal to 3)
    assert 3 not in exec_days


def test_multiple_events_increase_spacing_or_adjust():
    planner = _planner()
    events = [
        {"importance": "high", "days_until": 2, "event_name": "CPI"},
        {"importance": "high", "days_until": 3, "event_name": "Jobs"},
    ]
    req = _req(timeframe_days=7, events=events)
    plan = planner.create_staged_plan(req)
    exec_days = [t.execute_day for t in plan.tranches]
    # ensure monotonic and no duplicates
    assert exec_days == sorted(exec_days)
    assert len(exec_days) == len(set(exec_days))


def test_very_constrained_reduce_tranches():
    planner = _planner()
    events = [
        {"importance": "high", "days_until": 0, "event_name": "E1"},
        {"importance": "high", "days_until": 1, "event_name": "E2"},
        {"importance": "high", "days_until": 2, "event_name": "E3"},
    ]
    req = _req(timeframe_days=3, events=events)
    plan = planner.create_staged_plan(req)
    # Might need to reduce to 2 tranches
    assert plan.num_tranches <= 2


def test_no_intelligence_simple_spacing():
    planner = _planner()
    req = _req(timeframe_days=6, events=[])
    plan = planner.create_staged_plan(req)
    exec_days = [t.execute_day for t in plan.tranches]
    assert exec_days == sorted(exec_days)
    assert len(exec_days) == len(set(exec_days))


def test_extra_cost_calculation_positive():
    planner = _planner()
    req = _req(timeframe_days=7, spread_bps=5.0)
    plan = planner.create_staged_plan(req)
    assert plan.total_extra_cost_bps >= 0


def test_tranche_rationale_generation():
    planner = _planner()
    events = [{"importance": "high", "days_until": 4, "event_name": "ECB"}]
    req = _req(timeframe_days=7, events=events)
    plan = planner.create_staged_plan(req)
    assert all(t.rationale for t in plan.tranches)
