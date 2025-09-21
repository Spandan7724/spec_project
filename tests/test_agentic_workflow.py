import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime

import pytest

from agentic import initialize_state, AgentGraphState
from agentic.graph import build_agentic_app
from agentic.nodes import run_decision_agent
from agentic.state import MarketAnalysis, EconomicAnalysis, RiskAssessment
from agentic.response import serialize_state


@pytest.mark.asyncio
async def test_initialize_state_and_graph_execution():
    payload = {
        "currency_pair": "usd/eur",
        "amount": 1000,
        "risk_tolerance": "low",
        "timeframe_days": 5,
    }

    state = initialize_state(payload)
    assert state.request.currency_pair == "USD/EUR"
    assert state.request.amount == 1000

    async def stub_market(s: AgentGraphState) -> AgentGraphState:
        market = MarketAnalysis(
            summary="Spot USD/EUR at 1.1000",
            bias="bullish",
            regime="uptrend",
            confidence=0.7,
            mid_rate=1.1,
            rate_timestamp=datetime.utcnow(),
        )
        return s.with_market(market)

    async def stub_economic(s: AgentGraphState) -> AgentGraphState:
        economic = EconomicAnalysis(
            summary="No major releases",
            overall_bias="neutral",
            upcoming_events=[],
            high_impact_events=[],
            confidence=0.5,
        )
        return s.with_economic(economic)

    async def stub_risk(s: AgentGraphState) -> AgentGraphState:
        risk = RiskAssessment(
            summary="Low volatility regime",
            risk_level="low",
            var_95=-0.01,
            volatility=0.04,
            confidence=0.6,
        )
        return s.with_risk(risk)

    app = build_agentic_app(
        market_node=stub_market,
        economic_node=stub_economic,
        risk_node=stub_risk,
        decision_node=run_decision_agent,
    )

    result_dict = await app.ainvoke(state)
    result_state = AgentGraphState(**result_dict)

    assert result_state.recommendation.action == "convert_now"
    assert result_state.recommendation.summary.startswith("Recommended action")

    serialized = serialize_state(result_state)
    assert serialized["market_analysis"]["bias"] == "bullish"
    assert serialized["recommendation"]["action"] == "convert_now"
