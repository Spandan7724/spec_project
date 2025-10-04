import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic.graph import build_agentic_app
from agentic.state import AgentGraphState, initialize_state


class DummyIndicators:
    trend_direction = "up"
    rsi_14 = 55.0

    @property
    def is_bullish(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"sma_20": 1.2, "sma_50": 1.1, "macd": 0.05}


class DummyEvent:
    def __init__(self, offset_days: float):
        self.event_id = "evt"
        self.title = "High Impact Event"
        self.currency = "USD"
        self.country = "US"
        self.release_date = datetime.now(timezone.utc) + timedelta(days=offset_days)
        self.impact = type("impact", (), {"value": "high"})()
        self.status = type("status", (), {"value": "scheduled"})()
        self.forecast_value = None
        self.previous_value = None
        self.actual_value = None
        self.relevance_score = 1.0
        self.is_high_impact = True

    def affects_pair(self, _pair: str) -> bool:  # pragma: no cover - simple stub
        return True


class DummyCalendar:
    def __init__(self, events):
        self.events = events
        self.sources = ["test_source"]

    def get_events_for_pair(self, _currency_pair: str):
        return self.events


class DummyHistoricalDataset:
    def __init__(self, returns: np.ndarray):
        self._returns = returns

    def to_dataframe(self):
        closes = np.cumprod(1 + self._returns)
        index = pd.date_range(end=datetime.now(timezone.utc), periods=len(self._returns))
        return pd.DataFrame({"close": closes}, index=index)


class HappyPathRateCollector:
    async def get_rate(self, base_currency: str, quote_currency: str):
        timestamp = datetime.now(timezone.utc)

        class DummyRate:
            def __init__(self):
                self.mid_rate = 1.1
                self.rate = 1.1
                self.timestamp = timestamp
                self.source = type("source", (), {"value": "dummy"})()

        dummy_rate = DummyRate()
        return type(
            "Result",
            (),
            {
                "has_data": True,
                "best_rate": dummy_rate,
                "rates": [dummy_rate],
                "success_rate": 100.0,
                "errors": [],
            },
        )()


class FallbackRateCollector:
    async def get_rate(self, base_currency: str, quote_currency: str):
        raise RuntimeError("rate providers offline")


class HappyPathIndicatorEngine:
    async def calculate_indicators(self, currency_pair: str):
        return DummyIndicators()


class EmptyIndicatorEngine:
    async def calculate_indicators(self, currency_pair: str):
        return None


class HappyPathMLPredictor:
    async def predict(self, request):
        class DummyResponse:
            model_id = "dummy"
            model_confidence = 0.7
            processing_time_ms = 5
            cached = False

            def __init__(self):
                self.predictions = {"1": {"mean": 0.001}}
                self.direction_probabilities = {"1": 0.6}

        return DummyResponse()


class UnavailableMLPredictor:
    async def predict(self, request):
        raise RuntimeError("ml predictor unavailable")


class HappyPathCalendarCollector:
    async def get_economic_calendar(self, days_ahead: int):
        return DummyCalendar([DummyEvent(2)])


class EmptyCalendarCollector:
    async def get_economic_calendar(self, days_ahead: int):
        return DummyCalendar([])


class HappyPathHistoricalCollector:
    async def get_historical_data(self, currency_pair: str, days: int):
        returns = np.array([0.001] * 120)
        return DummyHistoricalDataset(returns)


class EmptyHistoricalCollector:
    async def get_historical_data(self, currency_pair: str, days: int):
        return None


class EchoLLMManager:
    async def chat(self, messages):
        class Response:
            content = (
                "{\"summary\": \"LLM summary\", \"rationale\": [\"LLM rationale\"], "
                "\"warnings\": [], \"timeline\": \"Soon\", \"confidence\": 0.8}"
            )

        return Response()


def make_market_node(*, rate_collector, indicator_engine, ml_predictor):
    async def node(state):
        from agentic.nodes.market import run_market_agent

        return await run_market_agent(
            state,
            rate_collector=rate_collector,
            indicator_engine=indicator_engine,
            ml_predictor=ml_predictor,
        )

    return node


def make_economic_node(*, calendar_collector):
    async def node(state):
        from agentic.nodes.economic import run_economic_agent

        return await run_economic_agent(state, calendar_collector=calendar_collector)

    return node


def make_risk_node(*, historical_collector):
    async def node(state):
        from agentic.nodes.risk import run_risk_agent

        return await run_risk_agent(state, historical_collector=historical_collector)

    return node


def make_decision_node(*, llm_manager):
    async def node(state):
        from agentic.nodes.decision import run_decision_agent

        return await run_decision_agent(state, llm_manager=llm_manager)

    return node


@pytest.mark.asyncio
async def test_happy_path_workflow():
    state = initialize_state(
        {
            "currency_pair": "USD/EUR",
            "amount": 1000,
            "risk_tolerance": "moderate",
            "timeframe_days": 7,
        }
    )

    app = build_agentic_app(
        market_node=make_market_node(
            rate_collector=HappyPathRateCollector(),
            indicator_engine=HappyPathIndicatorEngine(),
            ml_predictor=HappyPathMLPredictor(),
        ),
        economic_node=make_economic_node(calendar_collector=HappyPathCalendarCollector()),
        risk_node=make_risk_node(historical_collector=HappyPathHistoricalCollector()),
        decision_node=make_decision_node(llm_manager=EchoLLMManager()),
    )

    result = await app.ainvoke(state)
    if isinstance(result, dict):
        result = AgentGraphState(**result)

    assert isinstance(result, AgentGraphState)
    assert result.recommendation.action in {"convert_now", "staged_conversion", "wait"}
    assert result.market_analysis.summary
    assert result.economic_analysis.summary
    assert result.risk_assessment.summary
    assert not result.recommendation.errors


@pytest.mark.asyncio
async def test_fallback_when_services_fail():
    state = initialize_state(
        {
            "currency_pair": "USD/EUR",
            "amount": 500,
            "risk_tolerance": "high",
            "timeframe_days": 3,
        }
    )

    app = build_agentic_app(
        market_node=make_market_node(
            rate_collector=FallbackRateCollector(),
            indicator_engine=EmptyIndicatorEngine(),
            ml_predictor=UnavailableMLPredictor(),
        ),
        economic_node=make_economic_node(calendar_collector=EmptyCalendarCollector()),
        risk_node=make_risk_node(historical_collector=EmptyHistoricalCollector()),
        decision_node=make_decision_node(llm_manager=None),
    )

    result = await app.ainvoke(state)
    if isinstance(result, dict):
        result = AgentGraphState(**result)

    assert isinstance(result, AgentGraphState)
    assert result.recommendation.action  # fallback action still produced
    assert result.market_analysis.errors
    # Economic agent should at least provide a summary even if no events exist.
    assert result.economic_analysis.summary
    assert result.risk_assessment.errors
    assert result.recommendation.errors  # decision agent should surface upstream issues
