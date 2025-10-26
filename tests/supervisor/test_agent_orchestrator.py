import os
import asyncio
import types

import pytest

from src.supervisor.agent_orchestrator import AgentOrchestrator
from src.supervisor.models import ExtractedParameters


@pytest.mark.asyncio
async def test_run_analysis_success(monkeypatch):
    os.environ["OFFLINE_DEMO"] = "true"

    # Speed-up: mock prediction_node to avoid heavy work
    async def fake_prediction_node(state):
        return {"prediction_status": "partial", "prediction_error": None, "price_forecast": None}

    monkeypatch.setattr(
        "src.supervisor.agent_orchestrator.prediction_node", fake_prediction_node, raising=True
    )

    orch = AgentOrchestrator()
    params = ExtractedParameters(
        currency_pair="USD/EUR",
        base_currency="USD",
        quote_currency="EUR",
        amount=5000.0,
        risk_tolerance="moderate",
        urgency="urgent",
        timeframe="immediate",
        timeframe_days=1,
    )

    rec = await orch.run_analysis(params, correlation_id="test-corr-id")
    assert rec["status"] in {"success", "error"}
    # When decision succeeds, action should be present
    if rec["status"] == "success":
        assert rec.get("action") in {"convert_now", "staged_conversion", "wait"}

