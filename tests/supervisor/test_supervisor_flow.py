import os
import pytest

from src.supervisor.supervisor import Supervisor
from src.supervisor.models import SupervisorRequest


@pytest.mark.asyncio
async def test_supervisor_end_to_end_with_confirmation(monkeypatch):
    os.environ["OFFLINE_DEMO"] = "true"

    # Mock orchestrator to avoid heavy agent calls
    async def fake_run_analysis(params, correlation_id: str):
        return {
            "status": "success",
            "action": "convert_now",
            "confidence": 0.8,
            "timeline": "Immediate execution recommended",
            "rationale": ["Test rationale"],
            "warnings": [],
        }

    sup = Supervisor()
    monkeypatch.setattr(sup.orchestrator, "run_analysis", fake_run_analysis)

    # First turn: provide all info
    r1 = await sup.aprocess_request(
        SupervisorRequest(user_input="Convert 5000 USD to EUR today, urgent, moderate")
    )
    assert r1.requires_input is True  # Should go to confirmation

    # Confirm
    r2 = await sup.aprocess_request(SupervisorRequest(user_input="yes", session_id=r1.session_id))
    assert r2.requires_input is False
    assert r2.recommendation is not None
    assert "RECOMMENDATION" in r2.message

