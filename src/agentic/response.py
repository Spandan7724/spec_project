"""Response formatting helpers for the agentic workflow."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .state import AgentGraphState


def serialize_state(state: AgentGraphState) -> Dict[str, Any]:
    """Convert AgentGraphState into a serializable dict for downstream consumers."""
    return {
        "request": asdict(state.request),
        "market_analysis": state.market_analysis.__dict__,
        "economic_analysis": state.economic_analysis.__dict__,
        "risk_assessment": state.risk_assessment.__dict__,
        "provider_costs": state.provider_costs.__dict__,
        "recommendation": state.recommendation.__dict__,
        "meta": {
            "started_at": state.meta.started_at.isoformat(),
            "last_updated": state.meta.last_updated.isoformat(),
            "correlation_id": state.meta.correlation_id,
            "notes": list(state.meta.notes),
        },
    }
