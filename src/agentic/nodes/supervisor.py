"""Supervisor entry/exit nodes for LangGraph (Phase 4.4 pass-through)."""
from __future__ import annotations

from typing import Any, Dict

from src.agentic.state import AgentState
from src.supervisor.response_formatter import ResponseFormatter


def supervisor_start_node(state: AgentState) -> Dict[str, Any]:
    """Entry node placeholder: in Phase 4, simply mark NLU as complete."""
    return {"processing_stage": "nlu_complete"}


def supervisor_end_node(state: AgentState) -> Dict[str, Any]:
    """Exit node: format final recommendation for user-friendly output."""
    formatter = ResponseFormatter()
    rec = state.get("recommendation") or {}
    # Allow both plain decision dicts and wrapped {status:...}
    if rec and "status" not in rec:
        # Wrap in success envelope for formatter compatibility
        rec = {"status": "success", **rec}
    text = formatter.format_recommendation(rec)
    return {"final_response": text, "processing_stage": "complete"}

