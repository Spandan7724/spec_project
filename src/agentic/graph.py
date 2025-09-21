"""LangGraph workflow assembly for the currency assistant."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Awaitable, Callable

from langgraph.graph import END, StateGraph

from .state import AgentGraphState, initialize_state
from .nodes import (
    run_decision_agent,
    run_economic_agent,
    run_market_agent,
    run_risk_agent,
)

AgentNode = Callable[[AgentGraphState], Awaitable[AgentGraphState]]


def build_agentic_app(
    *,
    market_node: AgentNode = run_market_agent,
    economic_node: AgentNode = run_economic_agent,
    risk_node: AgentNode = run_risk_agent,
    decision_node: AgentNode = run_decision_agent,
):
    """Compile a LangGraph application for the agentic workflow."""
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("market", market_node)
    workflow.add_node("economic", economic_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("decision", decision_node)

    workflow.set_entry_point("market")
    workflow.add_edge("market", "economic")
    workflow.add_edge("economic", "risk")
    workflow.add_edge("risk", "decision")

    workflow.add_edge("decision", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def _cached_app() -> Any:  # noqa: ANN401 - returns langgraph runnable
    """Cache compiled workflow for default usage."""
    return build_agentic_app()


async def arun_agentic_workflow(
    payload: dict,
):
    """Run the workflow asynchronously and return the populated state."""
    state = initialize_state(payload)
    app = _cached_app()
    result = await app.ainvoke(state)
    if isinstance(result, AgentGraphState):
        return result
    if isinstance(result, dict):
        return AgentGraphState(**result)
    raise TypeError("Unexpected workflow result type")


def run_agentic_workflow(
    payload: dict,
):
    """Synchronous helper around the workflow."""
    return asyncio.run(arun_agentic_workflow(payload))
