"""LangGraph workflow definition."""
import asyncio
import logging
from langgraph.graph import StateGraph, END
from src.agentic.state import AgentState
from src.agentic.nodes.market_data import market_data_node as market_data_node_async
from src.agentic.nodes.market_intelligence import market_intelligence_node as market_intelligence_node_async
from src.agentic.nodes.prediction import prediction_node as prediction_node_async
from src.agentic.nodes.decision import decision_node as decision_engine_node_real
from src.agentic.nodes.supervisor import supervisor_start_node, supervisor_end_node

logger = logging.getLogger(__name__)


def create_graph() -> StateGraph:
    """
    Create the multi-agent workflow graph.
    
    Graph structure:
        START -> NLU -> Layer1_Parallel -> Layer2 -> Layer3 -> Response -> END
    
    Returns:
        Compiled StateGraph
    """
    # Create graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes (market_data uses real async node wrapped for sync invoke)
    workflow.add_node("supervisor_start", supervisor_start_node)
    workflow.add_node("market_data", lambda state: asyncio.run(market_data_node_async(state)))
    workflow.add_node("market_intelligence", lambda state: asyncio.run(market_intelligence_node_async(state)))
    workflow.add_node("price_prediction", lambda state: asyncio.run(prediction_node_async(state)))
    workflow.add_node("decision_engine", decision_engine_node_real)
    workflow.add_node("supervisor_end", supervisor_end_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor_start")
    
    # Add edges - parallel execution for Layer 1
    workflow.add_edge("supervisor_start", "market_data")
    workflow.add_edge("supervisor_start", "market_intelligence")
    
    # Both Layer 1 agents feed into Layer 2
    workflow.add_edge("market_data", "price_prediction")
    workflow.add_edge("market_intelligence", "price_prediction")
    
    # Sequential for Layer 2 and 3
    workflow.add_edge("price_prediction", "decision_engine")
    workflow.add_edge("decision_engine", "supervisor_end")
    workflow.add_edge("supervisor_end", END)
    
    return workflow.compile()


# Deprecated placeholders retained for reference only (no longer wired into graph)
