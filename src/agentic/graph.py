"""LangGraph workflow definition."""
import asyncio
import logging
from langgraph.graph import StateGraph, END
from src.agentic.state import AgentState
from src.agentic.nodes.market_data import market_data_node as market_data_node_async
from src.agentic.nodes.market_intelligence import market_intelligence_node as market_intelligence_node_async
from src.agentic.nodes.prediction import prediction_node as prediction_node_async

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
    workflow.add_node("nlu", nlu_node)
    workflow.add_node("market_data", lambda state: asyncio.run(market_data_node_async(state)))
    workflow.add_node("market_intelligence", lambda state: asyncio.run(market_intelligence_node_async(state)))
    workflow.add_node("price_prediction", lambda state: asyncio.run(prediction_node_async(state)))
    workflow.add_node("decision_engine", decision_engine_node)
    workflow.add_node("response", response_node)
    
    # Set entry point
    workflow.set_entry_point("nlu")
    
    # Add edges - parallel execution for Layer 1
    workflow.add_edge("nlu", "market_data")
    workflow.add_edge("nlu", "market_intelligence")
    
    # Both Layer 1 agents feed into Layer 2
    workflow.add_edge("market_data", "price_prediction")
    workflow.add_edge("market_intelligence", "price_prediction")
    
    # Sequential for Layer 2 and 3
    workflow.add_edge("price_prediction", "decision_engine")
    workflow.add_edge("decision_engine", "response")
    workflow.add_edge("response", END)
    
    return workflow.compile()


# Placeholder node functions (will be implemented in later phases)
def nlu_node(state: AgentState) -> AgentState:
    """Natural Language Understanding node (placeholder)."""
    logger.info("NLU node called")
    state["processing_stage"] = "layer1"
    return state


def market_data_node(state: AgentState) -> AgentState:
    """Market Data agent node (placeholder)."""
    logger.info("Market Data node called")
    # Only update market data specific fields
    return {
        "market_data_status": "success",
        "market_snapshot": {"rate": 0.85, "timestamp": "2024-01-01T00:00:00Z"}
    }


def market_intelligence_node(state: AgentState) -> AgentState:
    """Market Intelligence agent node (placeholder)."""
    logger.info("Market Intelligence node called")
    # Only update intelligence specific fields
    return {
        "intelligence_status": "success",
        "intelligence_report": {"sentiment": "neutral", "news_count": 5}
    }


def price_prediction_node(state: AgentState) -> AgentState:
    """Deprecated placeholder; real node is async in src.agentic.nodes.prediction."""
    logger.warning("Deprecated price_prediction_node called; using async node instead")
    return {
        "prediction_status": "partial",
        "price_forecast": None,
        "prediction_error": "deprecated_placeholder",
    }


def decision_engine_node(state: AgentState) -> AgentState:
    """Decision Engine agent node (placeholder)."""
    logger.info("Decision Engine node called")
    return {
        "decision_status": "success",
        "recommendation": {"action": "wait", "reason": "Rate may improve"}
    }


def response_node(state: AgentState) -> AgentState:
    """Response generation node (placeholder)."""
    logger.info("Response node called")
    return {
        "processing_stage": "complete"
    }
