"""Tests for graph structure."""
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state


def test_graph_creation():
    """Test graph can be created."""
    graph = create_graph()
    assert graph is not None


def test_graph_execution(monkeypatch):
    # Use offline demo mode so network/LLM isn't required during unit test
    monkeypatch.setenv("OFFLINE_DEMO", "true")
    """Test basic graph execution."""
    graph = create_graph()
    state = initialize_state("Convert 1000 USD to EUR")
    
    # Execute graph
    result = graph.invoke(state)
    
    # Check state was updated
    assert result["processing_stage"] == "complete"
    assert result["market_data_status"] == "success"
    assert result["intelligence_status"] == "success"
    assert result["prediction_status"] == "success"
    assert result["decision_status"] == "success"
