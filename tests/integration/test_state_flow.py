"""Integration tests for state flow through graph."""


def test_complete_state_flow(monkeypatch):
    """Test complete state flow through graph."""
    # Use offline demo mode to avoid slow LLM/API calls
    monkeypatch.setenv("OFFLINE_DEMO", "true")
    
    from src.agentic.graph import create_graph
    from src.agentic.state import initialize_state, validate_state
    
    # Initialize state
    state = initialize_state(
        "Should I convert 5000 USD to EUR today?",
        currency_pair="USD/EUR",
        amount=5000.0,
        risk_tolerance="moderate",
        urgency="normal",
        timeframe="1_week"
    )
    
    # Validate initial state
    is_valid, errors = validate_state(state)
    assert is_valid, f"Invalid state: {errors}"
    
    # Create and execute graph
    graph = create_graph()
    result = graph.invoke(state)
    
    # Verify state progression
    assert result["correlation_id"] == state["correlation_id"]
    assert result["processing_stage"] == "complete"
    assert all(
        result[f"{agent}_status"] == "success"
        for agent in ["market_data", "intelligence", "prediction", "decision"]
    )

