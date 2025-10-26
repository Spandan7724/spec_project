"""Tests for state management."""
from src.agentic.state import (
    initialize_state,
    validate_state,
    add_conversation_turn,
    add_warning,
    add_error,
    update_processing_stage,
    record_agent_metric
)


def test_initialize_state():
    """Test state initialization."""
    state = initialize_state("Convert 1000 USD to EUR")
    
    assert state["user_query"] == "Convert 1000 USD to EUR"
    assert state["correlation_id"] is not None
    assert state["risk_tolerance"] == "moderate"
    assert state["urgency"] == "normal"
    assert state["processing_stage"] == "nlu"
    assert state["market_data_status"] == "pending"


def test_initialize_state_with_params():
    """Test state initialization with parameters."""
    state = initialize_state(
        "Convert money",
        currency_pair="USD/EUR",
        amount=1000.0,
        risk_tolerance="conservative"
    )
    
    assert state["currency_pair"] == "USD/EUR"
    assert state["amount"] == 1000.0
    assert state["risk_tolerance"] == "conservative"


def test_validate_state_valid():
    """Test validation of valid state."""
    state = initialize_state("Test query")
    is_valid, errors = validate_state(state)
    
    assert is_valid
    assert len(errors) == 0


def test_validate_state_invalid_risk():
    """Test validation with invalid risk tolerance."""
    state = initialize_state("Test query")
    state["risk_tolerance"] = "invalid"
    
    is_valid, errors = validate_state(state)
    
    assert not is_valid
    assert any("risk_tolerance" in e for e in errors)


def test_add_conversation_turn():
    """Test adding conversation turn."""
    state = initialize_state("Test query")
    state = add_conversation_turn(state, "user", "Hello")
    state = add_conversation_turn(state, "assistant", "Hi there")
    
    assert len(state["conversation_history"]) == 2
    assert state["conversation_history"][0]["role"] == "user"
    assert state["conversation_history"][1]["message"] == "Hi there"


def test_add_warning():
    """Test adding warnings."""
    state = initialize_state("Test query")
    state = add_warning(state, "Data quality issue")
    
    assert len(state["warnings"]) == 1
    assert state["warnings"][0] == "Data quality issue"


def test_add_error():
    """Test adding errors."""
    state = initialize_state("Test query")
    state = add_error(state, "API timeout")
    
    assert len(state["errors"]) == 1
    assert state["errors"][0] == "API timeout"


def test_update_processing_stage():
    """Test updating processing stage."""
    state = initialize_state("Test query")
    state = update_processing_stage(state, "layer1")
    
    assert state["processing_stage"] == "layer1"


def test_record_agent_metric():
    """Test recording agent metrics."""
    state = initialize_state("Test query")
    state = record_agent_metric(
        state,
        "market_data",
        150,
        "success"
    )
    
    assert "market_data" in state["agent_metrics"]
    assert state["agent_metrics"]["market_data"]["execution_time_ms"] == 150
    assert state["agent_metrics"]["market_data"]["status"] == "success"

