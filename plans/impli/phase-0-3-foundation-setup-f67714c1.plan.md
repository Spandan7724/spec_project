<!-- f67714c1-a54f-4e8d-9617-16955f212afc ecf74cf6-cce3-4b50-a5d4-d07f23943ac1 -->
# Phase 0.3: LangGraph State Design

## Overview

Create the complete state management system for LangGraph including state schema definition, initialization, validation, and a basic graph skeleton.

## Implementation Steps

### Step 1: Create State Schema (src/agentic/state.py)

Define the complete TypedDict schema for agent state:

```python
"""LangGraph state schema and management."""
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import uuid


class AgentState(TypedDict, total=False):
    """Complete state schema for the multi-agent system."""
    
    # Request metadata
    correlation_id: str
    timestamp: datetime
    
    # User request
    user_query: str
    currency_pair: str
    base_currency: str
    quote_currency: str
    amount: Optional[float]
    
    # Runtime parameters
    risk_tolerance: str  # conservative, moderate, aggressive
    urgency: str  # urgent, normal, flexible
    timeframe: str  # immediate, 1_day, 1_week, 1_month
    
    # Layer 1: Market Data
    market_snapshot: Optional[Dict[str, Any]]
    market_data_status: str  # pending, success, error, partial
    market_data_error: Optional[str]
    
    # Layer 1: Market Intelligence
    intelligence_report: Optional[Dict[str, Any]]
    intelligence_status: str  # pending, success, error, partial
    intelligence_error: Optional[str]
    
    # Layer 2: Price Prediction
    price_forecast: Optional[Dict[str, Any]]
    prediction_status: str  # pending, success, error, partial
    prediction_error: Optional[str]
    
    # Layer 3: Decision Engine
    recommendation: Optional[Dict[str, Any]]
    decision_status: str  # pending, success, error
    decision_error: Optional[str]
    
    # Conversation
    conversation_history: List[Dict[str, str]]
    clarifications_needed: List[str]
    
    # System
    warnings: List[str]
    errors: List[str]
    processing_stage: str  # nlu, layer1, layer2, layer3, response
    agent_metrics: Dict[str, Dict[str, Any]]


def initialize_state(user_query: str, **kwargs) -> AgentState:
    """
    Initialize agent state from user query.
    
    Args:
        user_query: User's natural language query
        **kwargs: Optional parameters (currency_pair, amount, etc.)
    
    Returns:
        Initialized AgentState
    """
    state: AgentState = {
        "correlation_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow(),
        "user_query": user_query,
        
        # Will be filled by NLU
        "currency_pair": kwargs.get("currency_pair", ""),
        "base_currency": kwargs.get("base_currency", ""),
        "quote_currency": kwargs.get("quote_currency", ""),
        "amount": kwargs.get("amount"),
        
        # Default runtime parameters
        "risk_tolerance": kwargs.get("risk_tolerance", "moderate"),
        "urgency": kwargs.get("urgency", "normal"),
        "timeframe": kwargs.get("timeframe", "1_day"),
        
        # Agent statuses
        "market_data_status": "pending",
        "market_data_error": None,
        "intelligence_status": "pending",
        "intelligence_error": None,
        "prediction_status": "pending",
        "prediction_error": None,
        "decision_status": "pending",
        "decision_error": None,
        
        # Agent outputs
        "market_snapshot": None,
        "intelligence_report": None,
        "price_forecast": None,
        "recommendation": None,
        
        # Conversation
        "conversation_history": [],
        "clarifications_needed": [],
        
        # System
        "warnings": [],
        "errors": [],
        "processing_stage": "nlu",
        "agent_metrics": {}
    }
    
    return state


def validate_state(state: AgentState) -> tuple[bool, List[str]]:
    """
    Validate agent state.
    
    Args:
        state: Agent state to validate
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required fields
    if not state.get("user_query"):
        errors.append("Missing user_query")
    
    if not state.get("correlation_id"):
        errors.append("Missing correlation_id")
    
    # Validate runtime parameters
    valid_risk = ["conservative", "moderate", "aggressive"]
    if state.get("risk_tolerance") not in valid_risk:
        errors.append(f"Invalid risk_tolerance: {state.get('risk_tolerance')}")
    
    valid_urgency = ["urgent", "normal", "flexible"]
    if state.get("urgency") not in valid_urgency:
        errors.append(f"Invalid urgency: {state.get('urgency')}")
    
    valid_timeframe = ["immediate", "1_day", "1_week", "1_month"]
    if state.get("timeframe") not in valid_timeframe:
        errors.append(f"Invalid timeframe: {state.get('timeframe')}")
    
    return len(errors) == 0, errors


def add_conversation_turn(
    state: AgentState,
    role: str,
    message: str
) -> AgentState:
    """
    Add a conversation turn to state.
    
    Args:
        state: Current state
        role: 'user' or 'assistant'
        message: Message content
    
    Returns:
        Updated state
    """
    state["conversation_history"].append({
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    })
    return state


def add_warning(state: AgentState, warning: str) -> AgentState:
    """Add a warning message to state."""
    state["warnings"].append(warning)
    return state


def add_error(state: AgentState, error: str) -> AgentState:
    """Add an error message to state."""
    state["errors"].append(error)
    return state


def update_processing_stage(state: AgentState, stage: str) -> AgentState:
    """Update current processing stage."""
    state["processing_stage"] = stage
    return state


def record_agent_metric(
    state: AgentState,
    agent_name: str,
    execution_time_ms: int,
    status: str,
    error: Optional[str] = None
) -> AgentState:
    """
    Record agent execution metrics.
    
    Args:
        state: Current state
        agent_name: Name of agent
        execution_time_ms: Execution time in milliseconds
        status: 'success', 'error', or 'partial'
        error: Optional error message
    
    Returns:
        Updated state
    """
    state["agent_metrics"][agent_name] = {
        "execution_time_ms": execution_time_ms,
        "status": status,
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }
    return state
```

### Step 2: Create Graph Skeleton (src/agentic/graph.py)

Basic LangGraph structure:

```python
"""LangGraph workflow definition."""
from typing import Literal
from langgraph.graph import StateGraph, END
from src.agentic.state import AgentState
import logging

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
    
    # Add nodes (will be implemented in later phases)
    workflow.add_node("nlu", nlu_node)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("market_intelligence", market_intelligence_node)
    workflow.add_node("price_prediction", price_prediction_node)
    workflow.add_node("decision_engine", decision_engine_node)
    workflow.add_node("response", response_node)
    
    # Set entry point
    workflow.set_entry_point("nlu")
    
    # Add edges
    workflow.add_edge("nlu", "market_data")
    workflow.add_edge("nlu", "market_intelligence")
    workflow.add_edge("market_data", "price_prediction")
    workflow.add_edge("market_intelligence", "price_prediction")
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
    state["market_data_status"] = "success"
    return state


def market_intelligence_node(state: AgentState) -> AgentState:
    """Market Intelligence agent node (placeholder)."""
    logger.info("Market Intelligence node called")
    state["intelligence_status"] = "success"
    return state


def price_prediction_node(state: AgentState) -> AgentState:
    """Price Prediction agent node (placeholder)."""
    logger.info("Price Prediction node called")
    state["prediction_status"] = "success"
    return state


def decision_engine_node(state: AgentState) -> AgentState:
    """Decision Engine agent node (placeholder)."""
    logger.info("Decision Engine node called")
    state["decision_status"] = "success"
    return state


def response_node(state: AgentState) -> AgentState:
    """Response generation node (placeholder)."""
    logger.info("Response node called")
    state["processing_stage"] = "complete"
    return state
```

### Step 3: Write Tests

**tests/unit/test_state.py**:

```python
"""Tests for state management."""
import pytest
from src.agentic.state import (
    AgentState,
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
```

**tests/unit/test_graph.py**:

```python
"""Tests for graph structure."""
import pytest
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state


def test_graph_creation():
    """Test graph can be created."""
    graph = create_graph()
    assert graph is not None


def test_graph_execution():
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
```

### Step 4: Integration Test

Test full state flow:

```python
# tests/integration/test_state_flow.py
def test_complete_state_flow():
    """Test complete state flow through graph."""
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
```

## Files to Create

1. `src/agentic/state.py` - Complete state schema and utilities
2. `src/agentic/graph.py` - Graph skeleton with placeholder nodes
3. `tests/unit/test_state.py` - State management tests
4. `tests/unit/test_graph.py` - Graph structure tests
5. `tests/integration/test_state_flow.py` - Integration test

## Success Criteria

- AgentState TypedDict defines all fields
- initialize_state() creates valid initial state
- validate_state() catches invalid parameters
- Helper functions update state correctly
- Graph can be created and compiled
- Graph executes with placeholder nodes
- All tests pass
- State flows correctly through processing stages

### To-dos

- [ ] Create complete directory structure for all project modules
- [ ] Extend config.yaml with app, database, cache, api, agents, and logging sections
- [ ] Create .env.example with all required environment variables
- [ ] Implement src/utils/errors.py with custom exception hierarchy
- [ ] Implement src/utils/logging.py with structured logging and correlation IDs
- [ ] Implement src/utils/validation.py with input validation functions
- [ ] Implement src/config.py with YAML and environment variable loading
- [ ] Write comprehensive tests for config, validation, and error modules
- [ ] Update pyproject.toml to remove unnecessary dependencies
- [ ] Run all tests and validate configuration loading works correctly