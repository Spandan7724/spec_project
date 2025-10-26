"""LangGraph state schema and management."""
from typing import TypedDict, Optional, List, Dict, Any, Annotated
from datetime import datetime
import uuid
from operator import add


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
    conversation_history: Annotated[List[Dict[str, str]], add]
    clarifications_needed: Annotated[List[str], add]
    
    # System
    warnings: Annotated[List[str], add]
    errors: Annotated[List[str], add]
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

