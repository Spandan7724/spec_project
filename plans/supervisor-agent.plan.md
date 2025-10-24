# Supervisor Agent Implementation Plan

## Overview

The Supervisor Agent is the orchestrator that manages the entire user interaction and agent workflow. It conducts an intelligent, conversational dialogue to collect all required parameters, dispatches agents in the correct sequence, handles partial failures gracefully, and generates the final recommendation.

## Design Decisions (Locked In)

1. **Smart Extraction**: Extract parameters from initial query when clear, ask for missing ones
2. **Session-Based Memory**: Conversation context within one conversion request only
3. **Professional Tone**: Formal, clear communication style
4. **Review Step**: Show all collected parameters, allow changes before running agents
5. **In-Memory State**: Simple session management (MVP)
6. **No Multilingual**: English only for MVP

## Conversation Flow Examples

### Full Initial Query

```
User: "I need to convert 5000 USD to EUR urgently"

Agent: "I'll help you convert 5000 USD to EUR.

I've noted:
✓ Amount: 5000 USD
✓ Target: EUR
✓ Urgency: Urgent

I need a few more details:

1️⃣ What is your risk tolerance?
   - conservative (minimize risk, prefer safety)
   - moderate (balanced approach)
   - aggressive (willing to accept higher risk for potential gains)

Please specify: conservative, moderate, or aggressive"

User: "moderate"

Agent: "Risk tolerance set to moderate.

2️⃣ What is your timeframe for this conversion?
   - immediate (today/within 24 hours)
   - 1_day (1-2 days)
   - 1_week (up to 7 days)
   - 1_month (up to 30 days)

Please specify: immediate, 1_day, 1_week, or 1_month"

User: "immediate"

Agent: "Thank you. Let me confirm all parameters:

✓ Currency pair: USD/EUR
✓ Amount: 5000 USD
✓ Risk tolerance: Moderate
✓ Urgency: Urgent
✓ Timeframe: Immediate

Is this correct?
- Type 'yes' to proceed with analysis
- Type 'change <parameter>' to modify
- Type 'restart' to start over"

User: "yes"

Agent: [Runs agents and returns recommendation]
```

## Architecture

### Components

```
src/supervisor/
├── __init__.py
├── models.py                      # Data contracts
├── nlu_extractor.py               # Parameter extraction from natural language
├── conversation_manager.py        # Multi-turn conversation state management
├── agent_orchestrator.py          # Agent dispatch and coordination
├── response_formatter.py          # Format final recommendation
└── config.py                      # Configuration

src/agentic/
└── supervisor_node.py             # NEW: Supervisor as entry point

tests/supervisor/
├── test_nlu_extractor.py
├── test_conversation_manager.py
├── test_agent_orchestrator.py
└── test_integration.py
```

## Data Contracts

### File: `src/supervisor/models.py`

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ConversationState(Enum):
    """Current state of conversation"""
    INITIAL = "initial"
    COLLECTING_CURRENCY_PAIR = "collecting_currency_pair"
    COLLECTING_AMOUNT = "collecting_amount"
    COLLECTING_RISK = "collecting_risk"
    COLLECTING_URGENCY = "collecting_urgency"
    COLLECTING_TIMEFRAME = "collecting_timeframe"
    CONFIRMING = "confirming"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ExtractedParameters:
    """Parameters extracted from user input"""
    currency_pair: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    amount: Optional[float] = None
    risk_tolerance: Optional[str] = None  # conservative|moderate|aggressive
    urgency: Optional[str] = None  # urgent|normal|flexible
    timeframe: Optional[str] = None  # immediate|1_day|1_week|1_month
    timeframe_days: Optional[int] = None
    
    def is_complete(self) -> bool:
        """Check if all required parameters are set"""
        return all([
            self.currency_pair,
            self.base_currency,
            self.quote_currency,
            self.amount is not None,
            self.risk_tolerance,
            self.urgency,
            self.timeframe,
            self.timeframe_days is not None
        ])
    
    def missing_parameters(self) -> List[str]:
        """Get list of missing parameters"""
        missing = []
        if not self.currency_pair:
            missing.append("currency_pair")
        if self.amount is None:
            missing.append("amount")
        if not self.risk_tolerance:
            missing.append("risk_tolerance")
        if not self.urgency:
            missing.append("urgency")
        if not self.timeframe:
            missing.append("timeframe")
        return missing

@dataclass
class ConversationSession:
    """Session state for one conversion request"""
    session_id: str
    state: ConversationState
    parameters: ExtractedParameters
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_updated = datetime.now()

@dataclass
class SupervisorRequest:
    """Input to supervisor"""
    user_input: str
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class SupervisorResponse:
    """Output from supervisor"""
    session_id: str
    state: ConversationState
    message: str  # Response to user
    requires_input: bool  # Does supervisor need more input?
    parameters: Optional[ExtractedParameters] = None
    recommendation: Optional[Dict[str, Any]] = None  # Final recommendation if completed
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
```

## NLU Parameter Extractor

### File: `src/supervisor/nlu_extractor.py`

```python
import re
import logging
from typing import Optional, Tuple
from .models import ExtractedParameters

logger = logging.getLogger(__name__)

class NLUExtractor:
    """Extract parameters from natural language input"""
    
    # Common currency codes
    CURRENCY_CODES = {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "INR", "BRL", "RUB", "MXN", "SGD", "HKD", "SEK",
        "NOK", "DKK", "PLN", "TRY", "ZAR", "KRW", "THB", "IDR"
    }
    
    # Currency name mappings
    CURRENCY_NAMES = {
        "dollar": "USD", "dollars": "USD", "usd": "USD",
        "euro": "EUR", "euros": "EUR", "eur": "EUR",
        "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "sterling": "GBP",
        "yen": "JPY", "jpy": "JPY",
        "franc": "CHF", "chf": "CHF", "swiss franc": "CHF",
        "canadian dollar": "CAD", "cad": "CAD",
        "australian dollar": "AUD", "aud": "AUD",
        "yuan": "CNY", "renminbi": "CNY", "cny": "CNY",
        "rupee": "INR", "rupees": "INR", "inr": "INR"
    }
    
    # Risk tolerance keywords
    RISK_KEYWORDS = {
        "conservative": ["conservative", "safe", "careful", "cautious", "low risk", "minimal risk"],
        "moderate": ["moderate", "balanced", "medium", "normal", "average"],
        "aggressive": ["aggressive", "risky", "high risk", "bold", "adventurous"]
    }
    
    # Urgency keywords
    URGENCY_KEYWORDS = {
        "urgent": ["urgent", "asap", "immediately", "now", "today", "hurry", "quick", "fast"],
        "normal": ["normal", "soon", "regular", "standard", "few days"],
        "flexible": ["flexible", "patient", "wait", "whenever", "no rush", "take time"]
    }
    
    # Timeframe keywords
    TIMEFRAME_KEYWORDS = {
        "immediate": ["immediate", "now", "today", "asap", "right now"],
        "1_day": ["tomorrow", "1 day", "one day", "24 hours", "next day"],
        "1_week": ["week", "7 days", "this week", "next week", "few days"],
        "1_month": ["month", "30 days", "4 weeks", "next month"]
    }
    
    def extract(self, text: str) -> ExtractedParameters:
        """Extract all possible parameters from text"""
        
        text_lower = text.lower()
        params = ExtractedParameters()
        
        # Extract currency pair
        base, quote = self._extract_currency_pair(text)
        if base and quote:
            params.base_currency = base
            params.quote_currency = quote
            params.currency_pair = f"{base}/{quote}"
        
        # Extract amount
        params.amount = self._extract_amount(text)
        
        # Extract risk tolerance
        params.risk_tolerance = self._extract_risk_tolerance(text_lower)
        
        # Extract urgency
        params.urgency = self._extract_urgency(text_lower)
        
        # Extract timeframe
        params.timeframe = self._extract_timeframe(text_lower)
        if params.timeframe:
            params.timeframe_days = self._timeframe_to_days(params.timeframe)
        
        return params
    
    def _extract_currency_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract currency pair (base and quote)"""
        
        # Pattern 1: "USD/EUR" or "USD-EUR" or "USDEUR"
        pattern1 = r'\b([A-Z]{3})\s*[/\-]?\s*([A-Z]{3})\b'
        match = re.search(pattern1, text)
        if match:
            base, quote = match.groups()
            if base in self.CURRENCY_CODES and quote in self.CURRENCY_CODES:
                return base, quote
        
        # Pattern 2: "USD to EUR" or "USD into EUR"
        pattern2 = r'\b([A-Z]{3})\s+(?:to|into|→)\s+([A-Z]{3})\b'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            base, quote = match.groups()
            base, quote = base.upper(), quote.upper()
            if base in self.CURRENCY_CODES and quote in self.CURRENCY_CODES:
                return base, quote
        
        # Pattern 3: Currency names (e.g., "dollars to euros")
        text_lower = text.lower()
        found_currencies = []
        
        for name, code in self.CURRENCY_NAMES.items():
            if name in text_lower:
                found_currencies.append((text_lower.index(name), code))
        
        if len(found_currencies) >= 2:
            # Sort by position, first is base, second is quote
            found_currencies.sort()
            return found_currencies[0][1], found_currencies[1][1]
        
        return None, None
    
    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract amount from text"""
        
        # Pattern 1: "5000" or "5,000" or "5000.50"
        pattern = r'\b([\d,]+(?:\.\d+)?)\b'
        
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Remove commas and convert
                amount = float(match.replace(',', ''))
                if amount > 0:
                    return amount
            except ValueError:
                continue
        
        return None
    
    def _extract_risk_tolerance(self, text: str) -> Optional[str]:
        """Extract risk tolerance"""
        
        for level, keywords in self.RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return level
        
        return None
    
    def _extract_urgency(self, text: str) -> Optional[str]:
        """Extract urgency"""
        
        for level, keywords in self.URGENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return level
        
        return None
    
    def _extract_timeframe(self, text: str) -> Optional[str]:
        """Extract timeframe"""
        
        for timeframe, keywords in self.TIMEFRAME_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return timeframe
        
        return None
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convert timeframe to days"""
        mapping = {
            "immediate": 1,
            "1_day": 1,
            "1_week": 7,
            "1_month": 30
        }
        return mapping.get(timeframe, 7)
    
    def validate_currency_pair(self, base: str, quote: str) -> Tuple[bool, Optional[str]]:
        """Validate currency pair"""
        
        if base not in self.CURRENCY_CODES:
            return False, f"Invalid base currency: {base}. Must be a valid ISO currency code."
        
        if quote not in self.CURRENCY_CODES:
            return False, f"Invalid quote currency: {quote}. Must be a valid ISO currency code."
        
        if base == quote:
            return False, f"Base and quote currencies cannot be the same: {base}"
        
        return True, None
```

## Conversation Manager

### File: `src/supervisor/conversation_manager.py`

```python
import uuid
import logging
from typing import Optional, Dict
from .models import (
    ConversationSession, ConversationState, ExtractedParameters,
    SupervisorRequest, SupervisorResponse
)
from .nlu_extractor import NLUExtractor

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manage multi-turn conversation flow"""
    
    def __init__(self):
        self.extractor = NLUExtractor()
        self.sessions: Dict[str, ConversationSession] = {}
    
    def process_input(self, request: SupervisorRequest) -> SupervisorResponse:
        """Process user input and return appropriate response"""
        
        # Get or create session
        session = self._get_or_create_session(request.session_id)
        
        # Add user message to history
        session.add_message("user", request.user_input)
        
        # Handle based on current state
        if session.state == ConversationState.INITIAL:
            return self._handle_initial(session, request.user_input)
        
        elif session.state == ConversationState.CONFIRMING:
            return self._handle_confirmation(session, request.user_input)
        
        elif session.state in [
            ConversationState.COLLECTING_CURRENCY_PAIR,
            ConversationState.COLLECTING_AMOUNT,
            ConversationState.COLLECTING_RISK,
            ConversationState.COLLECTING_URGENCY,
            ConversationState.COLLECTING_TIMEFRAME
        ]:
            return self._handle_collection(session, request.user_input)
        
        else:
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message="Invalid session state. Please type 'restart' to begin again.",
                requires_input=True
            )
    
    def _get_or_create_session(self, session_id: Optional[str]) -> ConversationSession:
        """Get existing session or create new one"""
        
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_id = session_id or str(uuid.uuid4())
        session = ConversationSession(
            session_id=new_id,
            state=ConversationState.INITIAL,
            parameters=ExtractedParameters()
        )
        self.sessions[new_id] = session
        return session
    
    def _handle_initial(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Handle initial user query"""
        
        # Extract parameters from initial input
        extracted = self.extractor.extract(user_input)
        session.parameters = extracted
        
        # Build response showing what was extracted
        message_parts = []
        
        if extracted.currency_pair:
            message_parts.append(f"I'll help you convert {extracted.currency_pair}.")
            message_parts.append("\nI've noted:")
            
            if extracted.amount:
                message_parts.append(f"✓ Amount: {extracted.amount:,.2f} {extracted.base_currency}")
            else:
                message_parts.append(f"✓ Currency pair: {extracted.currency_pair}")
            
            if extracted.urgency:
                message_parts.append(f"✓ Urgency: {extracted.urgency.capitalize()}")
            
            if extracted.risk_tolerance:
                message_parts.append(f"✓ Risk tolerance: {extracted.risk_tolerance.capitalize()}")
            
            if extracted.timeframe:
                message_parts.append(f"✓ Timeframe: {extracted.timeframe}")
        else:
            message_parts.append("I'll help you with your currency conversion.")
        
        # Determine next step
        missing = extracted.missing_parameters()
        
        if not missing:
            # All parameters collected, go to confirmation
            return self._move_to_confirmation(session)
        
        # Ask for next missing parameter
        next_param = missing[0]
        return self._ask_for_parameter(session, next_param, message_parts)
    
    def _handle_collection(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Handle collection of specific parameter"""
        
        # Determine which parameter we're collecting
        param_state_map = {
            ConversationState.COLLECTING_CURRENCY_PAIR: "currency_pair",
            ConversationState.COLLECTING_AMOUNT: "amount",
            ConversationState.COLLECTING_RISK: "risk_tolerance",
            ConversationState.COLLECTING_URGENCY: "urgency",
            ConversationState.COLLECTING_TIMEFRAME: "timeframe"
        }
        
        param_name = param_state_map[session.state]
        
        # Extract the parameter
        extracted = self.extractor.extract(user_input)
        
        # Update session parameters
        if param_name == "currency_pair":
            if extracted.base_currency and extracted.quote_currency:
                session.parameters.base_currency = extracted.base_currency
                session.parameters.quote_currency = extracted.quote_currency
                session.parameters.currency_pair = extracted.currency_pair
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="I couldn't understand the currency pair. Please provide it in format like 'USD/EUR' or 'USD to EUR'",
                    requires_input=True
                )
        
        elif param_name == "amount":
            if extracted.amount:
                session.parameters.amount = extracted.amount
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Please provide a valid amount (e.g., 5000, 10000.50)",
                    requires_input=True
                )
        
        elif param_name == "risk_tolerance":
            risk = extracted.risk_tolerance or self._parse_risk_direct(user_input.lower())
            if risk:
                session.parameters.risk_tolerance = risk
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Please specify: conservative, moderate, or aggressive",
                    requires_input=True
                )
        
        elif param_name == "urgency":
            urgency = extracted.urgency or self._parse_urgency_direct(user_input.lower())
            if urgency:
                session.parameters.urgency = urgency
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Please specify: urgent, normal, or flexible",
                    requires_input=True
                )
        
        elif param_name == "timeframe":
            timeframe = extracted.timeframe or self._parse_timeframe_direct(user_input.lower())
            if timeframe:
                session.parameters.timeframe = timeframe
                session.parameters.timeframe_days = self.extractor._timeframe_to_days(timeframe)
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Please specify: immediate, 1_day, 1_week, or 1_month",
                    requires_input=True
                )
        
        # Confirm parameter set
        confirmation = f"{param_name.replace('_', ' ').title()} set to {getattr(session.parameters, param_name)}."
        
        # Check if all parameters collected
        missing = session.parameters.missing_parameters()
        
        if not missing:
            return self._move_to_confirmation(session, prefix=confirmation)
        
        # Ask for next parameter
        next_param = missing[0]
        return self._ask_for_parameter(session, next_param, [confirmation])
    
    def _handle_confirmation(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Handle confirmation/change requests"""
        
        user_input_lower = user_input.lower().strip()
        
        # Check for confirmation
        if user_input_lower in ["yes", "y", "confirm", "proceed", "go", "ok"]:
            session.state = ConversationState.PROCESSING
            
            return SupervisorResponse(
                session_id=session.session_id,
                state=ConversationState.PROCESSING,
                message="Analyzing market conditions and generating recommendation...",
                requires_input=False,
                parameters=session.parameters
            )
        
        # Check for restart
        if user_input_lower in ["restart", "start over", "reset"]:
            session.parameters = ExtractedParameters()
            session.state = ConversationState.INITIAL
            
            return SupervisorResponse(
                session_id=session.session_id,
                state=ConversationState.INITIAL,
                message="Let's start over. What currency conversion do you need?",
                requires_input=True
            )
        
        # Check for change request
        if user_input_lower.startswith("change"):
            return self._handle_parameter_change(session, user_input)
        
        # Didn't understand
        return SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message="Please type 'yes' to proceed, 'change <parameter>' to modify, or 'restart' to start over.",
            requires_input=True
        )
    
    def _handle_parameter_change(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Handle parameter change request"""
        
        # Parse change command: "change risk to aggressive"
        parts = user_input.lower().split()
        
        if len(parts) < 2:
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message="Please specify what to change (e.g., 'change risk to aggressive', 'change amount to 3000')",
                requires_input=True
            )
        
        param_name = parts[1]
        new_value_parts = parts[3:] if len(parts) > 3 and parts[2] == "to" else parts[2:]
        new_value = " ".join(new_value_parts)
        
        # Map parameter names
        param_map = {
            "risk": "risk_tolerance",
            "urgency": "urgency",
            "timeframe": "timeframe",
            "amount": "amount"
        }
        
        actual_param = param_map.get(param_name)
        
        if not actual_param:
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message=f"Unknown parameter: {param_name}. You can change: risk, urgency, timeframe, amount",
                requires_input=True
            )
        
        # Update parameter
        if actual_param == "amount":
            try:
                amount = float(new_value.replace(',', ''))
                session.parameters.amount = amount
            except ValueError:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message=f"Invalid amount: {new_value}",
                    requires_input=True
                )
        
        elif actual_param == "risk_tolerance":
            risk = self._parse_risk_direct(new_value)
            if risk:
                session.parameters.risk_tolerance = risk
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Risk tolerance must be: conservative, moderate, or aggressive",
                    requires_input=True
                )
        
        elif actual_param == "urgency":
            urgency = self._parse_urgency_direct(new_value)
            if urgency:
                session.parameters.urgency = urgency
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Urgency must be: urgent, normal, or flexible",
                    requires_input=True
                )
        
        elif actual_param == "timeframe":
            timeframe = self._parse_timeframe_direct(new_value)
            if timeframe:
                session.parameters.timeframe = timeframe
                session.parameters.timeframe_days = self.extractor._timeframe_to_days(timeframe)
            else:
                return SupervisorResponse(
                    session_id=session.session_id,
                    state=session.state,
                    message="Timeframe must be: immediate, 1_day, 1_week, or 1_month",
                    requires_input=True
                )
        
        # Show updated parameters
        return self._move_to_confirmation(
            session,
            prefix=f"Updated: {actual_param.replace('_', ' ').title()} changed to {getattr(session.parameters, actual_param)}."
        )
    
    def _move_to_confirmation(self, session: ConversationSession, prefix: str = "") -> SupervisorResponse:
        """Move to confirmation state and show all parameters"""
        
        session.state = ConversationState.CONFIRMING
        
        params = session.parameters
        
        message_parts = []
        if prefix:
            message_parts.append(prefix)
            message_parts.append("")
        
        message_parts.append("Thank you. Let me confirm all parameters:")
        message_parts.append("")
        message_parts.append(f"✓ Currency pair: {params.currency_pair}")
        message_parts.append(f"✓ Amount: {params.amount:,.2f} {params.base_currency}")
        message_parts.append(f"✓ Risk tolerance: {params.risk_tolerance.capitalize()}")
        message_parts.append(f"✓ Urgency: {params.urgency.capitalize()}")
        message_parts.append(f"✓ Timeframe: {params.timeframe.replace('_', ' ').title()} ({params.timeframe_days} days)")
        message_parts.append("")
        message_parts.append("Is this correct?")
        message_parts.append("- Type 'yes' to proceed with analysis")
        message_parts.append("- Type 'change <parameter>' to modify (e.g., 'change risk to conservative')")
        message_parts.append("- Type 'restart' to start over")
        
        message = "\n".join(message_parts)
        
        response = SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message=message,
            requires_input=True
        )
        
        session.add_message("assistant", message)
        return response
    
    def _ask_for_parameter(
        self, 
        session: ConversationSession, 
        param_name: str,
        prefix_parts: list = None
    ) -> SupervisorResponse:
        """Ask user for specific parameter"""
        
        prefix_parts = prefix_parts or []
        
        if prefix_parts:
            prefix_parts.append("")
            prefix_parts.append("I need a few more details:")
            prefix_parts.append("")
        
        questions = {
            "currency_pair": {
                "state": ConversationState.COLLECTING_CURRENCY_PAIR,
                "message": [
                    "What currency pair would you like to convert?",
                    "",
                    "Please provide in format: USD/EUR or 'USD to EUR'"
                ]
            },
            "amount": {
                "state": ConversationState.COLLECTING_AMOUNT,
                "message": [
                    "What amount do you want to convert?",
                    "",
                    "Please specify the amount (e.g., 5000, 10000.50):"
                ]
            },
            "risk_tolerance": {
                "state": ConversationState.COLLECTING_RISK,
                "message": [
                    "1️⃣ What is your risk tolerance?",
                    "   - conservative (minimize risk, prefer safety)",
                    "   - moderate (balanced approach)",
                    "   - aggressive (willing to accept higher risk for potential gains)",
                    "",
                    "Please specify: conservative, moderate, or aggressive"
                ]
            },
            "urgency": {
                "state": ConversationState.COLLECTING_URGENCY,
                "message": [
                    "2️⃣ How urgent is this conversion?",
                    "   - urgent (need to convert ASAP)",
                    "   - normal (within a few days is acceptable)",
                    "   - flexible (can wait for optimal timing)",
                    "",
                    "Please specify: urgent, normal, or flexible"
                ]
            },
            "timeframe": {
                "state": ConversationState.COLLECTING_TIMEFRAME,
                "message": [
                    "3️⃣ What is your timeframe for this conversion?",
                    "   - immediate (today/within 24 hours)",
                    "   - 1_day (1-2 days)",
                    "   - 1_week (up to 7 days)",
                    "   - 1_month (up to 30 days)",
                    "",
                    "Please specify: immediate, 1_day, 1_week, or 1_month"
                ]
            }
        }
        
        question_data = questions[param_name]
        session.state = question_data["state"]
        
        message_parts = prefix_parts + question_data["message"]
        message = "\n".join(message_parts)
        
        response = SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message=message,
            requires_input=True
        )
        
        session.add_message("assistant", message)
        return response
    
    # Helper parsers for direct input
    def _parse_risk_direct(self, text: str) -> Optional[str]:
        """Parse risk from direct response"""
        if "conservative" in text or "safe" in text:
            return "conservative"
        if "moderate" in text or "balanced" in text:
            return "moderate"
        if "aggressive" in text or "risky" in text:
            return "aggressive"
        return None
    
    def _parse_urgency_direct(self, text: str) -> Optional[str]:
        """Parse urgency from direct response"""
        if "urgent" in text or "asap" in text:
            return "urgent"
        if "normal" in text or "regular" in text:
            return "normal"
        if "flexible" in text or "patient" in text:
            return "flexible"
        return None
    
    def _parse_timeframe_direct(self, text: str) -> Optional[str]:
        """Parse timeframe from direct response"""
        text = text.replace("-", "_").replace(" ", "_")
        if "immediate" in text:
            return "immediate"
        if "1_day" in text or "1day" in text or "oneday" in text:
            return "1_day"
        if "1_week" in text or "1week" in text or "week" in text:
            return "1_week"
        if "1_month" in text or "1month" in text or "month" in text:
            return "1_month"
        return None
```

## Agent Orchestrator

### File: `src/supervisor/agent_orchestrator.py`

```python
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from src.agentic.state import AgentGraphState, AgentRequest
from src.agentic.nodes.market import MarketAnalysisAgent
from src.agentic.nodes.economic import EconomicAnalysisAgent
from src.agentic.nodes.prediction import PredictionAgent
from src.agentic.nodes.decision import DecisionEngineAgent

from .models import ExtractedParameters

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrate agent execution"""
    
    def __init__(self):
        self.market_agent = MarketAnalysisAgent()
        self.economic_agent = EconomicAnalysisAgent()
        self.prediction_agent = PredictionAgent()
        self.decision_agent = DecisionEngineAgent()
    
    async def run_analysis(self, parameters: ExtractedParameters, correlation_id: str) -> Dict[str, Any]:
        """Run all agents and generate recommendation"""
        
        logger.info(f"[{correlation_id}] Starting agent orchestration")
        
        # Build agent request
        agent_request = AgentRequest.from_payload({
            "currency_pair": parameters.currency_pair,
            "amount": parameters.amount,
            "risk_tolerance": parameters.risk_tolerance,
            "timeframe_days": parameters.timeframe_days
        })
        
        # Initialize state
        state = AgentGraphState(request=agent_request)
        state.meta.correlation_id = correlation_id
        
        warnings = []
        
        try:
            # Layer 1: Market + Economic (parallel)
            logger.info(f"[{correlation_id}] Running Layer 1: Market + Economic")
            
            market_task = self.market_agent(state)
            economic_task = self.economic_agent(state)
            
            state = await market_task
            state = await economic_task
            
            if state.market and state.market.errors:
                warnings.extend(state.market.errors)
            
            if state.economic and state.economic.errors:
                warnings.extend(state.economic.errors)
            
        except Exception as e:
            logger.error(f"[{correlation_id}] Layer 1 failed: {e}")
            warnings.append(f"Market/Economic analysis failed: {str(e)}")
        
        try:
            # Layer 2: Prediction
            logger.info(f"[{correlation_id}] Running Layer 2: Prediction")
            state = await self.prediction_agent(state)
            
            if state.market and state.market.errors:
                warnings.extend(state.market.errors)
        
        except Exception as e:
            logger.error(f"[{correlation_id}] Prediction failed: {e}")
            warnings.append(f"Price prediction failed: {str(e)}")
        
        try:
            # Layer 3: Decision
            logger.info(f"[{correlation_id}] Running Layer 3: Decision")
            state = await self.decision_agent(state)
            
            if state.decision and state.decision.warnings:
                warnings.extend(state.decision.warnings)
        
        except Exception as e:
            logger.error(f"[{correlation_id}] Decision failed: {e}")
            return {
                "status": "error",
                "error": f"Decision engine failed: {str(e)}",
                "warnings": warnings
            }
        
        # Extract recommendation
        if not state.decision:
            return {
                "status": "error",
                "error": "No decision generated",
                "warnings": warnings
            }
        
        recommendation = {
            "status": "success",
            "action": state.decision.action,
            "confidence": state.decision.confidence,
            "timeline": state.decision.timeline,
            "rationale": state.decision.rationale,
            "warnings": warnings,
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add optional fields
        if hasattr(state.decision, 'staged_plan') and state.decision.staged_plan:
            recommendation["staged_plan"] = state.decision.staged_plan
        
        if hasattr(state.decision, 'expected_rate'):
            recommendation["expected_outcome"] = {
                "expected_rate": state.decision.expected_rate
            }
        
        if hasattr(state.decision, 'risk_level'):
            recommendation["risk_summary"] = {
                "risk_level": state.decision.risk_level
            }
        
        if hasattr(state.decision, 'cost_estimate_bps'):
            recommendation["cost_estimate"] = {
                "total_bps": state.decision.cost_estimate_bps
            }
        
        logger.info(f"[{correlation_id}] Analysis complete: {recommendation['action']}")
        
        return recommendation
```

## Response Formatter

### File: `src/supervisor/response_formatter.py`

```python
from typing import Dict, Any

class ResponseFormatter:
    """Format final recommendation for user display"""
    
    def format_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Format recommendation as user-friendly text"""
        
        if recommendation.get("status") == "error":
            return self._format_error(recommendation)
        
        lines = []
        lines.append("━" * 60)
        lines.append("RECOMMENDATION FOR CURRENCY CONVERSION")
        lines.append("━" * 60)
        lines.append("")
        
        # Action
        action = recommendation["action"].replace("_", " ").upper()
        lines.append(f"ACTION: {action}")
        
        # Confidence
        confidence = recommendation["confidence"]
        conf_level = "High" if confidence > 0.7 else "Moderate" if confidence > 0.4 else "Low"
        lines.append(f"CONFIDENCE: {confidence:.2f} ({conf_level})")
        
        # Timeline
        lines.append(f"TIMELINE: {recommendation['timeline']}")
        lines.append("")
        
        # Staged plan (if applicable)
        if "staged_plan" in recommendation:
            lines.append("STAGED CONVERSION PLAN:")
            plan = recommendation["staged_plan"]
            for tranche in plan["tranches"]:
                lines.append(f"  • Tranche {tranche['number']}: {tranche['percentage']:.0f}% on Day {tranche['execute_day']}")
            lines.append("")
        
        # Expected outcome (if available)
        if "expected_outcome" in recommendation:
            lines.append("EXPECTED OUTCOME:")
            outcome = recommendation["expected_outcome"]
            if "expected_rate" in outcome:
                lines.append(f"  • Expected rate: {outcome['expected_rate']:.4f}")
            lines.append("")
        
        # Rationale
        if recommendation["rationale"]:
            lines.append("RATIONALE:")
            for i, reason in enumerate(recommendation["rationale"], 1):
                lines.append(f"  {i}. {reason}")
            lines.append("")
        
        # Risk (if available)
        if "risk_summary" in recommendation:
            risk = recommendation["risk_summary"]
            lines.append(f"RISK ASSESSMENT: {risk['risk_level'].title()}")
            lines.append("")
        
        # Costs (if available)
        if "cost_estimate" in recommendation:
            cost = recommendation["cost_estimate"]
            lines.append(f"ESTIMATED COSTS: {cost['total_bps']:.1f} bps")
            lines.append("")
        
        # Warnings
        if recommendation.get("warnings"):
            lines.append("WARNINGS:")
            for warning in recommendation["warnings"]:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        lines.append("Would you like to:")
        lines.append("  - Execute this recommendation")
        lines.append("  - Get alternative scenarios")
        lines.append("  - Start a new analysis")
        
        return "\n".join(lines)
    
    def _format_error(self, recommendation: Dict[str, Any]) -> str:
        """Format error response"""
        
        lines = []
        lines.append("━" * 60)
        lines.append("ERROR GENERATING RECOMMENDATION")
        lines.append("━" * 60)
        lines.append("")
        lines.append(f"Error: {recommendation.get('error', 'Unknown error')}")
        lines.append("")
        
        if recommendation.get("warnings"):
            lines.append("Additional details:")
            for warning in recommendation["warnings"]:
                lines.append(f"  • {warning}")
        
        lines.append("")
        lines.append("Please try again or contact support if the issue persists.")
        
        return "\n".join(lines)
```

## Implementation Todos

- [ ] Create data contracts in `src/supervisor/models.py`
- [ ] Implement NLU extractor in `src/supervisor/nlu_extractor.py`
- [ ] Implement conversation manager in `src/supervisor/conversation_manager.py`
- [ ] Implement agent orchestrator in `src/supervisor/agent_orchestrator.py`
- [ ] Implement response formatter in `src/supervisor/response_formatter.py`
- [ ] Create supervisor config in `src/supervisor/config.py`
- [ ] Write unit tests for NLU extraction
- [ ] Write unit tests for conversation flow
- [ ] Write integration tests for full conversation
- [ ] Add session timeout/cleanup mechanism
- [ ] Document conversation flows and examples