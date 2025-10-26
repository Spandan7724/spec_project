from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConversationState(Enum):
    """Current state of conversation."""

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
    """Parameters extracted from user input."""

    currency_pair: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    amount: Optional[float] = None
    risk_tolerance: Optional[str] = None  # conservative|moderate|aggressive
    urgency: Optional[str] = None  # urgent|normal|flexible
    timeframe: Optional[str] = None  # immediate|1_day|1_week|1_month
    timeframe_days: Optional[int] = None

    def is_complete(self) -> bool:
        """Check if all required parameters are set."""

        return all(
            [
                self.currency_pair,
                self.base_currency,
                self.quote_currency,
                self.amount is not None,
                self.risk_tolerance,
                self.urgency,
                self.timeframe,
                self.timeframe_days is not None,
            ]
        )

    def missing_parameters(self) -> List[str]:
        """Get list of missing parameter names."""

        missing: List[str] = []
        if not self.currency_pair:
            missing.append("currency_pair")
        if self.amount is None:
            missing.append("amount")
        if not self.risk_tolerance:
            missing.append("risk_tolerance")
        if not self.urgency:
            missing.append("urgency")
        # Timeframe: accept either categorical timeframe or explicit days
        if self.timeframe is None and self.timeframe_days is None:
            missing.append("timeframe")
        # If categorical timeframe was provided but numeric days not derived yet
        if self.timeframe is not None and self.timeframe_days is None:
            missing.append("timeframe_days")
        return missing


@dataclass
class ConversationSession:
    """Session state for one conversion request."""

    session_id: str
    state: ConversationState
    parameters: ExtractedParameters
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history."""

        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        self.last_updated = datetime.now()


@dataclass
class SupervisorRequest:
    """Input to supervisor."""

    user_input: str
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class SupervisorResponse:
    """Output from supervisor."""

    session_id: str
    state: ConversationState
    message: str  # Response to user
    requires_input: bool  # Does supervisor need more input?
    parameters: Optional[ExtractedParameters] = None
    recommendation: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
