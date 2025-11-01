from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ConversationInput(BaseModel):
    user_input: str
    session_id: Optional[str] = None


class AnalysisRequest(BaseModel):
    # Session & correlation
    session_id: str
    correlation_id: str

    # Core params
    currency_pair: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    amount: float
    risk_tolerance: str  # conservative|moderate|aggressive
    urgency: str  # urgent|normal|flexible

    # Legacy categorical timeframe
    timeframe: Optional[str] = Field(
        default=None, description="immediate|1_day|1_week|1_month"
    )

    # Flexible timeframe (free text)
    timeframe_text: Optional[str] = Field(
        default=None, description="e.g., 'in 10 days', 'by 2025-11-15', '3-5 days', 'in 12 hours'"
    )

    # Canonical timeframe (optional override if provided)
    timeframe_days: Optional[int] = None
    timeframe_mode: Optional[str] = None  # immediate|deadline|duration
    deadline_utc: Optional[str] = None
    window_days: Optional[Dict[str, int]] = None
    time_unit: Optional[str] = None  # hours|days
    timeframe_hours: Optional[int] = None

