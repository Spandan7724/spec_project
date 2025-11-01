from __future__ import annotations

from typing import Any, Dict, Optional, List

from pydantic import BaseModel


class ConversationOutput(BaseModel):
    session_id: str
    state: str
    message: str
    requires_input: bool
    parameters: Optional[Dict[str, Any]] = None


class AnalysisStartResponse(BaseModel):
    correlation_id: str
    status: str


class AnalysisStatus(BaseModel):
    status: str
    progress: int
    message: str


class RiskBreakdownResponse(BaseModel):
    """Risk breakdown for visualization."""
    correlation_id: str
    risk_level: Optional[str] = None
    event_risk: Optional[str] = None
    volatility_risk: Optional[str] = None
    liquidity_risk: Optional[str] = None
    market_regime: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class CostBreakdownResponse(BaseModel):
    """Cost breakdown for visualization."""
    correlation_id: str
    total_cost_bps: Optional[float] = None
    total_cost_absolute: Optional[float] = None
    spread_cost_bps: Optional[float] = None
    fee_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    cost_percentage: Optional[float] = None
    breakdown: Optional[Dict[str, Any]] = None


class TimelinePoint(BaseModel):
    """Single point in timeline."""
    index: int
    day: int
    amount: float
    percentage: float
    note: str


class TimelineDataResponse(BaseModel):
    """Timeline data for visualization."""
    correlation_id: str
    action: Optional[str] = None
    timeline: Optional[str] = None
    timeline_points: List[TimelinePoint] = []
    expected_outcome: Optional[Dict[str, Any]] = None


class PredictionChartPoint(BaseModel):
    """Single prediction data point for charting."""
    horizon: Any  # Could be int or string
    mean_rate: float
    mean_change_pct: float
    p10: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    direction_probability: Optional[float] = None


class PredictionChartResponse(BaseModel):
    """Prediction chart data."""
    correlation_id: str
    currency_pair: str
    latest_close: float
    chart_data: List[PredictionChartPoint]
    confidence: Optional[float] = None


class NewsArticle(BaseModel):
    """News article evidence."""
    title: Optional[str] = None
    url: Optional[str] = None
    sentiment: Optional[str] = None
    relevance: Optional[float] = None


class CalendarEvent(BaseModel):
    """Calendar event."""
    when_utc: str
    currency: str
    event: str
    importance: Optional[str] = None
    source_url: Optional[str] = None


class EvidenceResponse(BaseModel):
    """Supporting evidence for analysis."""
    correlation_id: str
    news: Dict[str, Any]
    calendar: Dict[str, Any]
    market: Dict[str, Any]
    policy_bias: Optional[Dict[str, Any]] = None

