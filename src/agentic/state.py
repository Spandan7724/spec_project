"""Shared data models and helpers for the agentic LangGraph workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

_VALID_RISK_LEVELS = {"low", "moderate", "high"}
_DEFAULT_TIMEFRAME_DAYS = 7


def _normalize_currency_pair(pair: str) -> tuple[str, str, str]:
    """Return normalized pair along with base and quote currencies."""
    cleaned = pair.replace(" ", "").upper()

    if "/" in cleaned:
        base, quote = cleaned.split("/", maxsplit=1)
    elif len(cleaned) == 6:
        base, quote = cleaned[:3], cleaned[3:]
    else:
        raise ValueError(f"Invalid currency pair format: {pair}")

    if not base.isalpha() or not quote.isalpha() or len(base) != 3 or len(quote) != 3:
        raise ValueError(f"Invalid currency codes in pair: {pair}")

    return f"{base}/{quote}", base, quote


@dataclass
class AgentRequest:
    """Normalized user request payload for the graph."""

    raw_input: Dict[str, Any]
    currency_pair: str
    base_currency: str
    quote_currency: str
    amount: float
    risk_tolerance: str = "moderate"
    timeframe_days: int = _DEFAULT_TIMEFRAME_DAYS
    user_notes: Optional[str] = None

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "AgentRequest":
        """Create AgentRequest from an incoming payload."""
        if "currency_pair" not in payload:
            raise ValueError("currency_pair is required")

        normalized_pair, base, quote = _normalize_currency_pair(payload["currency_pair"])

        try:
            amount = float(payload.get("amount", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("amount must be numeric") from exc

        if amount <= 0:
            raise ValueError("amount must be greater than zero")

        risk = str(payload.get("risk_tolerance", "moderate")).lower()
        if risk not in _VALID_RISK_LEVELS:
            risk = "moderate"

        try:
            timeframe = int(payload.get("timeframe_days", _DEFAULT_TIMEFRAME_DAYS))
        except (TypeError, ValueError):
            timeframe = _DEFAULT_TIMEFRAME_DAYS

        timeframe = max(1, timeframe)

        return AgentRequest(
            raw_input=payload,
            currency_pair=normalized_pair,
            base_currency=base,
            quote_currency=quote,
            amount=amount,
            risk_tolerance=risk,
            timeframe_days=timeframe,
            user_notes=payload.get("user_notes"),
        )


@dataclass
class MarketAnalysis:
    """Output of the market analysis agent."""

    summary: Optional[str] = None
    bias: Optional[str] = None  # bullish, bearish, neutral
    regime: Optional[str] = None  # trending, ranging, volatile
    confidence: Optional[float] = None
    mid_rate: Optional[float] = None
    rate_timestamp: Optional[datetime] = None
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    ml_forecasts: Dict[str, Any] = field(default_factory=dict)
    indicators_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data_source_notes: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.summary is not None or bool(self.errors)


@dataclass
class EconomicAnalysis:
    """Output of the economic analysis agent."""

    summary: Optional[str] = None
    overall_bias: Optional[str] = None  # supportive, risk_on, risk_off, neutral
    upcoming_events: List[Dict[str, Any]] = field(default_factory=list)
    high_impact_events: List[Dict[str, Any]] = field(default_factory=list)
    event_window_days: int = 14
    confidence: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    data_source_notes: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.summary is not None or bool(self.errors)


@dataclass
class RiskAssessment:
    """Output of the risk assessment agent."""

    summary: Optional[str] = None
    risk_level: Optional[str] = None  # low, medium, high
    var_95: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    scenarios: Dict[str, Any] = field(default_factory=dict)
    hedging_notes: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.summary is not None or bool(self.errors)


@dataclass
class ProviderCostAnalysis:
    """Placeholder for future provider cost agent output."""

    status: str = "unavailable"
    summary: Optional[str] = None
    best_option: Optional[str] = None
    comparison_table: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status != "unavailable" or bool(self.errors)


@dataclass
class Recommendation:
    """Final decision coordinator output."""

    action: Optional[str] = None  # convert_now, wait, staged_conversion
    confidence: Optional[float] = None
    summary: Optional[str] = None
    rationale: List[str] = field(default_factory=list)
    timeline: Optional[str] = None
    next_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.action is not None or bool(self.errors)


@dataclass
class AgentMeta:
    """Metadata captured during graph execution."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def touch(self) -> None:
        self.last_updated = datetime.utcnow()


@dataclass
class AgentGraphState:
    """Shared state object passed between LangGraph nodes."""

    request: AgentRequest
    market_analysis: MarketAnalysis = field(default_factory=MarketAnalysis)
    economic_analysis: EconomicAnalysis = field(default_factory=EconomicAnalysis)
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    provider_costs: ProviderCostAnalysis = field(default_factory=ProviderCostAnalysis)
    recommendation: Recommendation = field(default_factory=Recommendation)
    meta: AgentMeta = field(default_factory=AgentMeta)

    def with_market(self, market: MarketAnalysis) -> "AgentGraphState":
        self.market_analysis = market
        self.meta.touch()
        return self

    def with_economic(self, economic: EconomicAnalysis) -> "AgentGraphState":
        self.economic_analysis = economic
        self.meta.touch()
        return self

    def with_risk(self, risk: RiskAssessment) -> "AgentGraphState":
        self.risk_assessment = risk
        self.meta.touch()
        return self

    def with_provider_costs(self, costs: ProviderCostAnalysis) -> "AgentGraphState":
        self.provider_costs = costs
        self.meta.touch()
        return self

    def with_recommendation(self, recommendation: Recommendation) -> "AgentGraphState":
        self.recommendation = recommendation
        self.meta.touch()
        return self


def initialize_state(payload: Dict[str, Any]) -> AgentGraphState:
    """Create the initial graph state from an inbound request payload."""
    request = AgentRequest.from_payload(payload)
    return AgentGraphState(request=request)
