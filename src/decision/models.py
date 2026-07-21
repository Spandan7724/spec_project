from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DecisionRequest:
    """Input to the decision engine.

    All percentages in request subfields (e.g., prediction mean_change_pct) are expected
    to be expressed as percent units (e.g., +0.3 means +0.3%). Prediction changes follow
    ``currency_pair`` market direction; the engine translates them into the economic
    direction of selling ``source_currency`` to buy ``target_currency``. Costs are bps.
    """

    # User constraints
    amount: float
    risk_tolerance: str  # conservative | moderate | aggressive
    urgency: str  # urgent | normal | flexible
    timeframe: str  # immediate | 1_day | 1_week | 1_month
    timeframe_days: int
    currency_pair: Optional[str] = None
    # Conversion contract: sell source_currency to buy target_currency.  The
    # market pair may be direct (source/target) or inverse (target/source).
    source_currency: Optional[str] = None
    target_currency: Optional[str] = None

    # Upstream agent data
    market: Dict[str, Any] = field(default_factory=dict)
    intelligence: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None

    # Costs (basis points). If None, defaults from config are used.
    spread_bps: Optional[float] = None
    fee_bps: Optional[float] = None

    # System/diagnostics
    components_available: Dict[str, bool] = field(
        default_factory=lambda: {"market": False, "intelligence": False, "prediction": False}
    )
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.currency_pair and "/" in self.currency_pair:
            pair_base, pair_quote = self.currency_pair.upper().split("/", 1)
            self.currency_pair = f"{pair_base}/{pair_quote}"
            self.source_currency = (self.source_currency or pair_base).upper()
            self.target_currency = (self.target_currency or pair_quote).upper()
            if self.source_currency == self.target_currency or {
                self.source_currency,
                self.target_currency,
            } != {pair_base, pair_quote}:
                raise ValueError(
                    "source_currency and target_currency must be the two currencies in currency_pair"
                )
        elif self.source_currency and self.target_currency:
            self.source_currency = self.source_currency.upper()
            self.target_currency = self.target_currency.upper()
            if self.source_currency == self.target_currency:
                raise ValueError("source_currency and target_currency must differ")
            self.currency_pair = f"{self.source_currency}/{self.target_currency}"


@dataclass
class TrancheSpec:
    tranche_number: int
    percentage: float  # 0-100 (percent of total amount)
    execute_day: int  # day offset from now (0-based)
    rationale: str = ""


@dataclass
class StagedPlan:
    num_tranches: int
    tranches: List[TrancheSpec] = field(default_factory=list)
    spacing_days: float = 0.0
    total_extra_cost_bps: float = 0.0
    benefit: str = ""


@dataclass
class ExpectedOutcome:
    expected_rate: float = 0.0
    range_low: float = 0.0
    range_high: float = 0.0
    expected_improvement_bps: float = 0.0


@dataclass
class RiskSummary:
    risk_level: str = "low"  # low | moderate | high
    realized_vol_30d: float = 0.0  # percent
    var_95: float = 0.0  # percent
    event_risk: str = "none"  # none | low | moderate | high
    event_details: Optional[str] = None


@dataclass
class CostEstimate:
    spread_bps: float = 0.0
    fee_bps: float = 0.0
    total_bps: float = 0.0
    staged_multiplier: float = 1.0


@dataclass
class DecisionResponse:
    # Core decision
    action: str  # convert_now | staged_conversion | wait
    confidence: float  # 0.0-1.0
    timeline: str

    # Plans and estimates
    staged_plan: Optional[StagedPlan] = None
    expected_outcome: ExpectedOutcome = field(default_factory=ExpectedOutcome)
    risk_summary: RiskSummary = field(default_factory=RiskSummary)
    cost_estimate: CostEstimate = field(default_factory=CostEstimate)

    # Explanation/diagnostics
    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    utility_scores: Dict[str, float] = field(default_factory=dict)
    component_confidences: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
