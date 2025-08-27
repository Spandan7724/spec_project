"""
Data models for the decision engine.

Defines the data structures used for decision making, user profiles,
and recommendation outputs.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class RiskTolerance(str, Enum):
    """User risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"


class DecisionType(str, Enum):
    """Types of conversion decisions."""
    CONVERT_NOW = "convert_now"
    WAIT = "wait"
    INSUFFICIENT_DATA = "insufficient_data"


class ReasonCode(str, Enum):
    """Reason codes for decisions."""
    # Convert Now reasons
    FAVORABLE_PREDICTION = "favorable_prediction"
    HIGH_CONFIDENCE = "high_confidence"
    DEADLINE_APPROACHING = "deadline_approaching"
    COST_OPTIMIZED = "cost_optimized"
    VOLATILITY_RISING = "volatility_rising"
    
    # Wait reasons  
    BETTER_RATES_PREDICTED = "better_rates_predicted"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_VOLATILITY = "high_volatility"
    TIME_AVAILABLE = "time_available"
    COST_DISADVANTAGE = "cost_disadvantage"
    
    # Insufficient data
    MISSING_ML_PREDICTION = "missing_ml_prediction"
    INSUFFICIENT_HISTORY = "insufficient_history"
    API_ERROR = "api_error"


class UserProfile(BaseModel):
    """User profile for personalized recommendations."""
    user_id: str
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    fee_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)  # 0=ignore fees, 1=very sensitive
    typical_amount_usd: Optional[Decimal] = None
    preferred_providers: List[str] = field(default_factory=list)
    max_wait_days: int = Field(default=7, ge=1, le=30)
    
    @validator('fee_sensitivity')
    def validate_fee_sensitivity(cls, v):
        return max(0.0, min(1.0, v))


class ConversionRequest(BaseModel):
    """Request for conversion decision analysis."""
    from_currency: str = Field(..., pattern=r'^[A-Z]{3}$')
    to_currency: str = Field(..., pattern=r'^[A-Z]{3}$')
    amount: Decimal = Field(..., gt=0)
    user_profile: UserProfile
    deadline: Optional[datetime] = None
    current_rate: Optional[Decimal] = None
    
    @property
    def currency_pair(self) -> str:
        """Get currency pair string."""
        return f"{self.from_currency}/{self.to_currency}"
    
    @property
    def days_until_deadline(self) -> Optional[int]:
        """Calculate days until deadline."""
        if self.deadline is None:
            return None
        delta = self.deadline - datetime.utcnow()
        return max(0, delta.days)


class MLPrediction(BaseModel):
    """ML prediction data."""
    currency_pair: str
    current_rate: Decimal
    predicted_rates: List[Decimal]  # Predictions for next N hours/days
    confidence_intervals: List[Tuple[Decimal, Decimal]]  # (lower, upper) bounds
    model_confidence: float = Field(ge=0.0, le=1.0)
    prediction_horizon_hours: int
    timestamp: datetime
    
    @property
    def next_day_prediction(self) -> Optional[Decimal]:
        """Get 24-hour prediction if available."""
        if len(self.predicted_rates) >= 24:
            return self.predicted_rates[23]
        elif self.predicted_rates:
            return self.predicted_rates[-1]
        return None
    
    @property
    def expected_change_24h(self) -> Optional[float]:
        """Expected percentage change over next 24 hours."""
        next_day = self.next_day_prediction
        if next_day is None:
            return None
        return float((next_day - self.current_rate) / self.current_rate * 100)


class CostAnalysis(BaseModel):
    """Cost analysis for conversion."""
    base_rate: Decimal
    spread_cost: Decimal  # Cost due to bid-ask spread
    provider_fee: Decimal  # Fixed or percentage fee
    opportunity_cost: Decimal  # Cost of waiting (can be negative)
    total_cost: Decimal
    cost_percentage: float  # Total cost as percentage of conversion amount
    
    @classmethod
    def calculate(cls, amount: Decimal, base_rate: Decimal, 
                  spread: Decimal = Decimal('0.002'), 
                  fee_rate: Decimal = Decimal('0.001')) -> 'CostAnalysis':
        """Calculate costs for a conversion."""
        spread_cost = amount * spread
        provider_fee = amount * fee_rate
        total_cost = spread_cost + provider_fee
        
        return cls(
            base_rate=base_rate,
            spread_cost=spread_cost,
            provider_fee=provider_fee,
            opportunity_cost=Decimal('0'),  # Calculated separately
            total_cost=total_cost,
            cost_percentage=float(total_cost / amount * 100)
        )


class RiskAssessment(BaseModel):
    """Risk assessment for conversion decision."""
    volatility_score: float = Field(ge=0.0, le=1.0)  # 0=stable, 1=very volatile
    prediction_uncertainty: float = Field(ge=0.0, le=1.0)  # ML model uncertainty
    market_stability: float = Field(ge=0.0, le=1.0)  # Overall market conditions
    time_risk: float = Field(ge=0.0, le=1.0)  # Risk due to time constraints
    overall_risk: float = Field(ge=0.0, le=1.0)  # Combined risk score
    
    @classmethod
    def calculate(cls, ml_prediction: MLPrediction, 
                  days_until_deadline: Optional[int] = None) -> 'RiskAssessment':
        """Calculate risk assessment from ML prediction and context."""
        # Calculate prediction uncertainty (inverse of confidence)
        prediction_uncertainty = 1.0 - ml_prediction.model_confidence
        
        # Estimate volatility from prediction intervals
        if ml_prediction.confidence_intervals:
            avg_interval = sum(
                float(upper - lower) for lower, upper in ml_prediction.confidence_intervals[:24]
            ) / min(24, len(ml_prediction.confidence_intervals))
            volatility_score = min(1.0, avg_interval / float(ml_prediction.current_rate) * 10)
        else:
            volatility_score = 0.5  # Default moderate volatility
        
        # Calculate time risk
        if days_until_deadline is None:
            time_risk = 0.0  # No deadline pressure
        else:
            time_risk = max(0.0, 1.0 - days_until_deadline / 30.0)  # Higher risk as deadline approaches
        
        # Market stability (simplified - could be enhanced with external indicators)
        market_stability = 1.0 - volatility_score  # Inverse of volatility
        
        # Overall risk (weighted combination)
        overall_risk = (
            volatility_score * 0.3 +
            prediction_uncertainty * 0.3 + 
            (1.0 - market_stability) * 0.2 +
            time_risk * 0.2
        )
        
        return cls(
            volatility_score=volatility_score,
            prediction_uncertainty=prediction_uncertainty,
            market_stability=market_stability,
            time_risk=time_risk,
            overall_risk=overall_risk
        )


class DecisionRecommendation(BaseModel):
    """Final recommendation with reasoning."""
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence in recommendation
    expected_savings: Optional[Decimal] = None  # Expected savings if following advice
    expected_savings_percentage: Optional[float] = None
    reasoning: List[ReasonCode] = field(default_factory=list)
    explanation: str  # Human-readable explanation
    suggested_wait_days: Optional[int] = None  # If decision is WAIT
    alternative_rates: Optional[Dict[str, Decimal]] = None  # Other provider rates
    
    # Analysis components
    ml_prediction: Optional[MLPrediction] = None
    cost_analysis: Optional[CostAnalysis] = None
    risk_assessment: Optional[RiskAssessment] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    analysis_duration_ms: Optional[int] = None
    
    @property
    def is_actionable(self) -> bool:
        """Whether the recommendation is actionable (not insufficient data)."""
        return self.decision != DecisionType.INSUFFICIENT_DATA
    
    @property 
    def risk_level(self) -> str:
        """Get risk level description."""
        if not self.risk_assessment:
            return "Unknown"
        risk = self.risk_assessment.overall_risk
        if risk < 0.3:
            return "Low"
        elif risk < 0.7:
            return "Moderate" 
        else:
            return "High"


class DecisionContext(BaseModel):
    """Complete context for decision making."""
    request: ConversionRequest
    current_market_data: Dict[str, Decimal]  # Current rates from providers
    ml_prediction: Optional[MLPrediction] = None
    provider_fees: Dict[str, Decimal] = field(default_factory=dict)  # Provider fee rates
    market_hours: bool = True  # Whether markets are open
    
    @property
    def best_current_rate(self) -> Optional[Decimal]:
        """Get best available rate from providers."""
        if not self.current_market_data:
            return None
        return max(self.current_market_data.values())
    
    @property
    def rate_spread(self) -> Optional[Decimal]:
        """Get spread between best and worst rates."""
        if len(self.current_market_data) < 2:
            return None
        rates = list(self.current_market_data.values())
        return max(rates) - min(rates)