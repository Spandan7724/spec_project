"""
Specialized analyzers for decision engine components.

This module contains detailed analysis components for costs, risks,
and market conditions that support the main decision engine.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import statistics

from .models import (
    ConversionRequest, DecisionContext, MLPrediction,
    CostAnalysis, RiskAssessment, RiskTolerance
)

logger = logging.getLogger(__name__)


class CostAnalyzer:
    """
    Detailed cost-benefit analysis for currency conversions.
    
    Analyzes immediate conversion costs vs potential savings from waiting,
    considering provider fees, spreads, and opportunity costs.
    """
    
    def __init__(self):
        """Initialize cost analyzer with default parameters."""
        self.default_spreads = {
            # Major pairs (lower spreads)
            'USD/EUR': Decimal('0.0015'),
            'USD/GBP': Decimal('0.0020'), 
            'EUR/GBP': Decimal('0.0025'),
            'USD/JPY': Decimal('0.0018'),
            'GBP/JPY': Decimal('0.0035'),
            'EUR/JPY': Decimal('0.0030'),
            
            # Minor pairs (higher spreads)
            'USD/CHF': Decimal('0.0040'),
            'USD/CAD': Decimal('0.0035'),
            'AUD/USD': Decimal('0.0030'),
            'NZD/USD': Decimal('0.0045'),
            
            # Default for unknown pairs
            'default': Decimal('0.0050')
        }
        
        # Typical provider fee structures
        self.provider_fees = {
            'bank_transfer': Decimal('0.002'),  # 0.2%
            'credit_card': Decimal('0.025'),   # 2.5%
            'debit_card': Decimal('0.015'),    # 1.5%
            'digital_wallet': Decimal('0.008'), # 0.8%
            'crypto': Decimal('0.005'),        # 0.5%
            'default': Decimal('0.010')        # 1.0%
        }
    
    async def analyze_comprehensive_costs(
        self, 
        context: DecisionContext
    ) -> CostAnalysis:
        """
        Perform comprehensive cost analysis including all fee types.
        
        Args:
            context: Decision context with request and market data
            
        Returns:
            Detailed cost analysis with breakdown
        """
        request = context.request
        amount = request.amount
        currency_pair = request.currency_pair
        
        # Get current best rate
        current_rate = context.best_current_rate or Decimal('1.0')
        
        # Calculate spread costs
        spread = self._get_spread(currency_pair)
        spread_cost = amount * current_rate * spread
        
        # Calculate provider fees
        provider_fee = self._calculate_provider_fee(amount, context)
        
        # Calculate immediate conversion cost
        immediate_total = spread_cost + provider_fee
        
        # Calculate opportunity cost from ML prediction
        opportunity_cost = await self._calculate_opportunity_cost(
            context, current_rate
        )
        
        # Create detailed cost analysis
        cost_analysis = CostAnalysis(
            base_rate=current_rate,
            spread_cost=spread_cost,
            provider_fee=provider_fee,
            opportunity_cost=opportunity_cost,
            total_cost=immediate_total + opportunity_cost,
            cost_percentage=float((immediate_total + opportunity_cost) / amount * 100)
        )
        
        logger.debug(
            f"Cost analysis for {currency_pair}: "
            f"spread={spread_cost}, fee={provider_fee}, "
            f"opportunity={opportunity_cost}, total={cost_analysis.total_cost}"
        )
        
        return cost_analysis
    
    def _get_spread(self, currency_pair: str) -> Decimal:
        """Get estimated spread for currency pair."""
        return self.default_spreads.get(currency_pair, self.default_spreads['default'])
    
    def _calculate_provider_fee(
        self, 
        amount: Decimal, 
        context: DecisionContext
    ) -> Decimal:
        """Calculate provider fees based on conversion method."""
        # Use configured fee rates or defaults
        fee_rates = context.provider_fees or {}
        
        # For now, use default fee rate
        # TODO: Integrate with actual provider fee API
        default_fee_rate = fee_rates.get('default', self.provider_fees['default'])
        
        return amount * default_fee_rate
    
    async def _calculate_opportunity_cost(
        self, 
        context: DecisionContext,
        current_rate: Decimal
    ) -> Decimal:
        """
        Calculate opportunity cost of immediate conversion vs waiting.
        
        Positive value means cost of converting now (rates expected to improve).
        Negative value means benefit of converting now (rates expected to worsen).
        """
        ml_prediction = context.ml_prediction
        if not ml_prediction:
            return Decimal('0')
        
        amount = context.request.amount
        
        # Get expected rate change
        expected_change_pct = ml_prediction.expected_change_24h
        if expected_change_pct is None:
            return Decimal('0')
        
        # Convert percentage to decimal
        expected_change = Decimal(str(expected_change_pct / 100))
        
        # Calculate potential gain/loss from waiting
        expected_future_rate = current_rate * (1 + expected_change)
        rate_difference = expected_future_rate - current_rate
        
        # Opportunity cost is the potential gain/loss from waiting
        # Positive = cost of converting now (better to wait)
        # Negative = benefit of converting now (better not to wait)
        opportunity_cost = amount * rate_difference
        
        # Adjust for prediction confidence
        confidence = ml_prediction.model_confidence
        opportunity_cost *= Decimal(str(confidence))
        
        # Factor in time decay (opportunity cost decreases with longer waits)
        if context.request.days_until_deadline:
            days = context.request.days_until_deadline
            time_decay = Decimal(str(max(0.1, 1.0 - days / 30.0)))  # Decay over 30 days
            opportunity_cost *= time_decay
        
        return opportunity_cost
    
    def analyze_provider_comparison(
        self, 
        context: DecisionContext
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compare costs across different providers.
        
        Returns:
            Dictionary mapping provider names to cost breakdowns
        """
        comparison = {}
        
        for provider_name, rate in context.current_market_data.items():
            spread = self._get_spread(context.request.currency_pair)
            fee_rate = context.provider_fees.get(provider_name, self.provider_fees['default'])
            
            spread_cost = context.request.amount * rate * spread
            provider_fee = context.request.amount * fee_rate
            total_cost = spread_cost + provider_fee
            
            comparison[provider_name] = {
                'rate': rate,
                'spread_cost': spread_cost,
                'provider_fee': provider_fee,
                'total_cost': total_cost,
                'cost_percentage': float(total_cost / context.request.amount * 100)
            }
        
        return comparison


class RiskAnalyzer:
    """
    Advanced risk assessment for currency conversion decisions.
    
    Evaluates market volatility, prediction uncertainty, time risks,
    and external factors that could affect conversion outcomes.
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.volatility_windows = [6, 12, 24, 48, 168]  # Hours to analyze
        self.risk_weights = {
            'volatility': 0.35,
            'prediction_uncertainty': 0.25, 
            'time_risk': 0.20,
            'market_conditions': 0.20
        }
    
    async def analyze_comprehensive_risk(
        self, 
        context: DecisionContext
    ) -> RiskAssessment:
        """
        Perform comprehensive risk analysis.
        
        Args:
            context: Decision context with request and predictions
            
        Returns:
            Detailed risk assessment
        """
        ml_prediction = context.ml_prediction
        if not ml_prediction:
            return self._create_default_risk_assessment()
        
        # Analyze different risk components
        volatility_score = await self._analyze_volatility(ml_prediction)
        prediction_uncertainty = self._analyze_prediction_uncertainty(ml_prediction)
        time_risk = self._analyze_time_risk(context.request)
        market_stability = await self._analyze_market_conditions(context)
        
        # Calculate weighted overall risk
        overall_risk = (
            volatility_score * self.risk_weights['volatility'] +
            prediction_uncertainty * self.risk_weights['prediction_uncertainty'] +
            time_risk * self.risk_weights['time_risk'] +
            (1.0 - market_stability) * self.risk_weights['market_conditions']
        )
        
        risk_assessment = RiskAssessment(
            volatility_score=volatility_score,
            prediction_uncertainty=prediction_uncertainty,
            market_stability=market_stability,
            time_risk=time_risk,
            overall_risk=min(1.0, overall_risk)
        )
        
        logger.debug(
            f"Risk analysis for {context.request.currency_pair}: "
            f"volatility={volatility_score:.3f}, uncertainty={prediction_uncertainty:.3f}, "
            f"time_risk={time_risk:.3f}, overall={overall_risk:.3f}"
        )
        
        return risk_assessment
    
    async def _analyze_volatility(self, ml_prediction: MLPrediction) -> float:
        """
        Analyze market volatility from prediction intervals.
        
        Higher values indicate more volatile/risky conditions.
        """
        if not ml_prediction.confidence_intervals:
            return 0.5  # Default moderate volatility
        
        # Calculate average interval width as volatility proxy
        interval_widths = []
        current_rate = float(ml_prediction.current_rate)
        
        for lower, upper in ml_prediction.confidence_intervals[:24]:  # Use first 24 hours
            width = float(upper - lower)
            relative_width = width / current_rate  # Normalize by rate level
            interval_widths.append(relative_width)
        
        if not interval_widths:
            return 0.5
        
        # Use median interval width to avoid outliers
        median_width = statistics.median(interval_widths)
        
        # Convert to 0-1 scale (assuming 10% width = max volatility)
        volatility_score = min(1.0, median_width / 0.10)
        
        return volatility_score
    
    def _analyze_prediction_uncertainty(self, ml_prediction: MLPrediction) -> float:
        """
        Analyze ML prediction uncertainty.
        
        Returns uncertainty score (0 = very confident, 1 = very uncertain).
        """
        base_uncertainty = 1.0 - ml_prediction.model_confidence
        
        # Increase uncertainty if prediction change is very small
        # (harder to be confident about small changes)
        expected_change = abs(ml_prediction.expected_change_24h or 0)
        if expected_change < 0.1:  # Less than 0.1% change
            base_uncertainty = min(1.0, base_uncertainty + 0.2)
        
        # Increase uncertainty for longer prediction horizons
        if ml_prediction.prediction_horizon_hours > 24:
            horizon_penalty = min(0.3, (ml_prediction.prediction_horizon_hours - 24) / 168 * 0.3)
            base_uncertainty = min(1.0, base_uncertainty + horizon_penalty)
        
        return base_uncertainty
    
    def _analyze_time_risk(self, request: ConversionRequest) -> float:
        """
        Analyze risks related to timing constraints.
        
        Returns time risk score (0 = no time pressure, 1 = urgent deadline).
        """
        if not request.deadline:
            return 0.0  # No deadline = no time risk
        
        days_left = request.days_until_deadline or 0
        
        if days_left <= 0:
            return 1.0  # Deadline passed = maximum risk
        elif days_left == 1:
            return 0.8  # Very urgent
        elif days_left <= 3:
            return 0.6  # Urgent
        elif days_left <= 7:
            return 0.3  # Some pressure
        elif days_left <= 14:
            return 0.1  # Minimal pressure
        else:
            return 0.0  # No pressure
    
    async def _analyze_market_conditions(self, context: DecisionContext) -> float:
        """
        Analyze overall market stability conditions.
        
        Returns stability score (0 = very unstable, 1 = very stable).
        """
        # For now, use spread analysis as proxy for market conditions
        if not context.current_market_data or len(context.current_market_data) < 2:
            return 0.7  # Default moderate stability
        
        rates = list(context.current_market_data.values())
        
        # Calculate rate spread as percentage
        min_rate = min(rates)
        max_rate = max(rates)
        if min_rate == 0:
            return 0.5
        
        spread_pct = float((max_rate - min_rate) / min_rate * 100)
        
        # Convert spread to stability (wider spread = less stable)
        if spread_pct < 0.1:
            stability = 1.0  # Very stable
        elif spread_pct < 0.5:
            stability = 0.8  # Stable
        elif spread_pct < 1.0:
            stability = 0.6  # Moderate
        elif spread_pct < 2.0:
            stability = 0.4  # Unstable
        else:
            stability = 0.2  # Very unstable
        
        return stability
    
    def _create_default_risk_assessment(self) -> RiskAssessment:
        """Create default risk assessment when ML data is unavailable."""
        return RiskAssessment(
            volatility_score=0.5,
            prediction_uncertainty=0.8,  # High uncertainty without ML
            market_stability=0.5,
            time_risk=0.0,
            overall_risk=0.6
        )
    
    def analyze_risk_by_user_profile(
        self, 
        base_risk: RiskAssessment,
        user_profile
    ) -> RiskAssessment:
        """
        Adjust risk assessment based on user risk tolerance.
        
        Different users perceive and tolerate risk differently.
        """
        adjusted_risk = RiskAssessment(**base_risk.dict())
        
        # Conservative users are more sensitive to all risks
        if user_profile.risk_tolerance == RiskTolerance.CONSERVATIVE:
            adjusted_risk.overall_risk = min(1.0, base_risk.overall_risk * 1.2)
            adjusted_risk.volatility_score = min(1.0, base_risk.volatility_score * 1.1)
        
        # Aggressive users are less sensitive to risks
        elif user_profile.risk_tolerance == RiskTolerance.AGGRESSIVE:
            adjusted_risk.overall_risk = base_risk.overall_risk * 0.8
            adjusted_risk.volatility_score = base_risk.volatility_score * 0.9
        
        # Moderate users use base assessment unchanged
        
        return adjusted_risk


class MarketConditionAnalyzer:
    """
    Analyzes broader market conditions that might affect currency decisions.
    
    Considers market hours, volatility patterns, and external factors.
    """
    
    def __init__(self):
        """Initialize market condition analyzer."""
        self.major_market_sessions = {
            'sydney': (22, 7),    # UTC hours
            'tokyo': (0, 9),
            'london': (8, 17),
            'new_york': (13, 22)
        }
    
    def is_market_hours(self, currency_pair: str) -> bool:
        """
        Check if major markets for currency pair are open.
        
        Args:
            currency_pair: Currency pair like 'USD/EUR'
            
        Returns:
            True if major markets are open
        """
        current_hour = datetime.utcnow().hour
        
        # Get relevant markets for currency pair
        relevant_markets = self._get_relevant_markets(currency_pair)
        
        # Check if any relevant market is open
        for market in relevant_markets:
            start, end = self.major_market_sessions[market]
            if start <= current_hour <= end:
                return True
        
        return False
    
    def _get_relevant_markets(self, currency_pair: str) -> List[str]:
        """Get relevant trading markets for currency pair."""
        markets = []
        
        if 'USD' in currency_pair:
            markets.append('new_york')
        if 'EUR' in currency_pair or 'GBP' in currency_pair:
            markets.append('london')
        if 'JPY' in currency_pair or 'AUD' in currency_pair:
            markets.extend(['tokyo', 'sydney'])
        
        # Default to major markets if no specific match
        if not markets:
            markets = ['london', 'new_york']
        
        return markets
    
    def get_market_overlap_score(self, currency_pair: str) -> float:
        """
        Get market overlap score indicating liquidity.
        
        Higher scores indicate more market overlap and liquidity.
        """
        current_hour = datetime.utcnow().hour
        relevant_markets = self._get_relevant_markets(currency_pair)
        
        # Count how many relevant markets are open
        open_markets = 0
        for market in relevant_markets:
            start, end = self.major_market_sessions[market]
            if start <= current_hour <= end:
                open_markets += 1
        
        # Convert to 0-1 score
        if not relevant_markets:
            return 0.5
        
        return open_markets / len(relevant_markets)