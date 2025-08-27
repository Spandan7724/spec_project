"""
Core Decision Engine for Currency Conversion Recommendations.

This module implements the main decision logic that converts ML predictions
and market data into actionable "convert now" vs "wait" recommendations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .models import (
    ConversionRequest, DecisionContext, DecisionRecommendation,
    DecisionType, ReasonCode, RiskTolerance, MLPrediction,
    CostAnalysis, RiskAssessment
)

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Main decision engine that analyzes conversion opportunities.
    
    Takes ML predictions, market data, and user context to generate
    personalized "convert now" vs "wait" recommendations with reasoning.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self.min_confidence_threshold = 0.6  # Minimum ML confidence to make recommendations
        self.significant_change_threshold = 0.5  # Minimum % change to consider significant
        self.max_wait_recommendation = 14  # Maximum days to recommend waiting
        
    async def analyze_conversion_opportunity(
        self, 
        context: DecisionContext
    ) -> DecisionRecommendation:
        """
        Main entry point for decision analysis.
        
        Args:
            context: Complete context including request, market data, and ML predictions
            
        Returns:
            DecisionRecommendation with decision, confidence, and reasoning
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate input data
            if not self._validate_context(context):
                return self._create_insufficient_data_recommendation(
                    context, "Invalid or missing context data"
                )
            
            # Check if we have sufficient ML prediction data
            if not context.ml_prediction or context.ml_prediction.model_confidence < 0.3:
                return self._create_insufficient_data_recommendation(
                    context, "ML prediction unavailable or low quality"
                )
            
            # Perform cost-benefit analysis
            cost_analysis = await self._analyze_costs(context)
            
            # Assess risks
            risk_assessment = await self._assess_risks(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                context, cost_analysis, risk_assessment
            )
            
            # Add analysis metadata
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            recommendation.analysis_duration_ms = int(duration)
            recommendation.cost_analysis = cost_analysis
            recommendation.risk_assessment = risk_assessment
            recommendation.ml_prediction = context.ml_prediction
            
            logger.info(
                f"Decision analysis completed for {context.request.currency_pair}: "
                f"{recommendation.decision} (confidence: {recommendation.confidence:.2f})"
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Decision analysis failed: {e}")
            return self._create_insufficient_data_recommendation(
                context, f"Analysis error: {str(e)}"
            )
    
    def _validate_context(self, context: DecisionContext) -> bool:
        """Validate that context has required data."""
        if not context.request:
            return False
        if not context.current_market_data:
            return False
        if context.request.amount <= 0:
            return False
        return True
    
    async def _analyze_costs(self, context: DecisionContext) -> CostAnalysis:
        """
        Analyze costs of immediate conversion vs waiting.
        
        Considers provider fees, spreads, and opportunity costs.
        """
        request = context.request
        
        # Get best available rate
        best_rate = context.best_current_rate or Decimal('1.0')
        
        # Estimate typical spread (0.2% for major pairs)
        spread = self._estimate_spread(request.currency_pair)
        
        # Get provider fee (default 0.1% if not available)
        fee_rate = context.provider_fees.get('default', Decimal('0.001'))
        
        # Calculate immediate conversion costs
        cost_analysis = CostAnalysis.calculate(
            amount=request.amount,
            base_rate=best_rate,
            spread=spread,
            fee_rate=fee_rate
        )
        
        # Calculate opportunity cost based on ML prediction
        if context.ml_prediction and context.ml_prediction.expected_change_24h:
            predicted_change = Decimal(str(context.ml_prediction.expected_change_24h / 100))
            potential_gain = request.amount * predicted_change
            
            # Opportunity cost is negative if we expect rates to improve
            cost_analysis.opportunity_cost = -potential_gain
            cost_analysis.total_cost += cost_analysis.opportunity_cost
        
        return cost_analysis
    
    async def _assess_risks(self, context: DecisionContext) -> RiskAssessment:
        """
        Assess risks associated with the conversion decision.
        
        Considers prediction uncertainty, market volatility, and time constraints.
        """
        request = context.request
        ml_prediction = context.ml_prediction
        
        if not ml_prediction:
            # Default moderate risk if no prediction
            return RiskAssessment(
                volatility_score=0.5,
                prediction_uncertainty=0.8,
                market_stability=0.5,
                time_risk=0.0,
                overall_risk=0.6
            )
        
        return RiskAssessment.calculate(
            ml_prediction=ml_prediction,
            days_until_deadline=request.days_until_deadline
        )
    
    async def _generate_recommendation(
        self, 
        context: DecisionContext,
        cost_analysis: CostAnalysis,
        risk_assessment: RiskAssessment
    ) -> DecisionRecommendation:
        """
        Generate final recommendation based on analysis.
        
        Uses rule-based logic to combine ML predictions, costs, and risks
        into actionable advice.
        """
        request = context.request
        ml_prediction = context.ml_prediction
        user_profile = request.user_profile
        
        # Initialize recommendation
        recommendation = DecisionRecommendation(
            decision=DecisionType.WAIT,  # Default to wait
            confidence=0.5,
            reasoning=[],
            explanation=""
        )
        
        # Decision factors
        factors = await self._evaluate_decision_factors(
            context, cost_analysis, risk_assessment
        )
        
        # Apply decision rules
        decision_score = 0.0  # Positive = convert now, negative = wait
        reasoning = []
        
        # ML Prediction factor (40% weight)
        if ml_prediction and ml_prediction.expected_change_24h:
            change = ml_prediction.expected_change_24h
            confidence_weight = ml_prediction.model_confidence
            
            if change > self.significant_change_threshold:
                # Rates expected to get worse - favor convert now
                decision_score += 0.4 * confidence_weight
                reasoning.append(ReasonCode.FAVORABLE_PREDICTION)
            elif change < -self.significant_change_threshold:
                # Rates expected to improve - favor waiting
                decision_score -= 0.4 * confidence_weight
                reasoning.append(ReasonCode.BETTER_RATES_PREDICTED)
        
        # Risk factor (30% weight)
        risk_factor = self._calculate_risk_factor(risk_assessment, user_profile)
        decision_score += risk_factor * 0.3
        
        if risk_assessment.overall_risk > 0.7:
            reasoning.append(ReasonCode.HIGH_VOLATILITY)
        elif ml_prediction and ml_prediction.model_confidence < 0.5:
            reasoning.append(ReasonCode.LOW_CONFIDENCE)
        elif ml_prediction and ml_prediction.model_confidence > 0.8:
            reasoning.append(ReasonCode.HIGH_CONFIDENCE)
        
        # Time constraint factor (20% weight)
        time_factor = self._calculate_time_factor(request)
        decision_score += time_factor * 0.2
        
        if request.days_until_deadline and request.days_until_deadline <= 2:
            reasoning.append(ReasonCode.DEADLINE_APPROACHING)
        elif not request.deadline:
            reasoning.append(ReasonCode.TIME_AVAILABLE)
        
        # Cost factor (10% weight)
        cost_factor = self._calculate_cost_factor(cost_analysis, user_profile)
        decision_score += cost_factor * 0.1
        
        if cost_analysis.cost_percentage < 0.5:  # Low cost conversion
            reasoning.append(ReasonCode.COST_OPTIMIZED)
        elif cost_analysis.cost_percentage > 2.0:  # High cost conversion
            reasoning.append(ReasonCode.COST_DISADVANTAGE)
        
        # Make final decision
        if decision_score > 0.2:
            recommendation.decision = DecisionType.CONVERT_NOW
            recommendation.confidence = min(0.95, 0.5 + abs(decision_score))
        elif decision_score < -0.2:
            recommendation.decision = DecisionType.WAIT
            recommendation.confidence = min(0.95, 0.5 + abs(decision_score))
            recommendation.suggested_wait_days = self._calculate_wait_period(
                ml_prediction, request
            )
        else:
            # Neutral zone - lean towards user's risk tolerance
            if user_profile.risk_tolerance == RiskTolerance.CONSERVATIVE:
                recommendation.decision = DecisionType.CONVERT_NOW
            else:
                recommendation.decision = DecisionType.WAIT
            recommendation.confidence = 0.5
        
        # Calculate expected savings
        if cost_analysis.opportunity_cost and cost_analysis.opportunity_cost != 0:
            if recommendation.decision == DecisionType.WAIT:
                recommendation.expected_savings = -cost_analysis.opportunity_cost
            else:
                recommendation.expected_savings = Decimal('0')
            
            if recommendation.expected_savings:
                recommendation.expected_savings_percentage = float(
                    recommendation.expected_savings / request.amount * 100
                )
        
        # Generate explanation
        recommendation.reasoning = reasoning
        recommendation.explanation = self._generate_explanation(
            recommendation, factors, context
        )
        
        return recommendation
    
    async def _evaluate_decision_factors(
        self,
        context: DecisionContext,
        cost_analysis: CostAnalysis, 
        risk_assessment: RiskAssessment
    ) -> Dict[str, float]:
        """Evaluate various decision factors and return scores."""
        factors = {}
        
        # ML prediction strength
        if context.ml_prediction:
            factors['ml_strength'] = abs(context.ml_prediction.expected_change_24h or 0)
            factors['ml_confidence'] = context.ml_prediction.model_confidence
        else:
            factors['ml_strength'] = 0
            factors['ml_confidence'] = 0
        
        # Cost efficiency
        factors['cost_efficiency'] = max(0, 2.0 - cost_analysis.cost_percentage) / 2.0
        
        # Risk level
        factors['risk_level'] = risk_assessment.overall_risk
        
        # Time pressure
        if context.request.days_until_deadline:
            factors['time_pressure'] = max(0, 1.0 - context.request.days_until_deadline / 7.0)
        else:
            factors['time_pressure'] = 0
        
        return factors
    
    def _calculate_risk_factor(
        self, 
        risk_assessment: RiskAssessment, 
        user_profile
    ) -> float:
        """Calculate risk-adjusted decision factor."""
        base_risk = risk_assessment.overall_risk
        
        # Adjust based on user risk tolerance
        if user_profile.risk_tolerance == RiskTolerance.CONSERVATIVE:
            # Conservative users prefer immediate conversion when risk is high
            return base_risk  # High risk = favor convert now
        elif user_profile.risk_tolerance == RiskTolerance.AGGRESSIVE:
            # Aggressive users comfortable with risk, willing to wait
            return -base_risk * 0.5  # High risk = still willing to wait
        else:
            # Moderate users neutral on risk
            return 0
    
    def _calculate_time_factor(self, request: ConversionRequest) -> float:
        """Calculate time constraint factor."""
        if not request.deadline:
            return -0.2  # No deadline = slight favor to waiting
        
        days_left = request.days_until_deadline or 0
        
        if days_left <= 1:
            return 1.0  # Must convert now
        elif days_left <= 3:
            return 0.5  # Strong favor to convert now
        elif days_left <= 7:
            return 0.0  # Neutral
        else:
            return -0.3  # Favor waiting
    
    def _calculate_cost_factor(
        self, 
        cost_analysis: CostAnalysis, 
        user_profile
    ) -> float:
        """Calculate cost-based decision factor."""
        cost_pct = cost_analysis.cost_percentage
        fee_sensitivity = user_profile.fee_sensitivity
        
        # Base cost factor (lower cost = favor immediate conversion)
        base_factor = max(-0.5, min(0.5, (1.0 - cost_pct) * 0.5))
        
        # Adjust for user fee sensitivity
        return base_factor * fee_sensitivity
    
    def _calculate_wait_period(
        self, 
        ml_prediction: Optional[MLPrediction],
        request: ConversionRequest
    ) -> int:
        """Calculate recommended wait period in days."""
        if not ml_prediction:
            return 3  # Default wait period
        
        # Base wait period on prediction horizon and expected change
        expected_change = abs(ml_prediction.expected_change_24h or 0)
        
        if expected_change > 2.0:  # Significant change expected
            wait_days = 1
        elif expected_change > 1.0:
            wait_days = 3
        else:
            wait_days = 7
        
        # Respect deadline constraints
        if request.days_until_deadline:
            wait_days = min(wait_days, request.days_until_deadline - 1)
        
        # Respect maximum wait period
        return min(wait_days, self.max_wait_recommendation)
    
    def _estimate_spread(self, currency_pair: str) -> Decimal:
        """Estimate bid-ask spread for currency pair."""
        major_pairs = ['USD/EUR', 'USD/GBP', 'EUR/GBP', 'USD/JPY']
        
        if currency_pair in major_pairs:
            return Decimal('0.002')  # 0.2% for major pairs
        else:
            return Decimal('0.005')  # 0.5% for minor pairs
    
    def _generate_explanation(
        self,
        recommendation: DecisionRecommendation,
        factors: Dict[str, float],
        context: DecisionContext
    ) -> str:
        """Generate human-readable explanation for the recommendation."""
        decision = recommendation.decision
        confidence = recommendation.confidence
        
        if decision == DecisionType.CONVERT_NOW:
            base = f"I recommend converting now (confidence: {confidence:.0%}). "
        elif decision == DecisionType.WAIT:
            wait_days = recommendation.suggested_wait_days or 3
            base = f"I recommend waiting {wait_days} days (confidence: {confidence:.0%}). "
        else:
            return "Insufficient data to make a recommendation."
        
        # Add reasoning based on factors
        reasons = []
        
        if context.ml_prediction:
            change = context.ml_prediction.expected_change_24h or 0
            if abs(change) > 0.5:
                direction = "improve" if change < 0 else "worsen"
                reasons.append(f"rates are predicted to {direction} by {abs(change):.1f}%")
        
        if 'risk_level' in factors and factors['risk_level'] > 0.7:
            reasons.append("market volatility is high")
        
        if context.request.days_until_deadline and context.request.days_until_deadline <= 3:
            reasons.append("your deadline is approaching")
        
        if 'cost_efficiency' in factors and factors['cost_efficiency'] < 0.3:
            reasons.append("conversion costs are relatively high")
        
        if reasons:
            base += "This is because " + ", ".join(reasons) + "."
        
        # Add expected savings if significant
        if recommendation.expected_savings and abs(recommendation.expected_savings) > 10:
            savings_pct = recommendation.expected_savings_percentage or 0
            if savings_pct > 0:
                base += f" Following this advice could save you approximately {savings_pct:.1f}%."
            else:
                base += f" Immediate conversion avoids potential losses of {abs(savings_pct):.1f}%."
        
        return base
    
    def _create_insufficient_data_recommendation(
        self, 
        context: DecisionContext, 
        reason: str
    ) -> DecisionRecommendation:
        """Create recommendation for insufficient data scenarios."""
        return DecisionRecommendation(
            decision=DecisionType.INSUFFICIENT_DATA,
            confidence=0.0,
            reasoning=[ReasonCode.MISSING_ML_PREDICTION],
            explanation=f"Unable to provide recommendation: {reason}",
            ml_prediction=context.ml_prediction,
            timestamp=datetime.utcnow()
        )