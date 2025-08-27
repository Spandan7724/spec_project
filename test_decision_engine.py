"""
Integration tests for the Decision Engine.

Tests the complete workflow from data input to recommendation output,
ensuring all components work together correctly.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from core.decision_engine import DecisionEngine
from core.models import (
    ConversionRequest, DecisionContext, DecisionRecommendation,
    DecisionType, ReasonCode, RiskTolerance, MLPrediction, UserProfile
)
from core.analyzers import CostAnalyzer, RiskAnalyzer, MarketConditionAnalyzer


class TestDecisionEngine:
    """Test suite for the complete decision engine workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DecisionEngine()
        self.cost_analyzer = CostAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.market_analyzer = MarketConditionAnalyzer()
    
    def create_sample_user_profile(
        self, 
        risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    ) -> UserProfile:
        """Create sample user profile for testing."""
        return UserProfile(
            user_id="test_user_123",
            risk_tolerance=risk_tolerance,
            fee_sensitivity=0.5,
            typical_amount_usd=Decimal('10000'),
            max_wait_days=7
        )
    
    def create_sample_conversion_request(
        self, 
        amount: Decimal = Decimal('10000'),
        deadline: datetime = None
    ) -> ConversionRequest:
        """Create sample conversion request for testing."""
        return ConversionRequest(
            from_currency="USD",
            to_currency="EUR", 
            amount=amount,
            user_profile=self.create_sample_user_profile(),
            deadline=deadline,
            current_rate=Decimal('0.85')
        )
    
    def create_sample_ml_prediction(
        self, 
        expected_change: float = 1.5,
        confidence: float = 0.8
    ) -> MLPrediction:
        """Create sample ML prediction for testing."""
        current_rate = Decimal('0.85')
        predicted_rates = []
        confidence_intervals = []
        
        # Generate 24-hour predictions
        for hour in range(24):
            # Simulate gradual change over time
            change_factor = 1 + (expected_change / 100) * (hour / 24)
            predicted_rate = current_rate * Decimal(str(change_factor))
            predicted_rates.append(predicted_rate)
            
            # Add confidence intervals (±0.5%)
            uncertainty = predicted_rate * Decimal('0.005')
            confidence_intervals.append((
                predicted_rate - uncertainty,
                predicted_rate + uncertainty
            ))
        
        return MLPrediction(
            currency_pair="USD/EUR",
            current_rate=current_rate,
            predicted_rates=predicted_rates,
            confidence_intervals=confidence_intervals,
            model_confidence=confidence,
            prediction_horizon_hours=24,
            timestamp=datetime.utcnow()
        )
    
    def create_sample_decision_context(
        self,
        request: ConversionRequest = None,
        ml_prediction: MLPrediction = None,
        market_rates: Dict[str, Decimal] = None
    ) -> DecisionContext:
        """Create sample decision context for testing."""
        if request is None:
            request = self.create_sample_conversion_request()
        
        if ml_prediction is None:
            ml_prediction = self.create_sample_ml_prediction()
        
        if market_rates is None:
            market_rates = {
                'provider_a': Decimal('0.850'),
                'provider_b': Decimal('0.848'),
                'provider_c': Decimal('0.852')
            }
        
        return DecisionContext(
            request=request,
            current_market_data=market_rates,
            ml_prediction=ml_prediction,
            provider_fees={'default': Decimal('0.01')},
            market_hours=True
        )
    
    @pytest.mark.asyncio
    async def test_convert_now_recommendation(self):
        """Test scenario that should recommend converting now."""
        # Create scenario: rates expected to worsen significantly
        ml_prediction = self.create_sample_ml_prediction(
            expected_change=-2.0,  # Rates expected to drop 2%
            confidence=0.9
        )
        
        # User with approaching deadline
        request = self.create_sample_conversion_request(
            deadline=datetime.utcnow() + timedelta(days=2)
        )
        
        context = self.create_sample_decision_context(
            request=request,
            ml_prediction=ml_prediction
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Assertions
        assert recommendation.decision == DecisionType.CONVERT_NOW
        assert recommendation.confidence > 0.6
        assert ReasonCode.FAVORABLE_PREDICTION in recommendation.reasoning
        assert ReasonCode.DEADLINE_APPROACHING in recommendation.reasoning
        assert recommendation.is_actionable
        assert "convert now" in recommendation.explanation.lower()
    
    @pytest.mark.asyncio
    async def test_wait_recommendation(self):
        """Test scenario that should recommend waiting."""
        # Create scenario: rates expected to improve
        ml_prediction = self.create_sample_ml_prediction(
            expected_change=1.8,  # Rates expected to improve 1.8%
            confidence=0.85
        )
        
        # User with plenty of time
        request = self.create_sample_conversion_request(
            deadline=datetime.utcnow() + timedelta(days=14)
        )
        
        context = self.create_sample_decision_context(
            request=request,
            ml_prediction=ml_prediction
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Assertions
        assert recommendation.decision == DecisionType.WAIT
        assert recommendation.confidence > 0.6
        assert ReasonCode.BETTER_RATES_PREDICTED in recommendation.reasoning
        assert recommendation.suggested_wait_days is not None
        assert recommendation.suggested_wait_days > 0
        assert "wait" in recommendation.explanation.lower()
    
    @pytest.mark.asyncio
    async def test_conservative_user_behavior(self):
        """Test that conservative users get different recommendations."""
        # Conservative user profile
        conservative_profile = self.create_sample_user_profile(
            risk_tolerance=RiskTolerance.CONSERVATIVE
        )
        
        # Uncertain ML prediction (moderate change, low confidence)
        ml_prediction = self.create_sample_ml_prediction(
            expected_change=0.8,  # Small expected change
            confidence=0.4        # Low confidence
        )
        
        request = self.create_sample_conversion_request()
        request.user_profile = conservative_profile
        
        context = self.create_sample_decision_context(
            request=request,
            ml_prediction=ml_prediction
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Conservative users should lean towards convert now when uncertain
        assert recommendation.decision == DecisionType.CONVERT_NOW
        assert ReasonCode.LOW_CONFIDENCE in recommendation.reasoning
    
    @pytest.mark.asyncio
    async def test_aggressive_user_behavior(self):
        """Test that aggressive users are more willing to wait."""
        # Aggressive user profile
        aggressive_profile = self.create_sample_user_profile(
            risk_tolerance=RiskTolerance.AGGRESSIVE
        )
        
        # Volatile but potentially profitable scenario
        ml_prediction = self.create_sample_ml_prediction(
            expected_change=2.5,  # High expected improvement
            confidence=0.6        # Moderate confidence
        )
        
        # Add high volatility by widening confidence intervals
        wider_intervals = []
        for lower, upper in ml_prediction.confidence_intervals:
            spread = upper - lower
            center = (upper + lower) / 2
            wider_spread = spread * 3  # 3x wider intervals
            wider_intervals.append((
                center - wider_spread / 2,
                center + wider_spread / 2
            ))
        ml_prediction.confidence_intervals = wider_intervals
        
        request = self.create_sample_conversion_request()
        request.user_profile = aggressive_profile
        
        context = self.create_sample_decision_context(
            request=request,
            ml_prediction=ml_prediction
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Aggressive users should be willing to wait despite volatility
        assert recommendation.decision == DecisionType.WAIT
        assert recommendation.suggested_wait_days is not None
    
    @pytest.mark.asyncio
    async def test_insufficient_data_scenario(self):
        """Test handling of insufficient data scenarios."""
        # Create context with no ML prediction
        request = self.create_sample_conversion_request()
        context = DecisionContext(
            request=request,
            current_market_data={'provider_a': Decimal('0.85')},
            ml_prediction=None,  # No ML data
            market_hours=True
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Assertions
        assert recommendation.decision == DecisionType.INSUFFICIENT_DATA
        assert recommendation.confidence == 0.0
        assert not recommendation.is_actionable
        assert ReasonCode.MISSING_ML_PREDICTION in recommendation.reasoning
        assert "unable to provide" in recommendation.explanation.lower()
    
    @pytest.mark.asyncio
    async def test_cost_analysis_integration(self):
        """Test that cost analysis properly influences decisions."""
        # High fee sensitivity user
        fee_sensitive_profile = self.create_sample_user_profile()
        fee_sensitive_profile.fee_sensitivity = 1.0  # Very sensitive
        
        # High cost conversion scenario
        context = self.create_sample_decision_context()
        context.provider_fees = {'default': Decimal('0.03')}  # 3% fees
        context.request.user_profile = fee_sensitive_profile
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Should consider high costs in reasoning
        assert recommendation.cost_analysis is not None
        assert recommendation.cost_analysis.cost_percentage > 2.0
        
        # May recommend waiting if costs are too high
        if recommendation.decision == DecisionType.WAIT:
            assert ReasonCode.COST_DISADVANTAGE in recommendation.reasoning
    
    @pytest.mark.asyncio
    async def test_risk_analysis_integration(self):
        """Test that risk analysis properly influences decisions."""
        # Create high volatility scenario
        ml_prediction = self.create_sample_ml_prediction(confidence=0.9)
        
        # Create very wide confidence intervals (high volatility)
        wide_intervals = []
        for lower, upper in ml_prediction.confidence_intervals:
            center = (upper + lower) / 2
            wide_spread = center * Decimal('0.05')  # ±5% intervals
            wide_intervals.append((
                center - wide_spread,
                center + wide_spread
            ))
        ml_prediction.confidence_intervals = wide_intervals
        
        context = self.create_sample_decision_context(ml_prediction=ml_prediction)
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Should detect high volatility
        assert recommendation.risk_assessment is not None
        assert recommendation.risk_assessment.volatility_score > 0.5
        assert recommendation.risk_level in ["Moderate", "High"]
    
    @pytest.mark.asyncio
    async def test_expected_savings_calculation(self):
        """Test that expected savings are calculated correctly."""
        # Scenario with clear expected improvement
        ml_prediction = self.create_sample_ml_prediction(
            expected_change=2.0,  # 2% improvement expected
            confidence=0.9
        )
        
        request = self.create_sample_conversion_request(amount=Decimal('10000'))
        context = self.create_sample_decision_context(
            request=request,
            ml_prediction=ml_prediction
        )
        
        # Analyze decision
        recommendation = await self.engine.analyze_conversion_opportunity(context)
        
        # Should calculate expected savings
        if recommendation.decision == DecisionType.WAIT:
            assert recommendation.expected_savings is not None
            assert recommendation.expected_savings > 0
            assert recommendation.expected_savings_percentage is not None
            assert recommendation.expected_savings_percentage > 0
    
    async def test_decision_consistency(self):
        """Test that identical inputs produce consistent decisions."""
        context = self.create_sample_decision_context()
        
        # Make multiple decisions with identical context
        recommendations = []
        for _ in range(5):
            recommendation = await self.engine.analyze_conversion_opportunity(context)
            recommendations.append(recommendation)
        
        # All decisions should be identical
        first_decision = recommendations[0].decision
        for recommendation in recommendations:
            assert recommendation.decision == first_decision
            assert abs(recommendation.confidence - recommendations[0].confidence) < 0.01
    
    def test_market_condition_analysis(self):
        """Test market condition analyzer functionality."""
        # Test market hours detection
        assert isinstance(self.market_analyzer.is_market_hours('USD/EUR'), bool)
        
        # Test market overlap scoring
        overlap_score = self.market_analyzer.get_market_overlap_score('USD/EUR')
        assert 0 <= overlap_score <= 1
    
    async def test_cost_analyzer_functionality(self):
        """Test cost analyzer functionality."""
        context = self.create_sample_decision_context()
        
        # Test comprehensive cost analysis
        cost_analysis = await self.cost_analyzer.analyze_comprehensive_costs(context)
        
        assert cost_analysis.base_rate > 0
        assert cost_analysis.spread_cost >= 0
        assert cost_analysis.provider_fee >= 0
        assert cost_analysis.total_cost > 0
        assert cost_analysis.cost_percentage >= 0
        
        # Test provider comparison
        comparison = self.cost_analyzer.analyze_provider_comparison(context)
        assert len(comparison) == len(context.current_market_data)
        for provider_data in comparison.values():
            assert 'rate' in provider_data
            assert 'total_cost' in provider_data
    
    async def test_risk_analyzer_functionality(self):
        """Test risk analyzer functionality."""
        context = self.create_sample_decision_context()
        
        # Test comprehensive risk analysis
        risk_assessment = await self.risk_analyzer.analyze_comprehensive_risk(context)
        
        assert 0 <= risk_assessment.volatility_score <= 1
        assert 0 <= risk_assessment.prediction_uncertainty <= 1
        assert 0 <= risk_assessment.market_stability <= 1
        assert 0 <= risk_assessment.time_risk <= 1
        assert 0 <= risk_assessment.overall_risk <= 1


# Utility functions for running tests
async def run_sample_decision_analysis():
    """
    Run a sample decision analysis for demonstration purposes.
    
    This function can be used to test the decision engine interactively.
    """
    print("🧠 Running Sample Decision Analysis")
    print("=" * 50)
    
    # Create test engine and data
    engine = DecisionEngine()
    
    # Create sample scenario
    user_profile = UserProfile(
        user_id="demo_user",
        risk_tolerance=RiskTolerance.MODERATE,
        fee_sensitivity=0.6,
        typical_amount_usd=Decimal('5000')
    )
    
    request = ConversionRequest(
        from_currency="USD",
        to_currency="EUR",
        amount=Decimal('5000'),
        user_profile=user_profile,
        deadline=datetime.utcnow() + timedelta(days=5),
        current_rate=Decimal('0.85')
    )
    
    # Create ML prediction (rates expected to improve slightly)
    ml_prediction = MLPrediction(
        currency_pair="USD/EUR",
        current_rate=Decimal('0.85'),
        predicted_rates=[Decimal('0.851'), Decimal('0.852'), Decimal('0.853')],
        confidence_intervals=[(Decimal('0.845'), Decimal('0.855'))],
        model_confidence=0.75,
        prediction_horizon_hours=24,
        timestamp=datetime.utcnow()
    )
    
    # Create decision context
    context = DecisionContext(
        request=request,
        current_market_data={
            'bank_a': Decimal('0.850'),
            'bank_b': Decimal('0.848'),
            'exchange_c': Decimal('0.852')
        },
        ml_prediction=ml_prediction,
        provider_fees={'default': Decimal('0.015')},  # 1.5% fee
        market_hours=True
    )
    
    # Analyze decision
    start_time = datetime.utcnow()
    recommendation = await engine.analyze_conversion_opportunity(context)
    analysis_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Display results
    print(f"📊 Analysis Results (completed in {analysis_time:.3f}s)")
    print(f"   Decision: {recommendation.decision.value}")
    print(f"   Confidence: {recommendation.confidence:.1%}")
    
    if recommendation.expected_savings:
        print(f"   Expected Savings: ${recommendation.expected_savings:.2f}")
    
    if recommendation.suggested_wait_days:
        print(f"   Suggested Wait: {recommendation.suggested_wait_days} days")
    
    print(f"   Risk Level: {recommendation.risk_level}")
    print(f"   Reasoning: {', '.join([r.value for r in recommendation.reasoning])}")
    print(f"\n💡 Explanation: {recommendation.explanation}")
    
    # Display cost breakdown
    if recommendation.cost_analysis:
        cost = recommendation.cost_analysis
        print(f"\n💰 Cost Analysis:")
        print(f"   Spread Cost: ${cost.spread_cost:.2f}")
        print(f"   Provider Fee: ${cost.provider_fee:.2f}")
        print(f"   Total Cost: ${cost.total_cost:.2f} ({cost.cost_percentage:.2f}%)")
    
    # Display risk breakdown
    if recommendation.risk_assessment:
        risk = recommendation.risk_assessment
        print(f"\n⚠️  Risk Assessment:")
        print(f"   Volatility: {risk.volatility_score:.2f}")
        print(f"   Prediction Uncertainty: {risk.prediction_uncertainty:.2f}")
        print(f"   Overall Risk: {risk.overall_risk:.2f}")
    
    return recommendation


if __name__ == "__main__":
    # Run sample analysis
    asyncio.run(run_sample_decision_analysis())