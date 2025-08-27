"""
Cost Optimization Tools for Currency Conversion Analysis.

Provides cost calculation utilities, provider comparison tools, and 
optimization strategies for minimizing currency conversion costs.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderCostProfile:
    """Cost profile for a currency conversion provider."""
    name: str
    base_fee_percentage: float
    spread_bps: float  # Basis points
    processing_fee: float
    minimum_fee: float
    maximum_fee: float
    transfer_speed_hours: int
    reliability_score: float
    geographic_coverage: List[str]


@dataclass 
class ConversionCostBreakdown:
    """Detailed breakdown of conversion costs."""
    provider_name: str
    amount: float
    exchange_rate_used: float
    market_rate: float
    base_fee: float
    spread_cost: float
    processing_fee: float
    total_cost: float
    cost_percentage: float
    effective_rate: float


@dataclass
class CostOptimizationResult:
    """Result from cost optimization analysis."""
    recommended_provider: str
    estimated_savings: float
    total_cost: float
    cost_breakdown: ConversionCostBreakdown
    alternative_options: List[ConversionCostBreakdown]
    confidence: float


class ProviderCostAnalyzer:
    """Analyzes and compares costs across different currency providers."""
    
    def __init__(self):
        self.provider_profiles = self._initialize_provider_profiles()
    
    def _initialize_provider_profiles(self) -> Dict[str, ProviderCostProfile]:
        """Initialize known provider cost profiles."""
        return {
            "wise": ProviderCostProfile(
                name="Wise",
                base_fee_percentage=0.35,  # 0.35% base fee
                spread_bps=15,  # 15 basis points spread
                processing_fee=0.0,
                minimum_fee=0.5,
                maximum_fee=500.0,
                transfer_speed_hours=24,
                reliability_score=0.95,
                geographic_coverage=["USD", "EUR", "GBP", "CAD", "AUD"]
            ),
            "remitly": ProviderCostProfile(
                name="Remitly",
                base_fee_percentage=0.5,
                spread_bps=25,
                processing_fee=2.99,
                minimum_fee=2.99,
                maximum_fee=999.0,
                transfer_speed_hours=48,
                reliability_score=0.9,
                geographic_coverage=["USD", "EUR", "GBP", "INR", "PHP"]
            ),
            "traditional_bank": ProviderCostProfile(
                name="Traditional Bank",
                base_fee_percentage=1.0,
                spread_bps=50,
                processing_fee=15.0,
                minimum_fee=15.0,
                maximum_fee=9999.0,
                transfer_speed_hours=72,
                reliability_score=0.99,
                geographic_coverage=["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
            ),
            "revolut": ProviderCostProfile(
                name="Revolut",
                base_fee_percentage=0.0,  # Free up to certain limit
                spread_bps=20,
                processing_fee=0.0,
                minimum_fee=0.0,
                maximum_fee=1000.0,  # Free limit
                transfer_speed_hours=12,
                reliability_score=0.85,
                geographic_coverage=["USD", "EUR", "GBP", "CHF", "JPY"]
            ),
            "paypal": ProviderCostProfile(
                name="PayPal",
                base_fee_percentage=2.5,
                spread_bps=40,
                processing_fee=0.0,
                minimum_fee=0.99,
                maximum_fee=4999.0,
                transfer_speed_hours=1,
                reliability_score=0.8,
                geographic_coverage=["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]
            )
        }
    
    def calculate_provider_cost(self, 
                              provider_name: str,
                              amount: float,
                              exchange_rate: float,
                              market_rate: float) -> ConversionCostBreakdown:
        """
        Calculate detailed cost breakdown for a specific provider.
        
        Args:
            provider_name: Name of the provider
            amount: Conversion amount
            exchange_rate: Provider's offered rate
            market_rate: Current market rate
            
        Returns:
            Detailed cost breakdown
        """
        # Get provider profile
        profile = self.provider_profiles.get(
            provider_name.lower().replace(" ", "_"),
            self._get_default_provider_profile(provider_name)
        )
        
        # Calculate individual cost components
        base_fee = max(
            profile.minimum_fee,
            min(profile.maximum_fee, amount * (profile.base_fee_percentage / 100))
        )
        
        # Spread cost (difference between market and offered rate)
        spread_cost = amount * abs(exchange_rate - market_rate)
        
        # Processing fee
        processing_fee = profile.processing_fee
        
        # Total cost
        total_cost = base_fee + spread_cost + processing_fee
        cost_percentage = (total_cost / amount) * 100
        
        # Effective rate after all costs
        effective_amount = amount - total_cost
        effective_rate = exchange_rate * (effective_amount / amount)
        
        return ConversionCostBreakdown(
            provider_name=profile.name,
            amount=amount,
            exchange_rate_used=exchange_rate,
            market_rate=market_rate,
            base_fee=base_fee,
            spread_cost=spread_cost,
            processing_fee=processing_fee,
            total_cost=total_cost,
            cost_percentage=cost_percentage,
            effective_rate=effective_rate
        )
    
    def _get_default_provider_profile(self, provider_name: str) -> ProviderCostProfile:
        """Create default profile for unknown providers."""
        return ProviderCostProfile(
            name=provider_name,
            base_fee_percentage=1.5,  # 1.5% default
            spread_bps=30,  # 30 bps default
            processing_fee=5.0,
            minimum_fee=2.0,
            maximum_fee=1000.0,
            transfer_speed_hours=48,
            reliability_score=0.7,
            geographic_coverage=["USD", "EUR"]
        )
    
    def find_optimal_provider(self, 
                            currency_pair: str,
                            amount: float,
                            market_rate: float,
                            user_priorities: Optional[Dict[str, float]] = None) -> CostOptimizationResult:
        """
        Find the optimal provider based on cost and user priorities.
        
        Args:
            currency_pair: Currency pair for conversion
            amount: Conversion amount
            market_rate: Current market rate
            user_priorities: User preferences (cost_weight, speed_weight, reliability_weight)
            
        Returns:
            Optimization result with best provider recommendation
        """
        if user_priorities is None:
            user_priorities = {"cost_weight": 0.7, "speed_weight": 0.2, "reliability_weight": 0.1}
        
        # Check currency pair support
        currencies = currency_pair.split('/')
        supported_providers = [
            name for name, profile in self.provider_profiles.items()
            if all(currency in profile.geographic_coverage for currency in currencies)
        ]
        
        if not supported_providers:
            supported_providers = list(self.provider_profiles.keys())  # Use all as fallback
        
        # Calculate costs and scores for each provider
        provider_analyses = []
        for provider_name in supported_providers:
            profile = self.provider_profiles[provider_name]
            
            # Estimate exchange rate (market rate - spread)
            spread_factor = profile.spread_bps / 10000  # Convert bps to decimal
            estimated_rate = market_rate * (1 - spread_factor)
            
            # Calculate cost breakdown
            cost_breakdown = self.calculate_provider_cost(
                profile.name, amount, estimated_rate, market_rate
            )
            
            # Calculate composite score
            cost_score = 1.0 - min(1.0, cost_breakdown.cost_percentage / 5.0)  # Normalize to 0-1
            speed_score = 1.0 - min(1.0, profile.transfer_speed_hours / 168.0)  # Weekly max
            reliability_score = profile.reliability_score
            
            composite_score = (
                cost_score * user_priorities["cost_weight"] +
                speed_score * user_priorities["speed_weight"] + 
                reliability_score * user_priorities["reliability_weight"]
            )
            
            provider_analyses.append({
                'provider': profile,
                'cost_breakdown': cost_breakdown,
                'composite_score': composite_score,
                'individual_scores': {
                    'cost': cost_score,
                    'speed': speed_score,
                    'reliability': reliability_score
                }
            })
        
        # Sort by composite score
        provider_analyses.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Get best provider
        best_analysis = provider_analyses[0]
        best_provider = best_analysis['provider'].name
        best_cost_breakdown = best_analysis['cost_breakdown']
        
        # Calculate savings vs worst option
        worst_cost = max(analysis['cost_breakdown'].total_cost for analysis in provider_analyses)
        estimated_savings = worst_cost - best_cost_breakdown.total_cost
        
        return CostOptimizationResult(
            recommended_provider=best_provider,
            estimated_savings=estimated_savings,
            total_cost=best_cost_breakdown.total_cost,
            cost_breakdown=best_cost_breakdown,
            alternative_options=[analysis['cost_breakdown'] for analysis in provider_analyses[1:3]],
            confidence=0.8
        )


class FeeCalculator:
    """Calculates various types of fees and costs for currency conversions."""
    
    @staticmethod
    def calculate_spread_cost(amount: float, 
                            offered_rate: float,
                            market_rate: float) -> float:
        """Calculate cost of spread between market and offered rate."""
        spread_percentage = abs(offered_rate - market_rate) / market_rate
        return amount * spread_percentage
    
    @staticmethod
    def calculate_percentage_fee(amount: float, 
                               fee_percentage: float,
                               minimum_fee: float = 0.0,
                               maximum_fee: float = float('inf')) -> float:
        """Calculate percentage-based fee with min/max bounds."""
        fee = amount * (fee_percentage / 100)
        return max(minimum_fee, min(maximum_fee, fee))
    
    @staticmethod
    def calculate_tiered_fee(amount: float, 
                           tier_structure: List[Dict[str, Any]]) -> float:
        """
        Calculate fee based on tiered structure.
        
        Args:
            amount: Conversion amount
            tier_structure: List of tier dicts with 'threshold', 'rate' keys
            
        Returns:
            Calculated tiered fee
        """
        # Sort tiers by threshold
        tiers = sorted(tier_structure, key=lambda x: x['threshold'])
        
        total_fee = 0.0
        remaining_amount = amount
        
        for i, tier in enumerate(tiers):
            tier_threshold = tier['threshold']
            tier_rate = tier['rate'] / 100  # Convert percentage to decimal
            
            if remaining_amount <= 0:
                break
            
            # Calculate amount in this tier
            if i == len(tiers) - 1:  # Last tier
                tier_amount = remaining_amount
            else:
                next_threshold = tiers[i + 1]['threshold']
                tier_amount = min(remaining_amount, next_threshold - tier_threshold)
            
            # Calculate fee for this tier
            tier_fee = tier_amount * tier_rate
            total_fee += tier_fee
            remaining_amount -= tier_amount
        
        return total_fee
    
    @staticmethod
    def estimate_opportunity_cost(amount: float,
                                expected_return: float,
                                wait_days: int,
                                risk_free_rate: float = 0.02) -> float:
        """
        Estimate opportunity cost of waiting to convert.
        
        Args:
            amount: Conversion amount
            expected_return: Expected return from waiting (annual %)
            wait_days: Number of days to wait
            risk_free_rate: Risk-free rate for comparison
            
        Returns:
            Estimated opportunity cost
        """
        # Daily rates
        daily_expected = expected_return / 365
        daily_risk_free = risk_free_rate / 365
        
        # Net opportunity cost per day
        daily_opportunity_cost = amount * (daily_risk_free - daily_expected)
        
        return max(0.0, daily_opportunity_cost * wait_days)


class TimingOptimizer:
    """Optimizes conversion timing to minimize costs."""
    
    def __init__(self):
        self.fee_calculator = FeeCalculator()
    
    def analyze_optimal_timing(self,
                             currency_pair: str,
                             amount: float,
                             current_rate: float,
                             predicted_rates: Dict[int, float],
                             cost_profile: ProviderCostProfile) -> Dict[str, Any]:
        """
        Analyze optimal timing for conversion based on cost minimization.
        
        Args:
            currency_pair: Currency pair to convert
            amount: Conversion amount
            current_rate: Current exchange rate
            predicted_rates: Dict of {days_ahead: predicted_rate}
            cost_profile: Provider cost structure
            
        Returns:
            Timing optimization analysis
        """
        timing_analyses = []
        
        # Analyze immediate conversion
        immediate_cost = self._calculate_total_cost(
            amount, current_rate, current_rate, cost_profile, 0
        )
        
        timing_analyses.append({
            'days_to_wait': 0,
            'predicted_rate': current_rate,
            'total_cost': immediate_cost['total_cost'],
            'opportunity_cost': 0.0,
            'net_position': immediate_cost['total_cost']
        })
        
        # Analyze waiting scenarios
        for days, predicted_rate in predicted_rates.items():
            if days <= 30:  # Limit analysis to reasonable timeframe
                conversion_cost = self._calculate_total_cost(
                    amount, predicted_rate, current_rate, cost_profile, 0
                )
                
                opportunity_cost = self.fee_calculator.estimate_opportunity_cost(
                    amount, 0.02, days  # Assume 2% risk-free rate
                )
                
                net_position = conversion_cost['total_cost'] + opportunity_cost
                
                timing_analyses.append({
                    'days_to_wait': days,
                    'predicted_rate': predicted_rate,
                    'total_cost': conversion_cost['total_cost'],
                    'opportunity_cost': opportunity_cost,
                    'net_position': net_position
                })
        
        # Find optimal timing
        optimal = min(timing_analyses, key=lambda x: x['net_position'])
        savings_vs_immediate = timing_analyses[0]['net_position'] - optimal['net_position']
        
        return {
            'optimal_days_to_wait': optimal['days_to_wait'],
            'optimal_rate': optimal['predicted_rate'],
            'estimated_savings': savings_vs_immediate,
            'immediate_cost': timing_analyses[0]['total_cost'],
            'optimal_cost': optimal['total_cost'],
            'all_scenarios': timing_analyses,
            'recommendation': 'wait' if optimal['days_to_wait'] > 0 else 'convert_immediately',
            'confidence': 0.7
        }
    
    def _calculate_total_cost(self,
                            amount: float,
                            exchange_rate: float,
                            market_rate: float,
                            profile: ProviderCostProfile,
                            wait_days: int) -> Dict[str, float]:
        """Calculate total cost for a specific scenario."""
        # Base fee
        base_fee = self.fee_calculator.calculate_percentage_fee(
            amount, profile.base_fee_percentage, profile.minimum_fee, profile.maximum_fee
        )
        
        # Spread cost
        spread_cost = self.fee_calculator.calculate_spread_cost(
            amount, exchange_rate, market_rate
        )
        
        # Processing fee
        processing_fee = profile.processing_fee
        
        # Total cost
        total_cost = base_fee + spread_cost + processing_fee
        
        return {
            'base_fee': base_fee,
            'spread_cost': spread_cost,
            'processing_fee': processing_fee,
            'total_cost': total_cost,
            'cost_percentage': (total_cost / amount) * 100
        }


class MarketTimingAnalyzer:
    """Analyzes market timing factors that affect conversion costs."""
    
    def analyze_market_timing_costs(self,
                                  currency_pair: str,
                                  amount: float,
                                  volatility: float,
                                  trend_direction: str,
                                  time_horizon_days: int = 7) -> Dict[str, Any]:
        """
        Analyze how market timing affects conversion costs.
        
        Args:
            currency_pair: Currency pair to analyze
            amount: Conversion amount
            volatility: Market volatility estimate
            trend_direction: Market trend (up/down/sideways)
            time_horizon_days: Analysis time horizon
            
        Returns:
            Market timing cost analysis
        """
        # Calculate volatility impact on costs
        volatility_cost_impact = min(0.02, volatility * 0.1)  # Up to 2% additional cost
        
        # Calculate trend impact
        trend_multipliers = {
            'trending_up': 1.2,    # Higher costs in uptrend
            'trending_down': 0.8,  # Lower costs in downtrend
            'ranging': 1.0,        # Neutral impact
            'volatile': 1.3        # Higher costs due to uncertainty
        }
        
        trend_multiplier = trend_multipliers.get(trend_direction.lower(), 1.0)
        
        # Calculate timing cost factors
        base_spread_cost = amount * 0.005  # 0.5% base spread
        volatility_adjusted_spread = base_spread_cost * (1 + volatility_cost_impact)
        trend_adjusted_cost = volatility_adjusted_spread * trend_multiplier
        
        # Calculate optimal timing window
        if volatility > 0.03:  # High volatility
            optimal_window = "wait_for_stability"
            wait_benefit = amount * 0.003  # 0.3% potential benefit
        elif trend_direction == 'trending_up':
            optimal_window = "convert_soon"
            wait_benefit = -amount * 0.002  # Cost of waiting in uptrend
        else:
            optimal_window = "flexible_timing"
            wait_benefit = amount * 0.001  # Small benefit from flexibility
        
        return {
            'volatility_cost_impact': volatility_cost_impact,
            'trend_cost_multiplier': trend_multiplier,
            'estimated_spread_cost': trend_adjusted_cost,
            'optimal_timing_window': optimal_window,
            'wait_benefit_estimate': wait_benefit,
            'cost_uncertainty': volatility * 0.5,  # How uncertain our cost estimates are
            'timing_sensitivity': volatility + (0.3 if trend_direction in ['trending_up', 'trending_down'] else 0.1),
            'confidence': 0.7
        }


class CostOptimizationToolkit:
    """Main toolkit for cost optimization analysis."""
    
    def __init__(self):
        self.provider_analyzer = ProviderCostAnalyzer()
        self.timing_optimizer = TimingOptimizer()
        self.market_timing_analyzer = MarketTimingAnalyzer()
        self.fee_calculator = FeeCalculator()
    
    async def comprehensive_cost_analysis(self,
                                        currency_pair: str,
                                        amount: float,
                                        available_providers: List[str],
                                        market_rate: float,
                                        market_context: Optional[Dict[str, Any]] = None,
                                        user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cost optimization analysis.
        
        Args:
            currency_pair: Currency pair for conversion
            amount: Conversion amount
            available_providers: List of available provider names
            market_rate: Current market rate
            market_context: Market analysis context
            user_preferences: User preferences and priorities
            
        Returns:
            Comprehensive cost optimization results
        """
        analysis_results = {}
        
        # Provider comparison analysis
        if available_providers:
            provider_costs = []
            for provider_name in available_providers:
                # Estimate provider rate (market rate minus spread)
                profile = self.provider_analyzer.provider_profiles.get(
                    provider_name.lower().replace(" ", "_"),
                    self.provider_analyzer._get_default_provider_profile(provider_name)
                )
                
                spread_factor = profile.spread_bps / 10000
                estimated_rate = market_rate * (1 - spread_factor)
                
                cost_breakdown = self.provider_analyzer.calculate_provider_cost(
                    provider_name, amount, estimated_rate, market_rate
                )
                provider_costs.append(cost_breakdown)
            
            # Find best provider
            best_provider = min(provider_costs, key=lambda x: x.total_cost)
            worst_provider = max(provider_costs, key=lambda x: x.total_cost)
            potential_savings = worst_provider.total_cost - best_provider.total_cost
            
            analysis_results["provider_analysis"] = {
                "best_provider": best_provider.provider_name,
                "best_cost": best_provider.total_cost,
                "best_percentage": best_provider.cost_percentage,
                "potential_savings": potential_savings,
                "provider_breakdown": {
                    cost.provider_name: {
                        "total_cost": cost.total_cost,
                        "base_fee": cost.base_fee,
                        "spread_cost": cost.spread_cost,
                        "processing_fee": cost.processing_fee,
                        "percentage": cost.cost_percentage,
                        "effective_rate": cost.effective_rate
                    }
                    for cost in provider_costs
                }
            }
        else:
            analysis_results["provider_analysis"] = self._default_provider_analysis(amount)
        
        # Market timing analysis
        if market_context:
            volatility = market_context.get('volatility_score', 0.5) * 0.3  # Scale to realistic volatility
            trend = market_context.get('market_regime', 'ranging')
            
            timing_analysis = self.market_timing_analyzer.analyze_market_timing_costs(
                currency_pair, amount, volatility, trend
            )
            analysis_results["timing_analysis"] = timing_analysis
        else:
            analysis_results["timing_analysis"] = {
                "optimal_timing_window": "flexible_timing",
                "cost_uncertainty": 0.01,
                "timing_sensitivity": 0.3,
                "confidence": 0.5
            }
        
        # Calculate overall recommendations
        best_provider = analysis_results["provider_analysis"]["best_provider"]
        timing_window = analysis_results["timing_analysis"]["optimal_timing_window"]
        
        # Overall confidence
        provider_conf = 0.8 if available_providers else 0.3
        timing_conf = analysis_results["timing_analysis"]["confidence"]
        overall_confidence = (provider_conf + timing_conf) / 2
        
        return {
            "currency_pair": currency_pair,
            "amount": amount,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_confidence": overall_confidence,
            "components": analysis_results,
            "summary": {
                "recommended_provider": best_provider,
                "estimated_total_cost": analysis_results["provider_analysis"]["best_cost"],
                "potential_savings": analysis_results["provider_analysis"]["potential_savings"],
                "timing_recommendation": timing_window,
                "cost_as_percentage": analysis_results["provider_analysis"]["best_percentage"]
            }
        }
    
    def _default_provider_analysis(self, amount: float) -> Dict[str, Any]:
        """Default provider analysis when no providers specified."""
        default_cost = amount * 0.015  # 1.5% default cost
        return {
            "best_provider": "Online Service",
            "best_cost": default_cost,
            "best_percentage": 1.5,
            "potential_savings": amount * 0.005,  # 0.5% potential savings
            "provider_breakdown": {
                "Online Service": {
                    "total_cost": default_cost,
                    "base_fee": amount * 0.005,
                    "spread_cost": amount * 0.008,
                    "processing_fee": amount * 0.002,
                    "percentage": 1.5,
                    "effective_rate": 0.0
                }
            }
        }


# Convenience functions
def create_cost_optimization_toolkit() -> CostOptimizationToolkit:
    """Factory function to create cost optimization toolkit."""
    return CostOptimizationToolkit()


def mock_cost_data_for_testing(currency_pair: str, amount: float) -> Dict[str, Any]:
    """Generate mock cost data for testing purposes."""
    import random
    
    # Mock available providers
    available_providers = ["Wise", "Remitly", "Traditional Bank", "Revolut"]
    
    # Mock market rate
    market_rate = 0.85 if currency_pair == "USD/EUR" else 1.0
    
    # Mock market context
    market_context = {
        "volatility_score": random.uniform(0.2, 0.8),
        "market_regime": random.choice(["trending_up", "ranging", "volatile", "trending_down"]),
        "sentiment_score": random.uniform(-0.3, 0.3)
    }
    
    # Mock user preferences
    user_preferences = {
        "cost_weight": 0.7,
        "speed_weight": 0.2,
        "reliability_weight": 0.1,
        "fee_sensitivity": random.uniform(0.3, 0.9)
    }
    
    return {
        "currency_pair": currency_pair,
        "amount": amount,
        "available_providers": available_providers,
        "market_rate": market_rate,
        "market_context": market_context,
        "user_preferences": user_preferences
    }