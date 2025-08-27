"""
Risk Analysis Tools for Currency Conversion Decisions.

Provides statistical models, volatility calculations, and risk metrics
to support comprehensive risk assessment for currency conversions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Volatility analysis results."""
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float  # Where current volatility ranks historically
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float


@dataclass
class ScenarioResult:
    """Risk scenario analysis result."""
    name: str
    probability: float
    expected_return: float
    value_at_risk: float
    description: str


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position."""
    value_at_risk_1d: float
    value_at_risk_7d: float
    expected_shortfall: float
    sharpe_ratio_estimate: float
    maximum_drawdown: float
    confidence_interval_95: Tuple[float, float]


class VolatilityCalculator:
    """Calculates various volatility metrics for risk analysis."""
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize volatility calculator.
        
        Args:
            lookback_days: Number of days to use for historical analysis
        """
        self.lookback_days = lookback_days
    
    def calculate_historical_volatility(self, prices: List[float], 
                                      window_days: int = 30) -> VolatilityMetrics:
        """
        Calculate historical volatility and trend analysis.
        
        Args:
            prices: Historical price data
            window_days: Window size for volatility calculation
            
        Returns:
            Comprehensive volatility metrics
        """
        if len(prices) < window_days:
            return VolatilityMetrics(
                current_volatility=0.0,
                historical_volatility=0.0,
                volatility_percentile=0.5,
                trend="stable",
                confidence=0.0
            )
        
        # Calculate returns
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        
        # Current volatility (last window_days)
        recent_returns = returns[-window_days:]
        current_vol = self._calculate_volatility(recent_returns)
        
        # Historical volatility (rolling windows)
        historical_vols = []
        for i in range(window_days, len(returns)):
            window_returns = returns[i-window_days:i]
            vol = self._calculate_volatility(window_returns)
            historical_vols.append(vol)
        
        historical_avg = statistics.mean(historical_vols) if historical_vols else current_vol
        
        # Volatility percentile
        percentile = (sum(1 for v in historical_vols if v < current_vol) / 
                     len(historical_vols)) if historical_vols else 0.5
        
        # Volatility trend
        if len(historical_vols) >= 10:
            recent_trend = historical_vols[-5:]
            older_trend = historical_vols[-10:-5]
            recent_avg = statistics.mean(recent_trend)
            older_avg = statistics.mean(older_trend)
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        confidence = min(1.0, len(historical_vols) / 100.0)  # More data = higher confidence
        
        return VolatilityMetrics(
            current_volatility=current_vol,
            historical_volatility=historical_avg,
            volatility_percentile=percentile,
            trend=trend,
            confidence=confidence
        )
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility from returns."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        mean_return = statistics.mean(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        daily_vol = variance ** 0.5
        
        # Annualize (252 trading days)
        annual_vol = daily_vol * (252 ** 0.5)
        
        return annual_vol
    
    def estimate_future_volatility(self, prices: List[float], 
                                 forecast_days: int = 7) -> Dict[str, Any]:
        """
        Estimate volatility for the forecast period.
        
        Args:
            prices: Historical price data
            forecast_days: Number of days to forecast
            
        Returns:
            Volatility forecast with confidence bounds
        """
        if len(prices) < 30:
            return {"forecast_volatility": 0.02, "confidence": 0.0, "upper_bound": 0.04, "lower_bound": 0.01}
        
        # Calculate recent volatility trends
        vol_metrics = self.calculate_historical_volatility(prices, 30)
        current_vol = vol_metrics.current_volatility / (252 ** 0.5)  # Convert to daily
        
        # Simple volatility persistence model
        # Assume volatility mean-reverts to historical average
        historical_vol = vol_metrics.historical_volatility / (252 ** 0.5)
        persistence_factor = 0.7  # How much current volatility persists
        
        forecast_vol = (persistence_factor * current_vol + 
                       (1 - persistence_factor) * historical_vol)
        
        # Add uncertainty bounds
        vol_uncertainty = 0.3  # 30% uncertainty in volatility forecast
        upper_bound = forecast_vol * (1 + vol_uncertainty)
        lower_bound = forecast_vol * (1 - vol_uncertainty)
        
        return {
            "forecast_volatility": forecast_vol,
            "confidence": vol_metrics.confidence,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "forecast_days": forecast_days
        }


class RiskMetricsCalculator:
    """Calculates comprehensive risk metrics for currency positions."""
    
    def calculate_value_at_risk(self, amount: float, 
                              volatility: float,
                              confidence_level: float = 0.95,
                              time_horizon_days: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) for a currency position.
        
        Args:
            amount: Position size in base currency
            volatility: Daily volatility estimate
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
            time_horizon_days: Time horizon in days
            
        Returns:
            VaR estimate in base currency
        """
        # Z-score for the confidence level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(confidence_level, 1.65)
        
        # Scale volatility by time horizon
        scaled_volatility = volatility * (time_horizon_days ** 0.5)
        
        # Calculate VaR
        var = amount * z_score * scaled_volatility
        
        return var
    
    def calculate_expected_shortfall(self, amount: float,
                                   volatility: float,
                                   confidence_level: float = 0.95,
                                   time_horizon_days: int = 1) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            amount: Position size
            volatility: Daily volatility estimate
            confidence_level: Confidence level
            time_horizon_days: Time horizon in days
            
        Returns:
            Expected Shortfall estimate
        """
        # Approximate ES as 1.3 times VaR for normal distribution
        var = self.calculate_value_at_risk(amount, volatility, confidence_level, time_horizon_days)
        expected_shortfall = var * 1.3
        
        return expected_shortfall
    
    def estimate_sharpe_ratio(self, expected_return: float, 
                            volatility: float,
                            risk_free_rate: float = 0.02) -> float:
        """
        Estimate Sharpe ratio for the currency position.
        
        Args:
            expected_return: Expected annual return
            volatility: Annual volatility
            risk_free_rate: Risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio estimate
        """
        if volatility == 0:
            return 0.0
        
        excess_return = expected_return - risk_free_rate
        sharpe_ratio = excess_return / volatility
        
        return sharpe_ratio
    
    def calculate_comprehensive_metrics(self, 
                                      amount: float,
                                      prices: List[float],
                                      expected_return: Optional[float] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a currency position.
        
        Args:
            amount: Position size
            prices: Historical price data
            expected_return: Expected annual return (optional)
            
        Returns:
            Comprehensive risk metrics
        """
        if len(prices) < 10:
            return self._default_risk_metrics()
        
        # Calculate volatility
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        daily_vol = statistics.stdev(returns) if len(returns) > 1 else 0.01
        
        # Calculate VaR for different horizons
        var_1d = self.calculate_value_at_risk(amount, daily_vol, 0.95, 1)
        var_7d = self.calculate_value_at_risk(amount, daily_vol, 0.95, 7)
        
        # Calculate Expected Shortfall
        es = self.calculate_expected_shortfall(amount, daily_vol, 0.95, 1)
        
        # Calculate maximum drawdown
        max_dd = self._calculate_max_drawdown(prices)
        
        # Estimate Sharpe ratio
        if expected_return is None:
            expected_return = statistics.mean(returns) * 252 if returns else 0.0
        
        sharpe = self.estimate_sharpe_ratio(expected_return, daily_vol * (252 ** 0.5))
        
        # Calculate confidence interval for returns
        std_error = daily_vol / (len(returns) ** 0.5) if returns else 0.01
        ci_lower = expected_return - 1.96 * std_error
        ci_upper = expected_return + 1.96 * std_error
        
        return RiskMetrics(
            value_at_risk_1d=var_1d,
            value_at_risk_7d=var_7d,
            expected_shortfall=es,
            sharpe_ratio_estimate=sharpe,
            maximum_drawdown=max_dd,
            confidence_interval_95=(ci_lower, ci_upper)
        )
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown from price series."""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _default_risk_metrics(self) -> RiskMetrics:
        """Default risk metrics when insufficient data."""
        return RiskMetrics(
            value_at_risk_1d=0.0,
            value_at_risk_7d=0.0,
            expected_shortfall=0.0,
            sharpe_ratio_estimate=0.0,
            maximum_drawdown=0.0,
            confidence_interval_95=(0.0, 0.0)
        )


class ScenarioAnalyzer:
    """Analyzes different market scenarios and their risk implications."""
    
    def __init__(self):
        self.risk_calculator = RiskMetricsCalculator()
    
    def generate_scenarios(self, 
                          current_price: float,
                          volatility: float,
                          timeframe_days: int = 7,
                          num_scenarios: int = 3) -> List[ScenarioResult]:
        """
        Generate risk scenarios for currency conversion.
        
        Args:
            current_price: Current exchange rate
            volatility: Expected volatility
            timeframe_days: Analysis timeframe
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario results
        """
        scenarios = []
        
        # Calculate price movements for different scenarios
        daily_vol = volatility / (252 ** 0.5)  # Convert annual to daily
        time_vol = daily_vol * (timeframe_days ** 0.5)
        
        # Best case (95th percentile)
        best_case_change = 1.65 * time_vol  # 95% confidence interval
        best_case_price = current_price * (1 + best_case_change)
        
        scenarios.append(ScenarioResult(
            name="best_case",
            probability=0.15,
            expected_return=best_case_change,
            value_at_risk=0.0,  # No risk in best case
            description=f"Favorable market movement to {best_case_price:.4f}"
        ))
        
        # Most likely (median expectation)
        most_likely_change = 0.0  # Assume no drift for simplicity
        most_likely_price = current_price
        
        scenarios.append(ScenarioResult(
            name="most_likely",
            probability=0.70,
            expected_return=most_likely_change,
            value_at_risk=time_vol * current_price,  # 1-sigma risk
            description=f"Market stays near current level around {most_likely_price:.4f}"
        ))
        
        # Worst case (5th percentile)
        worst_case_change = -1.65 * time_vol
        worst_case_price = current_price * (1 + worst_case_change)
        
        scenarios.append(ScenarioResult(
            name="worst_case", 
            probability=0.15,
            expected_return=worst_case_change,
            value_at_risk=abs(worst_case_change) * current_price,
            description=f"Adverse market movement to {worst_case_price:.4f}"
        ))
        
        return scenarios
    
    def monte_carlo_simulation(self, 
                              current_price: float,
                              volatility: float,
                              timeframe_days: int = 7,
                              num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for price scenarios.
        
        Args:
            current_price: Starting price
            volatility: Annual volatility
            timeframe_days: Simulation timeframe
            num_simulations: Number of simulation paths
            
        Returns:
            Monte Carlo simulation results
        """
        try:
            import random
            
            daily_vol = volatility / (252 ** 0.5)
            simulation_results = []
            
            for _ in range(num_simulations):
                # Generate random path
                price = current_price
                for day in range(timeframe_days):
                    daily_return = random.normalvariate(0, daily_vol)
                    price *= (1 + daily_return)
                
                final_return = (price / current_price) - 1
                simulation_results.append(final_return)
            
            # Calculate statistics
            simulation_results.sort()
            
            return {
                "mean_return": statistics.mean(simulation_results),
                "volatility": statistics.stdev(simulation_results),
                "percentile_5": simulation_results[int(0.05 * num_simulations)],
                "percentile_25": simulation_results[int(0.25 * num_simulations)],
                "percentile_75": simulation_results[int(0.75 * num_simulations)],
                "percentile_95": simulation_results[int(0.95 * num_simulations)],
                "probability_loss": sum(1 for r in simulation_results if r < 0) / num_simulations,
                "probability_large_loss": sum(1 for r in simulation_results if r < -0.02) / num_simulations,
                "num_simulations": num_simulations,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return self._default_simulation_result()
    
    def _default_simulation_result(self) -> Dict[str, Any]:
        """Default simulation results when calculation fails."""
        return {
            "mean_return": 0.0,
            "volatility": 0.02,
            "percentile_5": -0.02,
            "percentile_25": -0.01,
            "percentile_75": 0.01,
            "percentile_95": 0.02,
            "probability_loss": 0.5,
            "probability_large_loss": 0.05,
            "num_simulations": 0,
            "confidence": 0.0
        }


class TimeRiskAssessor:
    """Assesses time-related risks for currency conversions."""
    
    def evaluate_deadline_pressure(self, 
                                  deadline_days: Optional[int],
                                  optimal_timing_days: int = 14) -> Dict[str, Any]:
        """
        Evaluate time pressure based on deadline constraints.
        
        Args:
            deadline_days: Days until conversion deadline (None if no deadline)
            optimal_timing_days: Optimal window for conversion decisions
            
        Returns:
            Time pressure analysis
        """
        if deadline_days is None:
            return {
                "time_pressure": 0.0,
                "flexibility": 1.0,
                "recommended_action": "wait_for_optimal_timing",
                "reasoning": "No deadline pressure allows for optimal timing"
            }
        
        # Calculate time pressure
        if deadline_days <= 1:
            time_pressure = 1.0
            flexibility = 0.0
            action = "convert_immediately"
        elif deadline_days <= 3:
            time_pressure = 0.8
            flexibility = 0.2
            action = "convert_soon"
        elif deadline_days <= 7:
            time_pressure = 0.5
            flexibility = 0.5
            action = "monitor_closely"
        elif deadline_days <= optimal_timing_days:
            time_pressure = 0.3
            flexibility = 0.7
            action = "wait_for_favorable_conditions"
        else:
            time_pressure = 0.1
            flexibility = 0.9
            action = "wait_for_optimal_timing"
        
        return {
            "time_pressure": time_pressure,
            "flexibility": flexibility,
            "recommended_action": action,
            "days_until_deadline": deadline_days,
            "reasoning": f"Time pressure {time_pressure:.1f} with {deadline_days} days remaining"
        }
    
    def calculate_opportunity_cost(self, 
                                 expected_move: float,
                                 volatility: float,
                                 wait_days: int) -> Dict[str, Any]:
        """
        Calculate opportunity cost of waiting vs converting now.
        
        Args:
            expected_move: Expected price movement (%)
            volatility: Price volatility
            wait_days: Days to wait
            
        Returns:
            Opportunity cost analysis
        """
        # Calculate potential gains/losses from waiting
        daily_vol = volatility / (252 ** 0.5)
        wait_vol = daily_vol * (wait_days ** 0.5)
        
        # Probability-weighted outcomes
        prob_favorable = 0.5 + (expected_move / (2 * wait_vol)) if wait_vol > 0 else 0.5
        prob_favorable = max(0.1, min(0.9, prob_favorable))  # Bound probabilities
        
        expected_gain = prob_favorable * abs(expected_move)
        expected_loss = (1 - prob_favorable) * abs(expected_move)
        
        net_expected = expected_gain - expected_loss
        
        return {
            "probability_favorable": prob_favorable,
            "expected_gain_percent": expected_gain,
            "expected_loss_percent": expected_loss,
            "net_expected_return": net_expected,
            "opportunity_cost": max(0, -net_expected),
            "wait_days": wait_days,
            "recommendation": "wait" if net_expected > 0 else "convert_now"
        }


class UserRiskProfiler:
    """Analyzes user risk tolerance and provides personalized risk assessments."""
    
    RISK_TOLERANCE_THRESHOLDS = {
        'conservative': {'max_volatility': 0.05, 'max_drawdown': 0.02, 'var_tolerance': 0.01},
        'moderate': {'max_volatility': 0.15, 'max_drawdown': 0.05, 'var_tolerance': 0.03},
        'aggressive': {'max_volatility': 0.30, 'max_drawdown': 0.10, 'var_tolerance': 0.07}
    }
    
    def assess_risk_compatibility(self, 
                                user_risk_tolerance: str,
                                market_volatility: float,
                                position_size: float,
                                var_estimate: float) -> Dict[str, Any]:
        """
        Assess compatibility between user risk tolerance and market conditions.
        
        Args:
            user_risk_tolerance: User's stated risk tolerance
            market_volatility: Current market volatility
            position_size: Size of currency position
            var_estimate: Value at Risk estimate
            
        Returns:
            Risk compatibility assessment
        """
        thresholds = self.RISK_TOLERANCE_THRESHOLDS.get(
            user_risk_tolerance.lower(),
            self.RISK_TOLERANCE_THRESHOLDS['moderate']
        )
        
        # Calculate compatibility scores
        vol_compatibility = 1.0 - max(0, (market_volatility - thresholds['max_volatility']) / thresholds['max_volatility'])
        var_compatibility = 1.0 - max(0, (var_estimate / position_size - thresholds['var_tolerance']) / thresholds['var_tolerance'])
        
        overall_compatibility = (vol_compatibility + var_compatibility) / 2
        
        # Generate recommendations
        if overall_compatibility > 0.7:
            recommendation = "current_conditions_suitable"
            position_adjustment = 1.0
        elif overall_compatibility > 0.4:
            recommendation = "reduce_position_size"
            position_adjustment = 0.7
        else:
            recommendation = "wait_for_better_conditions"
            position_adjustment = 0.3
        
        return {
            "overall_compatibility": overall_compatibility,
            "volatility_compatibility": vol_compatibility,
            "var_compatibility": var_compatibility,
            "recommendation": recommendation,
            "suggested_position_multiplier": position_adjustment,
            "risk_tolerance": user_risk_tolerance,
            "thresholds_used": thresholds
        }
    
    def generate_risk_guidance(self, 
                             risk_assessment: Dict[str, Any],
                             user_risk_tolerance: str) -> str:
        """
        Generate personalized risk guidance for the user.
        
        Args:
            risk_assessment: Results from risk analysis
            user_risk_tolerance: User's risk tolerance level
            
        Returns:
            Personalized risk guidance string
        """
        overall_risk = risk_assessment.get('overall_risk', 0.5)
        volatility = risk_assessment.get('volatility_score', 0.5)
        time_pressure = risk_assessment.get('time_risk', 0.3)
        
        guidance_parts = []
        
        # Risk level assessment
        if overall_risk > 0.7:
            guidance_parts.append(f"High risk environment detected.")
            if user_risk_tolerance.lower() == 'conservative':
                guidance_parts.append("Consider waiting for more stable conditions.")
            elif user_risk_tolerance.lower() == 'moderate':
                guidance_parts.append("Proceed with caution and consider partial conversion.")
            else:
                guidance_parts.append("Acceptable risk for aggressive tolerance.")
        
        # Volatility guidance
        if volatility > 0.6:
            guidance_parts.append("High volatility may create opportunities but increases risk.")
        
        # Time pressure guidance
        if time_pressure > 0.7:
            guidance_parts.append("Time constraints limit flexibility - focus on execution.")
        elif time_pressure < 0.3:
            guidance_parts.append("Ample time available allows for optimal timing selection.")
        
        return " ".join(guidance_parts) if guidance_parts else "Standard risk management practices apply."


class RiskAnalysisToolkit:
    """Main toolkit for comprehensive risk analysis."""
    
    def __init__(self):
        self.volatility_calculator = VolatilityCalculator()
        self.risk_calculator = RiskMetricsCalculator()
        self.scenario_analyzer = ScenarioAnalyzer()
        self.user_profiler = UserRiskProfiler()
    
    async def comprehensive_risk_analysis(self,
                                        currency_pair: str,
                                        amount: float,
                                        user_risk_tolerance: str,
                                        price_data: Optional[List[float]] = None,
                                        market_context: Optional[Dict[str, Any]] = None,
                                        deadline_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis for currency conversion.
        
        Args:
            currency_pair: Currency pair to analyze
            amount: Conversion amount
            user_risk_tolerance: User's risk tolerance
            price_data: Historical price data
            market_context: Market intelligence context
            deadline_days: Days until deadline
            
        Returns:
            Comprehensive risk analysis results
        """
        analysis_results = {}
        
        # Volatility analysis
        if price_data and len(price_data) >= 30:
            vol_metrics = self.volatility_calculator.calculate_historical_volatility(price_data)
            vol_forecast = self.volatility_calculator.estimate_future_volatility(price_data, 7)
            analysis_results["volatility_analysis"] = {
                "current_volatility": vol_metrics.current_volatility,
                "historical_volatility": vol_metrics.historical_volatility,
                "volatility_percentile": vol_metrics.volatility_percentile,
                "trend": vol_metrics.trend,
                "forecast": vol_forecast,
                "confidence": vol_metrics.confidence
            }
        else:
            # Default volatility assumptions
            analysis_results["volatility_analysis"] = {
                "current_volatility": 0.12,  # 12% annual volatility default
                "historical_volatility": 0.12,
                "volatility_percentile": 0.5,
                "trend": "stable",
                "forecast": {"forecast_volatility": 0.12 / (252**0.5), "confidence": 0.3},
                "confidence": 0.3
            }
        
        # Risk metrics calculation
        if price_data:
            risk_metrics = self.risk_calculator.calculate_comprehensive_metrics(
                amount, price_data
            )
            analysis_results["risk_metrics"] = {
                "value_at_risk_1d": risk_metrics.value_at_risk_1d,
                "value_at_risk_7d": risk_metrics.value_at_risk_7d,
                "expected_shortfall": risk_metrics.expected_shortfall,
                "sharpe_ratio_estimate": risk_metrics.sharpe_ratio_estimate,
                "maximum_drawdown": risk_metrics.maximum_drawdown,
                "confidence_interval": risk_metrics.confidence_interval_95
            }
        else:
            # Default risk metrics
            vol = analysis_results["volatility_analysis"]["forecast"]["forecast_volatility"]
            var_1d = self.risk_calculator.calculate_value_at_risk(amount, vol, 0.95, 1)
            var_7d = self.risk_calculator.calculate_value_at_risk(amount, vol, 0.95, 7)
            
            analysis_results["risk_metrics"] = {
                "value_at_risk_1d": var_1d,
                "value_at_risk_7d": var_7d,
                "expected_shortfall": var_1d * 1.3,
                "sharpe_ratio_estimate": 0.5,
                "maximum_drawdown": 0.02,
                "confidence_interval": (-0.02, 0.02)
            }
        
        # Scenario analysis
        current_price = price_data[-1] if price_data else 1.0
        volatility = analysis_results["volatility_analysis"]["current_volatility"]
        scenarios = self.scenario_analyzer.generate_scenarios(current_price, volatility, 7)
        
        analysis_results["scenarios"] = [
            {
                "name": scenario.name,
                "probability": scenario.probability,
                "expected_return": scenario.expected_return,
                "value_at_risk": scenario.value_at_risk,
                "description": scenario.description
            }
            for scenario in scenarios
        ]
        
        # User risk compatibility
        var_7d = analysis_results["risk_metrics"]["value_at_risk_7d"]
        compatibility = self.user_profiler.assess_risk_compatibility(
            user_risk_tolerance, volatility, amount, var_7d
        )
        analysis_results["user_compatibility"] = compatibility
        
        # Time risk assessment
        time_analysis = TimeRiskAssessor().evaluate_deadline_pressure(deadline_days)
        analysis_results["time_analysis"] = time_analysis
        
        # Generate overall assessment
        overall_risk = self._calculate_overall_risk_score(analysis_results)
        risk_guidance = self.user_profiler.generate_risk_guidance(
            {"overall_risk": overall_risk, 
             "volatility_score": volatility / 0.3,  # Normalize to 0-1 scale
             "time_risk": time_analysis["time_pressure"]},
            user_risk_tolerance
        )
        
        return {
            "currency_pair": currency_pair,
            "amount": amount,
            "user_risk_tolerance": user_risk_tolerance,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_risk_score": overall_risk,
            "components": analysis_results,
            "risk_guidance": risk_guidance,
            "confidence": self._calculate_analysis_confidence(analysis_results)
        }
    
    def _calculate_overall_risk_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall risk score from component analyses."""
        vol_analysis = analysis_results.get("volatility_analysis", {})
        user_compat = analysis_results.get("user_compatibility", {})
        time_analysis = analysis_results.get("time_analysis", {})
        
        # Normalize volatility to 0-1 scale (assuming 30% annual vol as high)
        vol_score = min(1.0, vol_analysis.get("current_volatility", 0.12) / 0.30)
        
        # Get compatibility (inverted - low compatibility = high risk)
        compat_score = 1.0 - user_compat.get("overall_compatibility", 0.5)
        
        # Get time pressure
        time_score = time_analysis.get("time_pressure", 0.3)
        
        # Weighted combination
        overall_risk = (vol_score * 0.4 + compat_score * 0.35 + time_score * 0.25)
        
        return max(0.0, min(1.0, overall_risk))
    
    def _calculate_analysis_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the risk analysis."""
        confidences = []
        
        vol_conf = analysis_results.get("volatility_analysis", {}).get("confidence", 0.0)
        confidences.append(vol_conf)
        
        scenario_conf = analysis_results.get("scenarios", [{}])[0].get("confidence", 0.0) if analysis_results.get("scenarios") else 0.0
        if scenario_conf > 0:
            confidences.append(scenario_conf)
        
        return statistics.mean(confidences) if confidences else 0.5


# Convenience functions
def create_risk_analysis_toolkit() -> RiskAnalysisToolkit:
    """Factory function to create risk analysis toolkit.""" 
    return RiskAnalysisToolkit()


def mock_risk_data_for_testing(currency_pair: str, user_risk_tolerance: str) -> Dict[str, Any]:
    """Generate mock risk data for testing purposes."""
    import random
    
    # Mock price data with some volatility
    base_price = 0.85 if currency_pair == "USD/EUR" else 1.0
    price_data = []
    for i in range(30):
        daily_change = random.normalvariate(0, 0.01)  # 1% daily volatility
        base_price *= (1 + daily_change)
        price_data.append(base_price)
    
    # Mock market context
    market_context = {
        "sentiment_score": random.uniform(-0.3, 0.3),
        "market_regime": random.choice(["trending_up", "ranging", "volatile"]),
        "news_impact": random.uniform(0.2, 0.8)
    }
    
    return {
        "currency_pair": currency_pair,
        "amount": 10000.0,
        "user_risk_tolerance": user_risk_tolerance,
        "price_data": price_data,
        "market_context": market_context,
        "deadline_days": random.choice([None, 3, 7, 14, 30])
    }