"""
Cost Optimization Agent for Currency Conversion Decisions.

Analyzes conversion costs, provider fees, timing impacts, and optimization 
strategies to minimize total conversion costs for users.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from decimal import Decimal

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import CostAnalysisResult

logger = logging.getLogger(__name__)


class CostOptimizationAgent(BaseAgent):
    """
    Specialized agent for cost optimization and provider selection.
    
    Capabilities:
    - Multi-provider fee comparison and cost analysis
    - Timing impact assessment on conversion costs
    - Hidden cost identification (spreads, markup, processing fees)
    - Savings opportunity quantification through optimal timing
    - Provider recommendation based on cost-benefit analysis
    """
    
    def get_system_prompt(self) -> str:
        """System prompt for cost optimization and provider analysis."""
        return """You are a Cost Optimization Agent specializing in currency conversion cost analysis and provider selection.

Your expertise includes:
- Multi-provider fee structure analysis and comparison
- Exchange rate spread analysis and markup identification
- Hidden cost detection (processing fees, intermediary banks, delay costs)
- Timing impact assessment on total conversion costs
- Cost-benefit analysis for different conversion strategies
- Savings optimization through provider selection and timing

Your Cost Analysis Framework:
1. **Provider Comparison**: Analyze fees, spreads, and total costs across available providers
2. **Timing Impact**: Assess how conversion timing affects total costs
3. **Hidden Cost Detection**: Identify all cost components beyond advertised rates
4. **Savings Optimization**: Quantify potential savings through optimization strategies
5. **Cost-Benefit Analysis**: Balance costs against market timing and user preferences

Response Format:
- Provide specific cost estimates and savings calculations
- Compare multiple providers with detailed breakdowns
- Include timing recommendations based on cost impact
- Consider both immediate costs and potential opportunity costs
- Focus on actionable cost optimization strategies

Your analysis should be precise, data-driven, and focused on minimizing total conversion costs while considering user constraints and market conditions."""
    
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze cost optimization strategies for currency conversion.
        
        Args:
            request_data: Contains currency_pair, amount, timeframe, provider options, and market context
            
        Returns:
            AgentResult with comprehensive cost analysis and optimization recommendations
        """
        try:
            # Extract request information
            currency_pair = request_data.get('currency_pair', 'Unknown')
            amount = request_data.get('amount', 0)
            timeframe_days = request_data.get('timeframe_days', 7)
            user_fee_sensitivity = request_data.get('user_fee_sensitivity', 0.5)
            
            # Provider and cost context
            available_providers = request_data.get('available_providers', [])
            current_rates = request_data.get('current_rates', {})
            market_context = request_data.get('market_context', {})
            
            # Build comprehensive cost analysis prompt
            analysis_prompt = f"""
            Perform comprehensive cost optimization analysis for {currency_pair} conversion:

            **Conversion Details:**
            - Currency Pair: {currency_pair}
            - Amount: ${amount:,}
            - Analysis Timeframe: {timeframe_days} days
            - User Fee Sensitivity: {user_fee_sensitivity} (0.0=low, 1.0=high)

            **Available Providers:** {len(available_providers)} providers
            {self._format_provider_context(available_providers)}

            **Current Market Rates:**
            {self._format_rate_context(current_rates, currency_pair)}

            **Market Context:**
            - Market Regime: {market_context.get('market_regime', 'Unknown')}
            - Volatility Level: {market_context.get('volatility_score', 'Unknown')}
            - Timing Sensitivity: {market_context.get('timing_recommendation', 'Unknown')}

            **Cost Analysis Required:**

            1. **Provider Cost Comparison**:
               - Calculate total cost for each available provider
               - Break down: base fee + spread markup + processing fees
               - Identify the most cost-effective provider
               - Account for transfer speeds and reliability

            2. **Timing Impact on Costs**:
               - How market volatility affects spread costs
               - Opportunity cost of waiting for better rates
               - Time-value of money considerations
               - Weekend/holiday premium impacts

            3. **Hidden Cost Analysis**:
               - Intermediary bank fees and correspondent charges
               - Currency conversion markup beyond advertised spreads
               - Processing delays and their cost implications
               - Account maintenance or minimum balance requirements

            4. **Optimization Strategies**:
               - Best timing for conversion to minimize costs
               - Optimal transaction sizing (single vs multiple conversions)
               - Provider arbitrage opportunities
               - Market timing vs cost certainty trade-offs

            5. **Savings Quantification**:
               - Potential savings from optimal provider selection
               - Savings from optimal timing (if applicable)
               - Cost difference between immediate vs delayed conversion
               - Total cost as percentage of conversion amount

            **Required Output Format:**
            Return a JSON-structured response with these exact fields:
            {{
                "best_provider": "<provider name>",
                "estimated_cost": <total cost in USD>,
                "cost_percentage": <cost as percentage of amount>,
                "potential_savings": <maximum potential savings in USD>,
                "timing_impact": <float 0.0-1.0 indicating timing sensitivity>,
                "provider_comparison": {{
                    "<provider_name>": {{
                        "total_cost": <cost in USD>,
                        "base_fee": <fee amount>,
                        "spread_cost": <spread cost>,
                        "processing_fee": <processing fee>,
                        "total_percentage": <total cost percentage>,
                        "transfer_time": "<estimated transfer time>",
                        "reliability_score": <0.0-1.0>
                    }}
                }},
                "confidence": <float between 0.0 and 1.0>,
                "reasoning": "<detailed cost analysis explanation>",
                "optimization_recommendations": [
                    "<list of specific cost optimization strategies>"
                ],
                "timing_recommendation": "<immediate/wait_1_day/wait_2_3_days/wait_week>",
                "cost_breakdown": {{
                    "exchange_rate_cost": <percentage>,
                    "provider_fees": <percentage>,
                    "hidden_costs": <percentage>,
                    "opportunity_cost": <percentage if waiting>
                }}
            }}
            
            Focus on precise cost calculations and actionable optimization strategies.
            Consider both immediate costs and time-value trade-offs.
            Provide clear reasoning for provider and timing recommendations.
            """
            
            # Get LLM analysis
            response = await self.chat_with_llm(analysis_prompt, include_tools=False)
            
            # Parse the structured response
            analysis_data = self._parse_llm_response(response.content, currency_pair, amount)
            
            # Validate and enhance cost data
            analysis_data = self._validate_cost_data(analysis_data, amount)
            
            # Calculate confidence
            confidence = analysis_data.get('confidence', 0.5)
            
            # Build reasoning summary
            reasoning = f"Cost optimization for {currency_pair} (${amount:,}): "
            reasoning += f"Best provider: {analysis_data.get('best_provider', 'Unknown')}, "
            reasoning += f"Est. cost: ${analysis_data.get('estimated_cost', 0):.2f} "
            reasoning += f"({analysis_data.get('cost_percentage', 0):.2f}%), "
            reasoning += f"Potential savings: ${analysis_data.get('potential_savings', 0):.2f}"
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data=analysis_data,
                reasoning=reasoning,
                confidence=confidence,
                execution_time_ms=getattr(response, 'execution_time_ms', None)
            )
            
        except Exception as e:
            logger.error(f"CostOptimizationAgent failed: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                reasoning=f"Cost optimization analysis failed: {str(e)}",
                error_message=str(e),
                confidence=0.0
            )
    
    def _format_provider_context(self, providers: List[Dict[str, Any]]) -> str:
        """Format provider information for LLM context."""
        if not providers:
            return "No specific provider data available - use general market knowledge"
        
        formatted = []
        for provider in providers:
            name = provider.get('name', 'Unknown')
            fee = provider.get('fee_percentage', 'Unknown')
            spread = provider.get('spread_bps', 'Unknown')
            speed = provider.get('transfer_speed', 'Unknown')
            formatted.append(f"- {name}: {fee}% fee, {spread}bps spread, {speed}")
        
        return "\n".join(formatted)
    
    def _format_rate_context(self, rates: Dict[str, Any], currency_pair: str) -> str:
        """Format current rate information for LLM context."""
        if not rates:
            return f"Current market rate for {currency_pair}: Use live market rates"
        
        pair_rate = rates.get(currency_pair, rates.get('rate', 'Unknown'))
        spread = rates.get('spread', 'Unknown')
        
        return f"Market rate: {pair_rate}, Current spread: {spread}"
    
    def _parse_llm_response(self, content: str, currency_pair: str, amount: float) -> Dict[str, Any]:
        """Parse LLM response and extract structured cost data."""
        try:
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate and sanitize the data
                return self._validate_parsed_data(parsed_data, amount)
            else:
                # Fallback: extract data from text
                return self._extract_from_text(content, currency_pair, amount)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using text extraction")
            return self._extract_from_text(content, currency_pair, amount)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_data(currency_pair, amount)
    
    def _validate_parsed_data(self, data: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """Validate and sanitize parsed JSON cost data."""
        validated = {
            "best_provider": str(data.get('best_provider', 'Unknown')),
            "estimated_cost": max(0.0, float(data.get('estimated_cost', amount * 0.01))),
            "cost_percentage": max(0.0, min(10.0, float(data.get('cost_percentage', 1.0)))),
            "potential_savings": max(0.0, float(data.get('potential_savings', 0.0))),
            "timing_impact": max(0.0, min(1.0, float(data.get('timing_impact', 0.3)))),
            "provider_comparison": data.get('provider_comparison', {}),
            "confidence": max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
            "reasoning": str(data.get('reasoning', 'Cost analysis completed')),
            "optimization_recommendations": data.get('optimization_recommendations', []),
            "timing_recommendation": str(data.get('timing_recommendation', 'immediate')),
            "cost_breakdown": data.get('cost_breakdown', {})
        }
        
        # Ensure provider comparison is valid
        if not isinstance(validated['provider_comparison'], dict):
            validated['provider_comparison'] = {}
        
        # Ensure optimization recommendations is a list
        if not isinstance(validated['optimization_recommendations'], list):
            validated['optimization_recommendations'] = []
        
        return validated
    
    def _extract_from_text(self, content: str, currency_pair: str, amount: float) -> Dict[str, Any]:
        """Extract cost information from text when JSON parsing fails."""
        content_lower = content.lower()
        
        # Default cost estimates
        estimated_cost = amount * 0.015  # 1.5% default cost
        cost_percentage = 1.5
        potential_savings = amount * 0.005  # 0.5% potential savings
        best_provider = "Bank Transfer"
        timing_impact = 0.3
        
        # Extract cost indicators from text
        if "expensive" in content_lower or "high cost" in content_lower:
            estimated_cost = amount * 0.025  # 2.5%
            cost_percentage = 2.5
        elif "cheap" in content_lower or "low cost" in content_lower:
            estimated_cost = amount * 0.008  # 0.8%
            cost_percentage = 0.8
        
        # Extract savings indicators
        if "significant savings" in content_lower:
            potential_savings = amount * 0.01  # 1% savings
        elif "small savings" in content_lower:
            potential_savings = amount * 0.003  # 0.3% savings
        
        # Extract timing indicators
        if "time sensitive" in content_lower or "urgent" in content_lower:
            timing_impact = 0.8
        elif "no rush" in content_lower:
            timing_impact = 0.1
        
        return {
            "best_provider": best_provider,
            "estimated_cost": estimated_cost,
            "cost_percentage": cost_percentage,
            "potential_savings": potential_savings,
            "timing_impact": timing_impact,
            "provider_comparison": self._create_default_provider_comparison(amount),
            "confidence": 0.3,  # Low confidence for text extraction
            "reasoning": content[:300] + "..." if len(content) > 300 else content,
            "optimization_recommendations": [
                "Compare multiple providers before converting",
                "Consider timing flexibility for better rates"
            ],
            "timing_recommendation": "immediate",
            "cost_breakdown": {
                "exchange_rate_cost": 0.5,
                "provider_fees": 1.0,
                "hidden_costs": 0.3,
                "opportunity_cost": 0.0
            }
        }
    
    def _create_fallback_data(self, currency_pair: str, amount: float) -> Dict[str, Any]:
        """Create fallback data when all parsing fails."""
        return {
            "best_provider": "Unknown",
            "estimated_cost": amount * 0.02,  # 2% fallback cost
            "cost_percentage": 2.0,
            "potential_savings": 0.0,
            "timing_impact": 0.5,
            "provider_comparison": {},
            "confidence": 0.1,  # Very low confidence for fallback
            "reasoning": f"Unable to complete full cost analysis for {currency_pair}",
            "optimization_recommendations": ["Gather more provider data", "Compare costs manually"],
            "timing_recommendation": "unclear",
            "cost_breakdown": {
                "exchange_rate_cost": 1.0,
                "provider_fees": 1.0,
                "hidden_costs": 0.5,
                "opportunity_cost": 0.0
            }
        }
    
    def _create_default_provider_comparison(self, amount: float) -> Dict[str, Dict[str, Any]]:
        """Create default provider comparison for fallback scenarios."""
        return {
            "Traditional Bank": {
                "total_cost": amount * 0.025,
                "base_fee": amount * 0.005,
                "spread_cost": amount * 0.015,
                "processing_fee": amount * 0.005,
                "total_percentage": 2.5,
                "transfer_time": "1-3 business days",
                "reliability_score": 0.9
            },
            "Online Service": {
                "total_cost": amount * 0.015,
                "base_fee": amount * 0.003,
                "spread_cost": amount * 0.008,
                "processing_fee": amount * 0.004,
                "total_percentage": 1.5,
                "transfer_time": "1-2 business days", 
                "reliability_score": 0.8
            },
            "Digital Wallet": {
                "total_cost": amount * 0.012,
                "base_fee": amount * 0.002,
                "spread_cost": amount * 0.006,
                "processing_fee": amount * 0.004,
                "total_percentage": 1.2,
                "transfer_time": "Minutes to hours",
                "reliability_score": 0.7
            }
        }
    
    def _validate_cost_data(self, data: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """Validate and enhance cost analysis data."""
        # Ensure estimated cost is reasonable (0.1% to 5% of amount)
        min_cost = amount * 0.001
        max_cost = amount * 0.05
        
        if data.get('estimated_cost', 0) < min_cost:
            data['estimated_cost'] = min_cost
            data['cost_percentage'] = 0.1
        elif data.get('estimated_cost', 0) > max_cost:
            data['estimated_cost'] = max_cost
            data['cost_percentage'] = 5.0
        
        # Ensure potential savings don't exceed estimated cost
        if data.get('potential_savings', 0) > data.get('estimated_cost', 0):
            data['potential_savings'] = data.get('estimated_cost', 0) * 0.5
        
        return data
    
    async def compare_providers(self, 
                              currency_pair: str,
                              amount: float,
                              providers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple providers for cost optimization.
        
        Args:
            currency_pair: Currency pair for conversion
            amount: Conversion amount
            providers: List of provider data
            
        Returns:
            Provider comparison results
        """
        request_data = {
            'currency_pair': currency_pair,
            'amount': amount,
            'available_providers': providers,
            'analysis_type': 'provider_comparison_only'
        }
        
        result = await self.process_request(request_data)
        
        if result.success:
            return {
                'best_provider': result.data.get('best_provider', 'Unknown'),
                'provider_comparison': result.data.get('provider_comparison', {}),
                'total_cost_range': self._calculate_cost_range(result.data.get('provider_comparison', {})),
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }
        else:
            return {
                'best_provider': 'Unknown',
                'provider_comparison': {},
                'total_cost_range': {'min': amount * 0.01, 'max': amount * 0.03},
                'confidence': 0.0,
                'reasoning': result.error_message or 'Provider comparison failed'
            }
    
    def _calculate_cost_range(self, provider_comparison: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate min and max costs across providers."""
        if not provider_comparison:
            return {'min': 0.0, 'max': 0.0}
        
        costs = [provider.get('total_cost', 0) for provider in provider_comparison.values()]
        return {
            'min': min(costs) if costs else 0.0,
            'max': max(costs) if costs else 0.0,
            'range': max(costs) - min(costs) if costs else 0.0
        }
    
    async def analyze_timing_impact(self, 
                                  currency_pair: str,
                                  amount: float,
                                  wait_days: int = 3) -> Dict[str, Any]:
        """
        Analyze how waiting affects conversion costs.
        
        Args:
            currency_pair: Currency pair to analyze
            amount: Conversion amount
            wait_days: Days to wait before converting
            
        Returns:
            Timing impact analysis
        """
        request_data = {
            'currency_pair': currency_pair,
            'amount': amount,
            'timeframe_days': wait_days,
            'analysis_type': 'timing_impact_only'
        }
        
        result = await self.process_request(request_data)
        
        if result.success:
            timing_impact = result.data.get('timing_impact', 0.3)
            potential_savings = result.data.get('potential_savings', 0.0)
            
            # Calculate opportunity cost
            opportunity_cost = timing_impact * amount * 0.001 * wait_days  # Rough estimate
            
            return {
                'timing_impact_score': timing_impact,
                'potential_savings': potential_savings,
                'opportunity_cost': opportunity_cost,
                'net_benefit': potential_savings - opportunity_cost,
                'recommendation': 'wait' if potential_savings > opportunity_cost else 'convert_now',
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }
        else:
            return {
                'timing_impact_score': 0.3,
                'potential_savings': 0.0,
                'opportunity_cost': amount * 0.001 * wait_days,
                'net_benefit': -amount * 0.001 * wait_days,
                'recommendation': 'convert_now',
                'confidence': 0.0,
                'reasoning': result.error_message or 'Timing analysis failed'
            }
    
    async def calculate_total_conversion_cost(self,
                                            currency_pair: str,
                                            amount: float,
                                            provider_name: str,
                                            include_opportunity_cost: bool = True) -> Dict[str, Any]:
        """
        Calculate total cost of conversion including all fees and opportunity costs.
        
        Args:
            currency_pair: Currency pair for conversion
            amount: Conversion amount
            provider_name: Provider to analyze
            include_opportunity_cost: Whether to include opportunity costs
            
        Returns:
            Comprehensive cost breakdown
        """
        # Estimate costs based on provider type
        cost_estimates = {
            'traditional_bank': {'base_fee': 0.005, 'spread': 0.015, 'processing': 0.005},
            'online_service': {'base_fee': 0.003, 'spread': 0.008, 'processing': 0.004},
            'digital_wallet': {'base_fee': 0.002, 'spread': 0.006, 'processing': 0.004},
            'crypto_exchange': {'base_fee': 0.001, 'spread': 0.003, 'processing': 0.002}
        }
        
        # Determine provider type
        provider_lower = provider_name.lower()
        if 'bank' in provider_lower:
            costs = cost_estimates['traditional_bank']
        elif 'wise' in provider_lower or 'remitly' in provider_lower:
            costs = cost_estimates['online_service']
        elif 'paypal' in provider_lower or 'revolut' in provider_lower:
            costs = cost_estimates['digital_wallet']
        elif 'crypto' in provider_lower or 'binance' in provider_lower:
            costs = cost_estimates['crypto_exchange']
        else:
            costs = cost_estimates['online_service']  # Default to online service
        
        # Calculate individual cost components
        base_fee = amount * costs['base_fee']
        spread_cost = amount * costs['spread']
        processing_fee = amount * costs['processing']
        total_provider_cost = base_fee + spread_cost + processing_fee
        
        # Calculate opportunity cost (if waiting)
        opportunity_cost = 0.0
        if include_opportunity_cost:
            # Simple opportunity cost: risk-free rate * amount * time
            daily_opportunity = amount * (0.02 / 365)  # 2% annual risk-free rate
            opportunity_cost = daily_opportunity * 1  # 1 day default
        
        total_cost = total_provider_cost + opportunity_cost
        
        return {
            'provider_name': provider_name,
            'base_fee': base_fee,
            'spread_cost': spread_cost,
            'processing_fee': processing_fee,
            'provider_cost': total_provider_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': total_cost,
            'cost_percentage': (total_cost / amount) * 100,
            'breakdown': {
                'base_fee_pct': (base_fee / amount) * 100,
                'spread_pct': (spread_cost / amount) * 100,
                'processing_pct': (processing_fee / amount) * 100,
                'opportunity_pct': (opportunity_cost / amount) * 100
            },
            'confidence': 0.7
        }