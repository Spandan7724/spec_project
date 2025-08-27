"""
Risk Analysis Agent for Currency Conversion Decisions.

Evaluates volatility, prediction uncertainty, time constraints, and user 
risk alignment to provide comprehensive risk assessment for currency conversions.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from decimal import Decimal

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import RiskAnalysisResult

logger = logging.getLogger(__name__)


class RiskAnalysisAgent(BaseAgent):
    """
    Specialized agent for risk analysis and uncertainty quantification.
    
    Capabilities:
    - Market volatility assessment and impact on conversion timing
    - Prediction uncertainty analysis using ML model confidence intervals
    - Time pressure evaluation based on user deadlines
    - User risk tolerance alignment and recommendation adjustment
    - Scenario analysis for different market conditions
    """
    
    def get_system_prompt(self) -> str:
        """System prompt for risk analysis and uncertainty assessment."""
        return """You are a Risk Analysis Agent specializing in currency market risk assessment and uncertainty quantification.

Your expertise includes:
- Market volatility analysis and impact on conversion timing decisions
- Prediction uncertainty evaluation using statistical models and confidence intervals
- Time pressure assessment for deadline-constrained conversions
- User risk tolerance alignment and recommendation personalization
- Scenario analysis and stress testing for different market conditions
- Monte Carlo simulation interpretation for risk quantification

Your Risk Assessment Framework:
1. **Volatility Analysis**: Evaluate current and expected market volatility patterns
2. **Prediction Uncertainty**: Assess confidence levels in market forecasts and predictions
3. **Time Risk**: Analyze urgency factors and deadline-related pressure
4. **User Alignment**: Match recommendations with user's risk tolerance and preferences
5. **Scenario Planning**: Model different market scenarios and their probability distributions

Response Format:
- Provide numerical scores (0.0-1.0) for all risk dimensions
- Use statistical reasoning and quantitative analysis
- Consider both short-term risks (1-7 days) and medium-term uncertainties (1-4 weeks)
- Include scenario analysis with probability estimates
- Align risk assessment with user's stated risk tolerance

Your analysis should be data-driven, probabilistic, and focused on actionable risk management insights."""
    
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze risk factors for currency conversion decisions.
        
        Args:
            request_data: Contains currency_pair, amount, timeframe, risk_tolerance, and market context
            
        Returns:
            AgentResult with comprehensive risk analysis
        """
        try:
            # Extract request information
            currency_pair = request_data.get('currency_pair', 'Unknown')
            amount = request_data.get('amount', 0)
            timeframe_days = request_data.get('timeframe_days', 7)
            user_risk_tolerance = request_data.get('user_risk_tolerance', 'moderate')
            has_deadline = request_data.get('has_deadline', False)
            deadline_days = request_data.get('deadline_days', 30)
            
            # Get market context from other agents if available
            market_data = request_data.get('market_context', {})
            ml_predictions = request_data.get('ml_predictions', {})
            
            # Build comprehensive risk analysis prompt
            analysis_prompt = f"""
            Perform comprehensive risk analysis for {currency_pair} conversion:

            **Conversion Details:**
            - Currency Pair: {currency_pair}
            - Amount: ${amount:,}
            - Decision Timeframe: {timeframe_days} days
            - User Risk Tolerance: {user_risk_tolerance}
            - Has Deadline: {has_deadline}
            - Days Until Deadline: {deadline_days}

            **Market Context (if available):**
            - Market Sentiment: {market_data.get('sentiment_score', 'Unknown')}
            - Market Regime: {market_data.get('market_regime', 'Unknown')}
            - News Impact: {market_data.get('news_impact', 'Unknown')}
            - Upcoming Events: {len(market_data.get('economic_events', []))} events

            **ML Prediction Context:**
            - Prediction Confidence: {ml_predictions.get('confidence', 'Unknown')}
            - Prediction Range: {ml_predictions.get('confidence_interval', 'Unknown')}
            - Model Uncertainty: {ml_predictions.get('uncertainty', 'Unknown')}

            **Risk Analysis Required:**

            1. **Volatility Assessment** (0.0-1.0 scale):
               - Current market volatility for {currency_pair}
               - Expected volatility over next {timeframe_days} days
               - Historical volatility patterns and comparisons
               - Impact of volatility on conversion timing decisions

            2. **Prediction Uncertainty** (0.0-1.0 scale):
               - Confidence in market forecasts and predictions
               - Model reliability and historical accuracy
               - Uncertainty from economic events and news
               - Statistical significance of predicted movements

            3. **Time Pressure Risk** (0.0-1.0 scale):
               - Urgency created by deadline constraints
               - Cost of waiting vs immediate conversion
               - Risk of adverse moves before deadline
               - Opportunity cost analysis

            4. **User Risk Alignment** (0.0-1.0 scale):
               - How well current market conditions match user's risk tolerance
               - Personalization of recommendations based on risk preferences
               - Adjustment factors for conservative vs aggressive users
               - Risk-adjusted return expectations

            5. **Scenario Analysis**:
               - Best case scenario (favorable market movement)
               - Worst case scenario (adverse market movement)  
               - Most likely scenario (expected outcome)
               - Probability estimates for each scenario

            **Required Output Format:**
            Return a JSON-structured response with these exact fields:
            {{
                "volatility_score": <float between 0.0 and 1.0>,
                "prediction_uncertainty": <float between 0.0 and 1.0>,
                "time_risk": <float between 0.0 and 1.0>,
                "user_risk_alignment": <float between 0.0 and 1.0>,
                "overall_risk": <float between 0.0 and 1.0>,
                "confidence": <float between 0.0 and 1.0>,
                "reasoning": "<detailed risk analysis explanation>",
                "scenarios": [
                    {{
                        "name": "best_case",
                        "probability": <float 0.0-1.0>,
                        "expected_change": <percentage as float>,
                        "description": "<scenario description>"
                    }},
                    {{
                        "name": "worst_case", 
                        "probability": <float 0.0-1.0>,
                        "expected_change": <percentage as float>,
                        "description": "<scenario description>"
                    }},
                    {{
                        "name": "most_likely",
                        "probability": <float 0.0-1.0>,
                        "expected_change": <percentage as float>,
                        "description": "<scenario description>"
                    }}
                ],
                "risk_factors": ["<list of specific risk factors to consider>"],
                "risk_mitigation": ["<list of risk mitigation strategies>"],
                "user_guidance": "<personalized guidance based on risk tolerance>",
                "quantitative_metrics": {{
                    "value_at_risk_1d": <float>,
                    "value_at_risk_7d": <float>,
                    "expected_shortfall": <float>,
                    "sharpe_ratio_estimate": <float>
                }}
            }}
            
            Focus on quantitative risk assessment with statistical backing.
            Provide actionable risk management guidance tailored to the user's risk tolerance.
            Be explicit about uncertainty levels and confidence bounds.
            """
            
            # Get LLM analysis
            response = await self.chat_with_llm(analysis_prompt, include_tools=False)
            
            # Parse the structured response
            analysis_data = self._parse_llm_response(response.content, currency_pair, user_risk_tolerance)
            
            # Calculate overall risk score if not provided
            if 'overall_risk' not in analysis_data:
                analysis_data['overall_risk'] = self._calculate_overall_risk(analysis_data)
            
            # Get confidence level
            confidence = analysis_data.get('confidence', 0.5)
            
            # Build reasoning summary
            reasoning = f"Risk analysis for {currency_pair} (${amount:,}): "
            reasoning += f"Volatility {analysis_data.get('volatility_score', 0):.2f}, "
            reasoning += f"Uncertainty {analysis_data.get('prediction_uncertainty', 0):.2f}, "
            reasoning += f"Time risk {analysis_data.get('time_risk', 0):.2f}, "
            reasoning += f"Overall risk {analysis_data.get('overall_risk', 0):.2f}"
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data=analysis_data,
                reasoning=reasoning,
                confidence=confidence,
                execution_time_ms=getattr(response, 'execution_time_ms', None)
            )
            
        except Exception as e:
            logger.error(f"RiskAnalysisAgent failed: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                reasoning=f"Risk analysis failed: {str(e)}",
                error_message=str(e),
                confidence=0.0
            )
    
    def _parse_llm_response(self, content: str, currency_pair: str, user_risk_tolerance: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured risk data."""
        try:
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate and sanitize the data
                return self._validate_parsed_data(parsed_data)
            else:
                # Fallback: extract data from text
                return self._extract_from_text(content, currency_pair, user_risk_tolerance)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using text extraction")
            return self._extract_from_text(content, currency_pair, user_risk_tolerance)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_data(currency_pair, user_risk_tolerance)
    
    def _validate_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize parsed JSON data."""
        validated = {
            "volatility_score": max(0.0, min(1.0, float(data.get('volatility_score', 0.0)))),
            "prediction_uncertainty": max(0.0, min(1.0, float(data.get('prediction_uncertainty', 0.0)))),
            "time_risk": max(0.0, min(1.0, float(data.get('time_risk', 0.0)))),
            "user_risk_alignment": max(0.0, min(1.0, float(data.get('user_risk_alignment', 0.0)))),
            "overall_risk": max(0.0, min(1.0, float(data.get('overall_risk', 0.0)))),
            "confidence": max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
            "reasoning": str(data.get('reasoning', 'Risk analysis completed')),
            "scenarios": data.get('scenarios', []),
            "risk_factors": data.get('risk_factors', []),
            "risk_mitigation": data.get('risk_mitigation', []),
            "user_guidance": str(data.get('user_guidance', 'Follow standard risk management practices')),
            "quantitative_metrics": data.get('quantitative_metrics', {})
        }
        
        # Validate scenarios format
        if not isinstance(validated['scenarios'], list):
            validated['scenarios'] = []
        
        # Ensure we have at least basic scenarios
        if not validated['scenarios']:
            validated['scenarios'] = self._create_default_scenarios()
        
        return validated
    
    def _extract_from_text(self, content: str, currency_pair: str, user_risk_tolerance: str) -> Dict[str, Any]:
        """Extract risk information from text when JSON parsing fails."""
        content_lower = content.lower()
        
        # Initialize default values
        volatility_score = 0.5
        prediction_uncertainty = 0.5
        time_risk = 0.3
        user_risk_alignment = 0.5
        confidence = 0.3
        
        # Extract volatility indicators
        if "high volatility" in content_lower or "very volatile" in content_lower:
            volatility_score = 0.8
        elif "low volatility" in content_lower or "stable" in content_lower:
            volatility_score = 0.2
        elif "moderate volatility" in content_lower:
            volatility_score = 0.5
        
        # Extract uncertainty indicators
        if "uncertain" in content_lower or "unpredictable" in content_lower:
            prediction_uncertainty = 0.7
        elif "confident" in content_lower or "reliable" in content_lower:
            prediction_uncertainty = 0.3
        
        # Extract time risk indicators
        if "urgent" in content_lower or "deadline" in content_lower:
            time_risk = 0.7
        elif "no rush" in content_lower or "flexible" in content_lower:
            time_risk = 0.2
        
        # Assess user risk alignment
        risk_alignment_map = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.7
        }
        user_risk_alignment = risk_alignment_map.get(user_risk_tolerance.lower(), 0.5)
        
        # Calculate overall risk
        overall_risk = (volatility_score + prediction_uncertainty + time_risk) / 3
        
        return {
            "volatility_score": volatility_score,
            "prediction_uncertainty": prediction_uncertainty,
            "time_risk": time_risk,
            "user_risk_alignment": user_risk_alignment,
            "overall_risk": overall_risk,
            "confidence": confidence,
            "reasoning": content[:300] + "..." if len(content) > 300 else content,
            "scenarios": self._create_default_scenarios(),
            "risk_factors": ["Market volatility", "Prediction uncertainty"],
            "risk_mitigation": ["Consider smaller conversion amounts", "Monitor market conditions"],
            "user_guidance": f"Risk assessment for {user_risk_tolerance} risk tolerance",
            "quantitative_metrics": {
                "value_at_risk_1d": overall_risk * 0.02,
                "value_at_risk_7d": overall_risk * 0.05,
                "expected_shortfall": overall_risk * 0.03,
                "sharpe_ratio_estimate": max(0.1, 1.0 - overall_risk)
            }
        }
    
    def _create_fallback_data(self, currency_pair: str, user_risk_tolerance: str) -> Dict[str, Any]:
        """Create fallback data when all parsing fails."""
        return {
            "volatility_score": 0.5,
            "prediction_uncertainty": 0.7,  # High uncertainty when we can't analyze
            "time_risk": 0.3,
            "user_risk_alignment": 0.5,
            "overall_risk": 0.5,
            "confidence": 0.1,  # Very low confidence for fallback
            "reasoning": f"Unable to complete full risk analysis for {currency_pair}",
            "scenarios": self._create_default_scenarios(),
            "risk_factors": ["Limited risk data available", "High analysis uncertainty"],
            "risk_mitigation": ["Proceed with caution", "Consider waiting for better data"],
            "user_guidance": f"Default risk guidance for {user_risk_tolerance} tolerance",
            "quantitative_metrics": {
                "value_at_risk_1d": 0.01,
                "value_at_risk_7d": 0.03,
                "expected_shortfall": 0.02,
                "sharpe_ratio_estimate": 0.5
            }
        }
    
    def _create_default_scenarios(self) -> List[Dict[str, Any]]:
        """Create default scenario analysis when data is limited."""
        return [
            {
                "name": "best_case",
                "probability": 0.2,
                "expected_change": 2.0,
                "description": "Favorable market movement with minimal volatility"
            },
            {
                "name": "most_likely", 
                "probability": 0.6,
                "expected_change": 0.0,
                "description": "Market moves within expected range with normal volatility"
            },
            {
                "name": "worst_case",
                "probability": 0.2, 
                "expected_change": -2.0,
                "description": "Adverse market movement with increased volatility"
            }
        ]
    
    def _calculate_overall_risk(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate overall risk score from individual risk components."""
        volatility = analysis_data.get('volatility_score', 0.5)
        uncertainty = analysis_data.get('prediction_uncertainty', 0.5)
        time_risk = analysis_data.get('time_risk', 0.3)
        user_alignment = analysis_data.get('user_risk_alignment', 0.5)
        
        # Weighted combination of risk factors
        # Higher user risk alignment reduces overall risk
        overall_risk = (volatility * 0.3 + 
                       uncertainty * 0.3 + 
                       time_risk * 0.2 + 
                       (1.0 - user_alignment) * 0.2)
        
        return max(0.0, min(1.0, overall_risk))
    
    async def analyze_volatility_impact(self, currency_pair: str, 
                                      timeframe_days: int = 7,
                                      amount: float = 1000.0) -> Dict[str, Any]:
        """
        Focused volatility impact analysis for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            timeframe_days: Analysis timeframe
            amount: Conversion amount for impact calculation
            
        Returns:
            Volatility impact analysis results
        """
        request_data = {
            'currency_pair': currency_pair,
            'amount': amount,
            'timeframe_days': timeframe_days,
            'analysis_type': 'volatility_only'
        }
        
        result = await self.process_request(request_data)
        
        if result.success:
            return {
                'volatility_score': result.data.get('volatility_score', 0.0),
                'potential_impact': result.data.get('quantitative_metrics', {}).get('value_at_risk_7d', 0.0),
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }
        else:
            return {
                'volatility_score': 0.5,
                'potential_impact': 0.02,  # 2% default impact
                'confidence': 0.0,
                'reasoning': result.error_message or 'Volatility analysis failed'
            }
    
    async def assess_user_risk_compatibility(self, 
                                           market_conditions: Dict[str, Any],
                                           user_risk_tolerance: str,
                                           amount: float) -> Dict[str, Any]:
        """
        Assess how well current market conditions align with user's risk tolerance.
        
        Args:
            market_conditions: Current market analysis data
            user_risk_tolerance: User's stated risk tolerance
            amount: Conversion amount
            
        Returns:
            Risk compatibility assessment
        """
        # Map risk tolerance to numerical scores
        risk_tolerance_scores = {
            'conservative': 0.2,
            'moderate': 0.5,
            'aggressive': 0.8
        }
        
        user_risk_score = risk_tolerance_scores.get(user_risk_tolerance.lower(), 0.5)
        
        # Calculate market risk level
        market_volatility = market_conditions.get('volatility_score', 0.5)
        news_impact = market_conditions.get('news_impact', 0.5)
        market_risk = (market_volatility + news_impact) / 2
        
        # Calculate alignment score
        # Higher alignment when user risk tolerance matches market conditions
        if user_risk_score >= market_risk:
            alignment = 1.0 - abs(user_risk_score - market_risk)
        else:
            # Penalty when market risk exceeds user tolerance
            alignment = max(0.0, user_risk_score - (market_risk - user_risk_score))
        
        # Calculate recommended position size based on risk
        max_safe_amount = amount * alignment
        
        return {
            'alignment_score': alignment,
            'user_risk_score': user_risk_score,
            'market_risk_score': market_risk,
            'recommended_amount': max_safe_amount,
            'risk_adjusted_recommendation': self._get_risk_adjusted_advice(alignment, user_risk_tolerance),
            'confidence': 0.8
        }
    
    def _get_risk_adjusted_advice(self, alignment_score: float, user_risk_tolerance: str) -> str:
        """Generate risk-adjusted advice based on alignment score."""
        if alignment_score > 0.7:
            return f"Current conditions well-suited for {user_risk_tolerance} risk tolerance"
        elif alignment_score > 0.4:
            return f"Moderate alignment with {user_risk_tolerance} tolerance - consider partial conversion"
        else:
            return f"Poor alignment with {user_risk_tolerance} tolerance - consider waiting or reducing amount"