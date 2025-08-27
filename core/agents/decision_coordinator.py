"""
Decision Coordinator Agent for Multi-Agent Currency Conversion Workflows.

Synthesizes analysis from MarketIntelligence, RiskAnalysis, and CostOptimization 
agents to produce final conversion recommendations with clear reasoning.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from decimal import Decimal

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import CurrencyDecisionState, WorkflowStatus
from ..models import DecisionRecommendation, DecisionType

logger = logging.getLogger(__name__)


class DecisionCoordinator(BaseAgent):
    """
    Coordinator agent that synthesizes multi-agent analysis into final recommendations.
    
    Capabilities:
    - Synthesis of market intelligence, risk analysis, and cost optimization inputs
    - Conflict resolution between different agent recommendations
    - Confidence weighting and consensus building across agents
    - Final decision generation with clear reasoning and action items
    - Integration with existing ML forecasting and business logic
    """
    
    def get_system_prompt(self) -> str:
        """System prompt for decision coordination and synthesis."""
        return """You are a Decision Coordinator Agent responsible for synthesizing multi-agent currency conversion analysis into final actionable recommendations.

Your role involves:
- Integrating insights from MarketIntelligence, RiskAnalysis, and CostOptimization agents
- Resolving conflicts and contradictions between different agent recommendations
- Weighting agent inputs based on confidence levels and market conditions  
- Producing clear, actionable decision recommendations with supporting reasoning
- Balancing multiple objectives: timing, cost, risk, and user preferences

Your Decision Framework:
1. **Agent Input Synthesis**: Analyze and weight inputs from all contributing agents
2. **Conflict Resolution**: Identify and resolve contradictory recommendations
3. **Consensus Building**: Find optimal balance between market timing, risk, and cost factors
4. **Decision Logic**: Apply decision rules that prioritize user objectives and constraints
5. **Recommendation Generation**: Produce clear "convert now" or "wait" decisions with reasoning

Key Considerations:
- Market intelligence provides timing and sentiment insights
- Risk analysis quantifies uncertainty and volatility impacts
- Cost optimization identifies best providers and timing for cost minimization
- User risk tolerance and fee sensitivity must be paramount in final decisions
- ML predictions and confidence intervals inform timing recommendations

Response Requirements:
- Provide definitive "convert_now" or "wait_X_days" decisions
- Include confidence scores for all recommendations
- Explain how different agent inputs were weighted and reconciled
- Give specific action items and next steps for the user
- Account for user constraints (deadlines, risk tolerance, fee sensitivity)

Your synthesis should be authoritative, well-reasoned, and actionable."""
    
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """
        Coordinate and synthesize multi-agent analysis into final decision.
        
        Args:
            request_data: Contains workflow_state with all agent results and context
            
        Returns:
            AgentResult with final decision recommendation
        """
        try:
            # Extract workflow state and agent results
            workflow_state = request_data.get('workflow_state')
            if not workflow_state:
                raise ValueError("Workflow state required for decision coordination")
            
            # Extract individual agent results
            market_analysis = self._extract_market_analysis(workflow_state)
            risk_analysis = self._extract_risk_analysis(workflow_state) 
            cost_analysis = self._extract_cost_analysis(workflow_state)
            
            # Get request context
            currency_pair = workflow_state.request.currency_pair
            amount = float(workflow_state.request.amount)
            user_risk_tolerance = workflow_state.request.user_profile.risk_tolerance.value
            user_fee_sensitivity = workflow_state.request.user_profile.fee_sensitivity
            has_deadline = workflow_state.request.deadline is not None
            
            # Build coordination prompt
            coordination_prompt = f"""
            Synthesize multi-agent analysis into final currency conversion decision:

            **Conversion Request:**
            - Currency Pair: {currency_pair}
            - Amount: ${amount:,}
            - User Risk Tolerance: {user_risk_tolerance}
            - User Fee Sensitivity: {user_fee_sensitivity}
            - Has Deadline: {has_deadline}
            - Deadline: {workflow_state.request.deadline.isoformat() if workflow_state.request.deadline else 'None'}

            **Market Intelligence Analysis:**
            {self._format_market_analysis(market_analysis)}

            **Risk Analysis Results:**
            {self._format_risk_analysis(risk_analysis)}

            **Cost Optimization Results:**
            {self._format_cost_analysis(cost_analysis)}

            **Agent Confidence Levels:**
            - Market Intelligence: {market_analysis.get('confidence', 0):.2f}
            - Risk Analysis: {risk_analysis.get('confidence', 0):.2f}
            - Cost Optimization: {cost_analysis.get('confidence', 0):.2f}

            **Decision Coordination Required:**

            1. **Input Synthesis**: 
               - How do market conditions align with risk tolerance and cost sensitivity?
               - What are the key tensions between timing, risk, and cost considerations?
               - Which agent inputs should be weighted most heavily given current conditions?

            2. **Conflict Resolution**:
               - Do agents agree on timing recommendations?
               - How should contradictory signals be resolved?
               - What are the trade-offs between competing recommendations?

            3. **User-Centric Decision**:
               - Given user's {user_risk_tolerance} risk tolerance and {user_fee_sensitivity:.1f} fee sensitivity
               - What decision best serves user's interests and constraints?
               - How should deadline pressure factor into the decision?

            4. **Final Recommendation**:
               - Convert immediately or wait (specify days if waiting)
               - Which provider to use and why
               - What amount to convert (full amount or partial)
               - Key risk factors and mitigation strategies

            **Required Output Format:**
            Return a JSON-structured response with these exact fields:
            {{
                "final_decision": "<convert_now/wait_1_day/wait_2_3_days/wait_1_week/wait_flexible>",
                "recommended_provider": "<provider name>",
                "recommended_amount": <amount to convert>,
                "confidence": <float between 0.0 and 1.0>,
                "reasoning": "<detailed explanation of decision logic>",
                "agent_consensus": {{
                    "market_weight": <float 0.0-1.0>,
                    "risk_weight": <float 0.0-1.0>, 
                    "cost_weight": <float 0.0-1.0>,
                    "consensus_score": <float 0.0-1.0>
                }},
                "key_factors": [
                    "<list of 3-5 most important decision factors>"
                ],
                "action_items": [
                    "<specific next steps for the user>"
                ],
                "risk_mitigation": [
                    "<risk management recommendations>"
                ],
                "alternative_scenarios": [
                    {{
                        "condition": "<if market condition changes>",
                        "recommendation": "<alternative action>",
                        "trigger": "<what to monitor>"
                    }}
                ],
                "expected_outcome": {{
                    "conversion_rate_estimate": <float>,
                    "total_cost_estimate": <float>,
                    "success_probability": <float 0.0-1.0>
                }}
            }}

            Focus on clear decision logic that balances all factors while prioritizing user objectives.
            Provide specific, actionable guidance the user can immediately act upon.
            """
            
            # Get LLM coordination analysis
            response = await self.chat_with_llm(coordination_prompt, include_tools=False)
            
            # Parse the structured response
            decision_data = self._parse_llm_response(response.content, workflow_state)
            
            # Create final recommendation
            final_recommendation = self._create_decision_recommendation(decision_data, workflow_state)
            
            # Calculate overall confidence
            confidence = decision_data.get('confidence', 0.5)
            
            # Build comprehensive reasoning
            reasoning = self._build_coordination_reasoning(decision_data, market_analysis, risk_analysis, cost_analysis)
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data=decision_data,
                reasoning=reasoning,
                confidence=confidence,
                execution_time_ms=getattr(response, 'execution_time_ms', None)
            )
            
        except Exception as e:
            logger.error(f"DecisionCoordinator failed: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                reasoning=f"Decision coordination failed: {str(e)}",
                error_message=str(e),
                confidence=0.0
            )
    
    def _extract_market_analysis(self, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Extract market intelligence data from workflow state."""
        if 'market_intelligence' in workflow_state.agent_results:
            result = workflow_state.agent_results['market_intelligence']
            if result.success:
                return result.data
        
        # Fallback data
        return {
            "sentiment_score": 0.0,
            "market_regime": "unknown",
            "timing_recommendation": "unclear",
            "confidence": 0.0
        }
    
    def _extract_risk_analysis(self, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Extract risk analysis data from workflow state."""
        if 'risk_analysis' in workflow_state.agent_results:
            result = workflow_state.agent_results['risk_analysis']
            if result.success:
                return result.data
                
        # Fallback data
        return {
            "overall_risk": 0.5,
            "volatility_score": 0.5,
            "user_risk_alignment": 0.5,
            "confidence": 0.0
        }
    
    def _extract_cost_analysis(self, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Extract cost optimization data from workflow state."""
        if 'cost_optimization' in workflow_state.agent_results:
            result = workflow_state.agent_results['cost_optimization']
            if result.success:
                return result.data
                
        # Fallback data
        return {
            "best_provider": "Unknown",
            "estimated_cost": float(workflow_state.request.amount) * 0.02,
            "timing_impact": 0.3,
            "confidence": 0.0
        }
    
    def _format_market_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format market analysis for LLM prompt."""
        return f"""
        - Market Sentiment: {analysis.get('sentiment_score', 0):.2f} (-1.0 to +1.0)
        - Market Regime: {analysis.get('market_regime', 'unknown')}
        - News Impact: {analysis.get('news_impact', 0):.2f}
        - Timing Recommendation: {analysis.get('timing_recommendation', 'unclear')}
        - Economic Events: {len(analysis.get('economic_events', []))} upcoming
        """
    
    def _format_risk_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format risk analysis for LLM prompt."""
        return f"""
        - Overall Risk Score: {analysis.get('overall_risk', 0.5):.2f} (0.0=low, 1.0=high)
        - Volatility Score: {analysis.get('volatility_score', 0.5):.2f}
        - Prediction Uncertainty: {analysis.get('prediction_uncertainty', 0.5):.2f}
        - Time Risk: {analysis.get('time_risk', 0.3):.2f}
        - User Risk Alignment: {analysis.get('user_risk_alignment', 0.5):.2f}
        - Risk Scenarios: {len(analysis.get('scenarios', []))} scenarios analyzed
        """
    
    def _format_cost_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format cost analysis for LLM prompt."""
        return f"""
        - Best Provider: {analysis.get('best_provider', 'Unknown')}
        - Estimated Cost: ${analysis.get('estimated_cost', 0):.2f}
        - Cost Percentage: {analysis.get('cost_percentage', 0):.2f}%
        - Potential Savings: ${analysis.get('potential_savings', 0):.2f}
        - Timing Impact: {analysis.get('timing_impact', 0.3):.2f}
        - Providers Compared: {len(analysis.get('provider_comparison', {}))}
        """
    
    def _parse_llm_response(self, content: str, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Parse LLM response and extract structured decision data."""
        try:
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate and sanitize the data
                return self._validate_parsed_data(parsed_data, workflow_state)
            else:
                # Fallback: extract data from text
                return self._extract_from_text(content, workflow_state)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using text extraction")
            return self._extract_from_text(content, workflow_state)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_data(workflow_state)
    
    def _validate_parsed_data(self, data: Dict[str, Any], workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Validate and sanitize parsed JSON decision data."""
        valid_decisions = ["convert_now", "wait_1_day", "wait_2_3_days", "wait_1_week", "wait_flexible"]
        
        validated = {
            "final_decision": str(data.get('final_decision', 'convert_now')),
            "recommended_provider": str(data.get('recommended_provider', 'Unknown')),
            "recommended_amount": max(0.0, float(data.get('recommended_amount', float(workflow_state.request.amount)))),
            "confidence": max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
            "reasoning": str(data.get('reasoning', 'Decision coordination completed')),
            "agent_consensus": data.get('agent_consensus', {}),
            "key_factors": data.get('key_factors', []),
            "action_items": data.get('action_items', []),
            "risk_mitigation": data.get('risk_mitigation', []),
            "alternative_scenarios": data.get('alternative_scenarios', []),
            "expected_outcome": data.get('expected_outcome', {})
        }
        
        # Validate decision value
        if validated['final_decision'] not in valid_decisions:
            validated['final_decision'] = 'convert_now'
        
        # Ensure lists are actually lists
        for list_field in ['key_factors', 'action_items', 'risk_mitigation', 'alternative_scenarios']:
            if not isinstance(validated[list_field], list):
                validated[list_field] = []
        
        # Validate agent consensus
        if not isinstance(validated['agent_consensus'], dict):
            validated['agent_consensus'] = {
                "market_weight": 0.33,
                "risk_weight": 0.33,
                "cost_weight": 0.34,
                "consensus_score": 0.5
            }
        
        return validated
    
    def _extract_from_text(self, content: str, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Extract decision information from text when JSON parsing fails."""
        content_lower = content.lower()
        
        # Extract decision
        final_decision = "convert_now"
        if "wait" in content_lower:
            if "1 week" in content_lower or "week" in content_lower:
                final_decision = "wait_1_week"
            elif "2-3 days" in content_lower or "few days" in content_lower:
                final_decision = "wait_2_3_days"
            elif "1 day" in content_lower or "tomorrow" in content_lower:
                final_decision = "wait_1_day"
            else:
                final_decision = "wait_flexible"
        
        # Extract provider recommendation
        recommended_provider = "Unknown"
        if "wise" in content_lower:
            recommended_provider = "Wise"
        elif "revolut" in content_lower:
            recommended_provider = "Revolut"
        elif "bank" in content_lower:
            recommended_provider = "Traditional Bank"
        
        return {
            "final_decision": final_decision,
            "recommended_provider": recommended_provider,
            "recommended_amount": float(workflow_state.request.amount),
            "confidence": 0.3,  # Low confidence for text extraction
            "reasoning": content[:500] + "..." if len(content) > 500 else content,
            "agent_consensus": {
                "market_weight": 0.33,
                "risk_weight": 0.33,
                "cost_weight": 0.34,
                "consensus_score": 0.5
            },
            "key_factors": ["Market conditions", "Risk assessment", "Cost analysis"],
            "action_items": [f"Execute {final_decision} with {recommended_provider}"],
            "risk_mitigation": ["Monitor market conditions", "Review decision if conditions change"],
            "alternative_scenarios": [],
            "expected_outcome": {
                "conversion_rate_estimate": 0.85,
                "total_cost_estimate": float(workflow_state.request.amount) * 0.015,
                "success_probability": 0.8
            }
        }
    
    def _create_fallback_data(self, workflow_state: CurrencyDecisionState) -> Dict[str, Any]:
        """Create fallback decision data when all parsing fails."""
        return {
            "final_decision": "convert_now",
            "recommended_provider": "Online Service",
            "recommended_amount": float(workflow_state.request.amount),
            "confidence": 0.1,  # Very low confidence for fallback
            "reasoning": f"Unable to complete full coordination analysis for {workflow_state.request.currency_pair}",
            "agent_consensus": {
                "market_weight": 0.33,
                "risk_weight": 0.33,
                "cost_weight": 0.34,
                "consensus_score": 0.1
            },
            "key_factors": ["Limited analysis data", "Default conservative approach"],
            "action_items": ["Proceed with immediate conversion using reliable provider"],
            "risk_mitigation": ["Use trusted provider", "Monitor exchange rates"],
            "alternative_scenarios": [],
            "expected_outcome": {
                "conversion_rate_estimate": float(workflow_state.request.current_rate),
                "total_cost_estimate": float(workflow_state.request.amount) * 0.02,
                "success_probability": 0.7
            }
        }
    
    def _create_decision_recommendation(self, 
                                      decision_data: Dict[str, Any],
                                      workflow_state: CurrencyDecisionState) -> DecisionRecommendation:
        """Create DecisionRecommendation object from analysis data."""
        # Map decision to enum
        decision_mapping = {
            "convert_now": DecisionType.CONVERT_NOW,
            "wait_1_day": DecisionType.WAIT,
            "wait_2_3_days": DecisionType.WAIT,
            "wait_1_week": DecisionType.WAIT,
            "wait_flexible": DecisionType.WAIT
        }
        
        decision_enum = decision_mapping.get(
            decision_data.get('final_decision', 'convert_now'),
            DecisionType.CONVERT_NOW
        )
        
        # Extract wait days for wait decisions
        wait_days = 0
        final_decision = decision_data.get('final_decision', 'convert_now')
        if final_decision == "wait_1_day":
            wait_days = 1
        elif final_decision == "wait_2_3_days":
            wait_days = 3
        elif final_decision == "wait_1_week":
            wait_days = 7
        elif final_decision == "wait_flexible":
            wait_days = 14
        
        # Build comprehensive explanation
        explanation = decision_data.get('reasoning', '')
        if decision_data.get('key_factors'):
            explanation += f"\n\nKey factors: {', '.join(decision_data['key_factors'])}"
        if decision_data.get('action_items'):
            explanation += f"\n\nNext steps: {'; '.join(decision_data['action_items'])}"
        
        return DecisionRecommendation(
            decision=decision_enum,
            confidence=decision_data.get('confidence', 0.5),
            explanation=explanation,
            recommended_provider=decision_data.get('recommended_provider'),
            optimal_amount=Decimal(str(decision_data.get('recommended_amount', workflow_state.request.amount))),
            wait_until=datetime.utcnow() + timedelta(days=wait_days) if wait_days > 0 else None,
            cost_estimate=Decimal(str(decision_data.get('expected_outcome', {}).get('total_cost_estimate', 0))),
            risk_factors=decision_data.get('risk_mitigation', [])
        )
    
    def _build_coordination_reasoning(self, 
                                   decision_data: Dict[str, Any],
                                   market_analysis: Dict[str, Any],
                                   risk_analysis: Dict[str, Any],
                                   cost_analysis: Dict[str, Any]) -> str:
        """Build comprehensive reasoning for the coordination decision."""
        reasoning_parts = []
        
        # Decision summary
        decision = decision_data.get('final_decision', 'convert_now')
        provider = decision_data.get('recommended_provider', 'Unknown')
        confidence = decision_data.get('confidence', 0.5)
        
        reasoning_parts.append(f"Coordination decision: {decision} via {provider} (confidence: {confidence:.2f})")
        
        # Agent consensus
        consensus = decision_data.get('agent_consensus', {})
        consensus_score = consensus.get('consensus_score', 0.5)
        reasoning_parts.append(f"Agent consensus: {consensus_score:.2f}")
        
        # Key influencing factors
        market_conf = market_analysis.get('confidence', 0)
        risk_conf = risk_analysis.get('confidence', 0)
        cost_conf = cost_analysis.get('confidence', 0)
        
        reasoning_parts.append(f"Agent confidences - Market: {market_conf:.2f}, Risk: {risk_conf:.2f}, Cost: {cost_conf:.2f}")
        
        return "; ".join(reasoning_parts)
    
    async def coordinate_multi_agent_decision(self, workflow_state: CurrencyDecisionState) -> DecisionRecommendation:
        """
        Main coordination method to produce final decision from workflow state.
        
        Args:
            workflow_state: Complete workflow state with all agent results
            
        Returns:
            Final decision recommendation
        """
        request_data = {"workflow_state": workflow_state}
        result = await self.process_request(request_data)
        
        if result.success:
            return self._create_decision_recommendation(result.data, workflow_state)
        else:
            # Create fallback recommendation
            return DecisionRecommendation(
                decision=DecisionType.CONVERT_NOW,
                confidence=0.3,
                explanation=f"Coordination failed: {result.error_message}. Defaulting to immediate conversion.",
                recommended_provider="Online Service",
                optimal_amount=workflow_state.request.amount,
                cost_estimate=workflow_state.request.amount * Decimal('0.02')
            )