"""
Market Intelligence Agent for Currency Decision Making.

Analyzes market conditions, news sentiment, economic events, and technical
indicators to provide insights for currency conversion decisions.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .base_agent import BaseAgent, AgentResult
from ..workflows.state_management import MarketAnalysisResult

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent(BaseAgent):
    """
    Specialized agent for market intelligence and sentiment analysis.
    
    Capabilities:
    - News sentiment analysis for currency pairs
    - Economic calendar event impact assessment
    - Market regime detection (trending vs ranging)
    - Cross-market correlation analysis
    - Technical indicator interpretation
    """
    
    def get_system_prompt(self) -> str:
        """System prompt for market intelligence analysis."""
        return """You are a Market Intelligence Agent specializing in currency markets and foreign exchange analysis.

Your expertise includes:
- Real-time news sentiment analysis for major currency pairs
- Economic event impact assessment (central bank meetings, GDP releases, inflation data)
- Market regime identification (trending, ranging, volatile conditions)
- Technical analysis and chart pattern recognition
- Cross-market correlation analysis (bonds, equities, commodities)

Your Analysis Framework:
1. **Sentiment Analysis**: Evaluate recent news and market commentary
2. **Economic Events**: Assess upcoming/recent economic releases and their impact
3. **Market Regime**: Determine current market conditions (trending/ranging/volatile)
4. **Technical Factors**: Analyze key technical levels and patterns
5. **Risk Factors**: Identify market risks and uncertainty sources

Response Format:
- Provide numerical scores (0.0-1.0) for sentiment, impact, and confidence
- Use clear reasoning for all assessments
- Focus on actionable insights for conversion timing
- Consider both short-term (1-7 days) and medium-term (1-4 weeks) factors

Always be specific about timeframes and confidence levels in your analysis."""
    
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze market intelligence for currency conversion decisions.
        
        Args:
            request_data: Contains currency_pair, amount, timeframe, and context
            
        Returns:
            AgentResult with market analysis and recommendations
        """
        try:
            # Extract request information
            currency_pair = request_data.get('currency_pair', 'Unknown')
            amount = request_data.get('amount', 0)
            timeframe_days = request_data.get('timeframe_days', 7)
            user_risk_tolerance = request_data.get('user_risk_tolerance', 'moderate')
            
            # Build comprehensive analysis prompt
            analysis_prompt = f"""
            Analyze the current market intelligence for {currency_pair} conversion:

            **Conversion Details:**
            - Currency Pair: {currency_pair}
            - Amount: ${amount:,}
            - Decision Timeframe: {timeframe_days} days
            - User Risk Tolerance: {user_risk_tolerance}

            **Analysis Required:**
            
            1. **News Sentiment Analysis** (past 24-48 hours):
               - Recent news impact on {currency_pair}
               - Market sentiment indicators
               - Social media/trader sentiment if relevant
               - Assign sentiment score: -1.0 (very negative) to +1.0 (very positive)
               
            2. **Economic Calendar Impact** (next {timeframe_days} days):
               - Upcoming economic releases affecting both currencies
               - Central bank meetings or speeches
               - GDP, inflation, employment data releases
               - Assign impact score: 0.0 (no impact) to 1.0 (high impact)
               
            3. **Market Regime Detection**:
               - Is {currency_pair} in a trending, ranging, or volatile regime?
               - Recent volatility patterns
               - Support/resistance levels being tested
               - Classify as: "trending_up", "trending_down", "ranging", "volatile"
               
            4. **Technical Analysis**:
               - Key technical levels for {currency_pair}
               - Moving average positions (20, 50, 200-day if relevant)
               - Recent breakouts or pattern formations
               - RSI, MACD signals if available
               
            5. **Cross-Market Analysis**:
               - Bond yield differentials between countries
               - Equity market performance correlation
               - Commodity impacts (oil, gold) on currencies
               - Risk-on vs risk-off sentiment
               
            **Required Output Format:**
            Return a JSON-structured response with these exact fields:
            {{
                "sentiment_score": <float between -1.0 and 1.0>,
                "news_impact": <float between 0.0 and 1.0>,
                "economic_events": [
                    {{"date": "YYYY-MM-DD", "event": "Event Name", "importance": "high/medium/low", "expected_impact": "positive/negative/neutral"}}
                ],
                "market_regime": "<trending_up/trending_down/ranging/volatile>",
                "technical_indicators": {{
                    "rsi": <value if available, else null>,
                    "moving_average_signal": "bullish/bearish/neutral",
                    "key_resistance": <level if known>,
                    "key_support": <level if known>
                }},
                "confidence": <float between 0.0 and 1.0>,
                "reasoning": "<detailed explanation of analysis>",
                "timing_recommendation": "<immediate/wait_1_3_days/wait_1_week/unclear>",
                "risk_factors": ["<list of key risks to consider>"]
            }}
            
            Focus on practical, actionable intelligence that helps with conversion timing decisions.
            Be honest about data limitations and uncertainty levels.
            """
            
            # Get LLM analysis
            response = await self.chat_with_llm(analysis_prompt, include_tools=False)
            
            # Parse the structured response
            analysis_data = self._parse_llm_response(response.content, currency_pair)
            
            # Calculate overall confidence
            confidence = analysis_data.get('confidence', 0.5)
            
            # Build reasoning summary
            reasoning = f"Market intelligence analysis for {currency_pair}: "
            reasoning += f"Sentiment {analysis_data.get('sentiment_score', 0):.2f}, "
            reasoning += f"News impact {analysis_data.get('news_impact', 0):.2f}, "
            reasoning += f"Regime: {analysis_data.get('market_regime', 'unknown')}"
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data=analysis_data,
                reasoning=reasoning,
                confidence=confidence,
                execution_time_ms=getattr(response, 'execution_time_ms', None)
            )
            
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent failed: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                reasoning=f"Market intelligence analysis failed: {str(e)}",
                error_message=str(e),
                confidence=0.0
            )
    
    def _parse_llm_response(self, content: str, currency_pair: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data."""
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
                return self._extract_from_text(content, currency_pair)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, using text extraction")
            return self._extract_from_text(content, currency_pair)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_data(currency_pair)
    
    def _validate_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize parsed JSON data."""
        validated = {
            "sentiment_score": max(-1.0, min(1.0, float(data.get('sentiment_score', 0.0)))),
            "news_impact": max(0.0, min(1.0, float(data.get('news_impact', 0.0)))),
            "economic_events": data.get('economic_events', []),
            "market_regime": str(data.get('market_regime', 'unknown')).lower(),
            "technical_indicators": data.get('technical_indicators', {}),
            "confidence": max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
            "reasoning": str(data.get('reasoning', 'Analysis completed')),
            "timing_recommendation": str(data.get('timing_recommendation', 'unclear')),
            "risk_factors": data.get('risk_factors', [])
        }
        
        # Ensure valid market regime
        valid_regimes = ['trending_up', 'trending_down', 'ranging', 'volatile', 'unknown']
        if validated['market_regime'] not in valid_regimes:
            validated['market_regime'] = 'unknown'
            
        return validated
    
    def _extract_from_text(self, content: str, currency_pair: str) -> Dict[str, Any]:
        """Extract key information from text when JSON parsing fails."""
        # Simple text-based extraction as fallback
        sentiment_score = 0.0
        news_impact = 0.0
        confidence = 0.5
        market_regime = "unknown"
        
        # Look for numerical values and keywords
        content_lower = content.lower()
        
        # Extract sentiment indicators
        if "positive" in content_lower or "bullish" in content_lower:
            sentiment_score = 0.3
        elif "negative" in content_lower or "bearish" in content_lower:
            sentiment_score = -0.3
        elif "very positive" in content_lower:
            sentiment_score = 0.7
        elif "very negative" in content_lower:
            sentiment_score = -0.7
            
        # Extract regime indicators
        if "trending" in content_lower:
            if "up" in content_lower or "bullish" in content_lower:
                market_regime = "trending_up"
            elif "down" in content_lower or "bearish" in content_lower:
                market_regime = "trending_down"
        elif "ranging" in content_lower or "sideways" in content_lower:
            market_regime = "ranging"
        elif "volatile" in content_lower:
            market_regime = "volatile"
            
        # Extract impact level
        if "high impact" in content_lower or "significant" in content_lower:
            news_impact = 0.8
        elif "medium impact" in content_lower or "moderate" in content_lower:
            news_impact = 0.5
        elif "low impact" in content_lower:
            news_impact = 0.3
        
        return {
            "sentiment_score": sentiment_score,
            "news_impact": news_impact,
            "economic_events": [],
            "market_regime": market_regime,
            "technical_indicators": {},
            "confidence": confidence,
            "reasoning": content[:200] + "..." if len(content) > 200 else content,
            "timing_recommendation": "unclear",
            "risk_factors": []
        }
    
    def _create_fallback_data(self, currency_pair: str) -> Dict[str, Any]:
        """Create fallback data when all parsing fails."""
        return {
            "sentiment_score": 0.0,
            "news_impact": 0.0,
            "economic_events": [],
            "market_regime": "unknown",
            "technical_indicators": {},
            "confidence": 0.1,  # Very low confidence for fallback
            "reasoning": f"Unable to complete full market analysis for {currency_pair}",
            "timing_recommendation": "unclear",
            "risk_factors": ["Limited market data available"]
        }
    
    async def analyze_currency_sentiment(self, currency_pair: str, timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Focused sentiment analysis for a currency pair.
        
        Args:
            currency_pair: Currency pair to analyze
            timeframe_hours: Hours of recent data to analyze
            
        Returns:
            Sentiment analysis results
        """
        request_data = {
            'currency_pair': currency_pair,
            'timeframe_days': timeframe_hours // 24,
            'analysis_type': 'sentiment_only'
        }
        
        result = await self.process_request(request_data)
        
        if result.success:
            return {
                'sentiment_score': result.data.get('sentiment_score', 0.0),
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }
        else:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'reasoning': result.error_message or 'Sentiment analysis failed'
            }