"""
Service layer for CLI - integrates with existing agentic workflow
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.agentic.graph import arun_agentic_workflow
from src.agentic.state import AgentGraphState
from src.agentic.response import serialize_state
from src.llm.manager import LLMManager
from src.cli.config import ConfigManager
from src.data_collection.rate_collector import MultiProviderRateCollector
from src.data_collection.economic.calendar_collector import EconomicCalendarCollector
from src.ml.prediction.predictor import MLPredictor

logger = logging.getLogger(__name__)


@dataclass
class AdvisorResult:
    """Result from a currency advisor query"""

    request_summary: str
    recommendation: str
    confidence: float
    action: str
    timeline: str
    rationale: List[str]
    warnings: List[str]
    market_analysis: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    processing_time: float
    correlation_id: str
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display"""
        return {
            "request_summary": self.request_summary,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "action": self.action,
            "timeline": self.timeline,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "market_analysis": self.market_analysis,
            "economic_analysis": self.economic_analysis,
            "risk_assessment": self.risk_assessment,
            "processing_time": self.processing_time,
            "correlation_id": self.correlation_id,
            "errors": self.errors,
        }


class CurrencyAdvisorService:
    """Main service layer for currency advisor CLI"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)

        # Initialize components (will be created lazily if needed)
        self._llm_manager = None
        self._rate_collector = None
        self._economic_collector = None
        self._ml_predictor = None
        self._config_manager = ConfigManager()
        self._default_request = self._config_manager.get_effective_defaults()

    @property
    def llm_manager(self) -> Optional[LLMManager]:
        """Lazy initialization of LLM manager"""
        if self._llm_manager is None:
            try:
                self._llm_manager = LLMManager()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Manager: {e}")
                self._llm_manager = None
        return self._llm_manager

    @property
    def rate_collector(self) -> Optional[MultiProviderRateCollector]:
        """Lazy initialization of rate collector"""
        if self._rate_collector is None:
            try:
                self._rate_collector = MultiProviderRateCollector()
            except Exception as e:
                logger.warning(f"Failed to initialize Rate Collector: {e}")
                self._rate_collector = None
        return self._rate_collector

    @property
    def economic_collector(self) -> Optional[EconomicCalendarCollector]:
        """Lazy initialization of economic collector"""
        if self._economic_collector is None:
            try:
                self._economic_collector = EconomicCalendarCollector()
            except Exception as e:
                logger.warning(f"Failed to initialize Economic Collector: {e}")
                self._economic_collector = None
        return self._economic_collector

    @property
    def ml_predictor(self) -> Optional[MLPredictor]:
        """Lazy initialization of ML predictor"""
        if self._ml_predictor is None:
            try:
                self._ml_predictor = MLPredictor()
            except Exception as e:
                logger.warning(f"Failed to initialize ML Predictor: {e}")
                self._ml_predictor = None
        return self._ml_predictor

    async def ask_question(self, question: str) -> AdvisorResult:
        """
        Process a natural language question and return a recommendation

        Args:
            question: Natural language question about currency conversion

        Returns:
            AdvisorResult with recommendation and analysis
        """
        start_time = datetime.now()

        if not self._has_currency_intent(question):
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._smalltalk_response(question, processing_time)

        try:
            # Parse natural language to structured request
            request_payload = await self._parse_question(question)

            # Run the agentic workflow
            result_state = await arun_agentic_workflow(request_payload)

            # Extract and format the result
            processing_time = (datetime.now() - start_time).total_seconds()

            return self._format_result(result_state, processing_time, question)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return AdvisorResult(
                request_summary=f"Failed to process: {question}",
                recommendation="I'm sorry, I encountered an error while processing your request.",
                confidence=0.0,
                action="error",
                timeline="N/A",
                rationale=[f"Error: {str(e)}"],
                warnings=["Service temporarily unavailable"],
                market_analysis={},
                economic_analysis={},
                risk_assessment={},
                processing_time=processing_time,
                correlation_id="error",
                errors=[str(e)]
            )

    def _has_currency_intent(self, question: str) -> bool:
        """Return True when the text looks like a real currency-conversion request."""
        text = question.strip()
        if not text:
            return False

        upper = text.upper()

        patterns = (
            re.compile(r"\b([A-Z]{3})/([A-Z]{3})\b"),
            re.compile(r"\b([A-Z]{3})\s+TO\s+([A-Z]{3})\b"),
        )
        if any(pattern.search(upper) for pattern in patterns):
            return True

        currency_tokens = {
            "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR", "NZD", "SEK",
            "NOK", "SGD", "MXN", "ZAR", "HKD", "KRW", "BRL", "ILS", "AED",
        }
        words = {token.strip(".,!?") for token in upper.split()}
        if currency_tokens & words:
            return True

        keywords = {"CONVERT", "EXCHANGE", "TRANSFER", "HEDGE", "RATE", "OUTLOOK"}
        if keywords & words:
            return True

        if re.search(r"\d", text) and re.search(r"\b(CONVERT|EXCHANGE|RATE)\b", upper):
            return True

        return False

    def _smalltalk_response(self, question: str, processing_time: float) -> AdvisorResult:
        """Return a gentle nudge asking for more details instead of running the graph."""
        summary = (
            "Hi there! To run an analysis, let me know the currency pair (for example USD/EUR), "
            "the amount, and roughly when you plan to convert."
        )
        rationale = [
            "I need a currency pair like USD/EUR or GBP to USD",
            "Optionally include an amount (e.g. $1500) and timing (this week, next month, etc.)",
        ]

        return AdvisorResult(
            request_summary=f"Conversation prompt: {question.strip() or 'greeting'}",
            recommendation=summary,
            confidence=0.0,
            action="info",
            timeline="Awaiting your currency details",
            rationale=rationale,
            warnings=[],
            market_analysis={},
            economic_analysis={},
            risk_assessment={},
            processing_time=processing_time,
            correlation_id="n/a",
            errors=[],
        )

    async def _parse_question(self, question: str) -> Dict[str, Any]:
        """
        Parse natural language question into structured request payload

        Args:
            question: Natural language question

        Returns:
            Dictionary compatible with AgentRequest.from_payload()
        """
        # This is a simple parser - in a production system you'd use NLP
        question_lower = question.lower()

        defaults = self._default_request
        payload = {
            "currency_pair": defaults.get("currency_pair"),
            "amount": float(defaults.get("amount", 1000.0)),
            "risk_tolerance": defaults.get("risk_tolerance", "moderate"),
            "timeframe_days": int(defaults.get("timeframe_days", 7)),
            "user_notes": question
        }

        # Extract currency pairs
        currency_pairs = self._extract_currency_pairs(question)
        if currency_pairs:
            payload["currency_pair"] = currency_pairs[0]

        # Extract amount
        amount = self._extract_amount(question)
        if amount:
            payload["amount"] = amount

        # Extract timeframe
        timeframe = self._extract_timeframe(question)
        if timeframe:
            payload["timeframe_days"] = timeframe

        # Extract risk tolerance
        risk_tolerance = self._extract_risk_tolerance(question)
        if risk_tolerance:
            payload["risk_tolerance"] = risk_tolerance

        if not payload["currency_pair"]:
            raise ValueError(
                "Could not identify currency pair in question and no default is configured. "
                "Update defaults via 'currency-assistant config --set default_base_currency=USD' etc."
            )

        return payload

    def _extract_currency_pairs(self, text: str) -> List[str]:
        """Extract currency pairs from text"""
        import re

        # Common patterns
        patterns = [
            r'([A-Z]{3})/([A-Z]{3})',  # USD/EUR format
            r'([A-Z]{3})\s+to\s+([A-Z]{3})',  # USD to EUR format
            r'(\$|€|£|¥)\s*\d+.*?([A-Z]{3})',  # $1000 USD format
        ]

        pairs = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    pair = f"{match[0].upper()}/{match[1].upper()}"
                    if pair not in pairs:
                        pairs.append(pair)

        return pairs

    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from text"""
        import re

        # Look for currency symbols and numbers
        patterns = [
            r'[$€£¥]\s*([\d,]+(?:\.\d+)?)',  # $1000, €1,000.50
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|GBP|JPY)',  # 1000 USD
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|euros?|pounds?)',  # 1000 dollars
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue

        return None

    def _extract_timeframe(self, text: str) -> Optional[int]:
        """Extract timeframe in days from text"""
        import re

        # Pattern matching for time expressions
        patterns = [
            (r'(\d+)\s+days?', lambda m: int(m.group(1))),
            (r'(\d+)\s+weeks?', lambda m: int(m.group(1)) * 7),
            (r'(\d+)\s+months?', lambda m: int(m.group(1)) * 30),
            (r'today|now', lambda m: 1),
            (r'tomorrow', lambda m: 2),
            (r'this week', lambda m: 7),
            (r'next week', lambda m: 14),
        ]

        text_lower = text.lower()
        for pattern, converter in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return converter(match)
                except (ValueError, AttributeError):
                    continue

        return None

    def _extract_risk_tolerance(self, text: str) -> Optional[str]:
        """Extract risk tolerance from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['conservative', 'safe', 'low risk']):
            return 'low'
        elif any(word in text_lower for word in ['aggressive', 'risky', 'high risk']):
            return 'high'
        elif any(word in text_lower for word in ['moderate', 'balanced', 'medium']):
            return 'moderate'

        return None

    def _format_result(self, state: AgentGraphState, processing_time: float, original_question: str) -> AdvisorResult:
        """Format the agentic workflow result into AdvisorResult"""

        rec = state.recommendation
        market = state.market_analysis
        economic = state.economic_analysis
        risk = state.risk_assessment

        return AdvisorResult(
            request_summary=f"Analysis for {state.request.currency_pair} conversion of {state.request.amount} {state.request.base_currency}",
            recommendation=rec.summary or "No recommendation available",
            confidence=rec.confidence or 0.0,
            action=rec.action or "unknown",
            timeline=rec.timeline or "Not specified",
            rationale=rec.rationale or [],
            warnings=rec.warnings or [],
            market_analysis={
                "summary": market.summary,
                "bias": market.bias,
                "confidence": market.confidence,
                "rate": market.mid_rate,
                "regime": market.regime,
                "errors": market.errors,
            },
            economic_analysis={
                "summary": economic.summary,
                "bias": economic.overall_bias,
                "high_impact_events": len(economic.high_impact_events),
                "errors": economic.errors,
            },
            risk_assessment={
                "summary": risk.summary,
                "risk_level": risk.risk_level,
                "volatility": risk.volatility,
                "var_95": risk.var_95,
                "errors": risk.errors,
            },
            processing_time=processing_time,
            correlation_id=state.meta.correlation_id or "unknown",
            errors=rec.errors or []
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "overall_health": "unknown"
        }

        # Check LLM Manager
        try:
            if self.llm_manager:
                providers = self.llm_manager.list_providers()
                healthy_providers = [name for name, info in providers.items() if info['healthy']]

                status["components"]["llm_manager"] = {
                    "status": "healthy" if healthy_providers else "unhealthy",
                    "healthy_providers": healthy_providers,
                    "total_providers": list(providers.keys())
                }
            else:
                status["components"]["llm_manager"] = {
                    "status": "unavailable",
                    "error": "Failed to initialize"
                }
        except Exception as e:
            status["components"]["llm_manager"] = {
                "status": "error",
                "error": str(e)
            }

        # Check Rate Collector
        try:
            if self.rate_collector:
                providers_info = self.rate_collector.get_provider_info()
                status["components"]["rate_collector"] = {
                    "status": "healthy",
                    "providers_count": len(providers_info),
                    "providers": [info['name'] for info in providers_info]
                }
            else:
                status["components"]["rate_collector"] = {
                    "status": "unavailable",
                    "error": "Failed to initialize"
                }
        except Exception as e:
            status["components"]["rate_collector"] = {
                "status": "error",
                "error": str(e)
            }

        # Check Economic Collector
        try:
            if self.economic_collector:
                status["components"]["economic_collector"] = {
                    "status": "healthy"
                }
            else:
                status["components"]["economic_collector"] = {
                    "status": "unavailable",
                    "error": "Failed to initialize"
                }
        except Exception as e:
            status["components"]["economic_collector"] = {
                "status": "error",
                "error": str(e)
            }

        # Check ML Predictor
        try:
            if self.ml_predictor:
                status["components"]["ml_predictor"] = {
                    "status": "healthy"
                }
            else:
                status["components"]["ml_predictor"] = {
                    "status": "unavailable",
                    "error": "Failed to initialize"
                }
        except Exception as e:
            status["components"]["ml_predictor"] = {
                "status": "error",
                "error": str(e)
            }

        # Determine overall health
        component_statuses = [comp.get("status", "unknown") for comp in status["components"].values()]

        if all(status in ["healthy"] for status in component_statuses):
            status["overall_health"] = "healthy"
        elif any(status in ["error"] for status in component_statuses):
            status["overall_health"] = "degraded"
        elif any(status in ["unavailable"] for status in component_statuses):
            status["overall_health"] = "partial"
        else:
            status["overall_health"] = "unknown"

        return status
