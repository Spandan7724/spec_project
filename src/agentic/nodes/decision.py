"""Decision coordination agent for LangGraph workflow."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from llm.manager import LLMManager
from ..state import AgentGraphState, Recommendation

logger = logging.getLogger(__name__)


class DecisionCoordinatorAgent:
    """Synthesizes agent outputs into a final recommendation."""

    def __init__(self, llm_manager: Optional[LLMManager] = None) -> None:
        if llm_manager is not None:
            self.llm_manager = llm_manager
        else:
            try:
                self.llm_manager = LLMManager()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLMManager unavailable: %s", exc)
                self.llm_manager = None

    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        request = state.request
        market = state.market_analysis
        economic = state.economic_analysis
        risk = state.risk_assessment
        provider = state.provider_costs

        recommendation = Recommendation()
        heuristic_rationale: List[str] = []
        warnings: List[str] = []

        if market.summary:
            heuristic_rationale.append(f"Market: {market.summary}")
        if economic.summary:
            heuristic_rationale.append(f"Economic: {economic.summary}")
        if risk.summary:
            heuristic_rationale.append(f"Risk: {risk.summary}")
        if provider.summary and provider.status != "unavailable":
            heuristic_rationale.append(f"Provider: {provider.summary}")
        elif provider.errors:
            warnings.append("Provider costs unavailable")

        upstream_errors = market.errors + economic.errors + risk.errors
        if upstream_errors:
            warnings.extend(upstream_errors)

        high_event_risk = len(economic.high_impact_events) > 0 if economic.high_impact_events else False
        risk_level = risk.risk_level or "unknown"
        market_bias = market.bias or "neutral"

        if high_event_risk or risk_level == "high":
            recommendation.action = "wait"
            recommendation.confidence = 0.55 if high_event_risk else 0.5
            recommendation.timeline = (
                "Reassess after key events" if high_event_risk else "Re-evaluate once volatility cools"
            )
            heuristic_rationale.append("High-impact event or elevated volatility detected")
        elif market_bias == "bullish" and risk_level in {"low", "medium"}:
            recommendation.action = "convert_now"
            recommendation.confidence = 0.6
            recommendation.timeline = "Execute within next 24 hours"
            heuristic_rationale.append("Market momentum supportive with manageable risk profile")
        else:
            recommendation.action = "staged_conversion"
            recommendation.confidence = 0.5
            recommendation.timeline = "Split transfers over coming days"
            heuristic_rationale.append("Mixed signals; diversify execution timing")

        recommendation.summary = (
            f"Recommended action: {recommendation.action.replace('_', ' ')} for {request.currency_pair}."
        )
        recommendation.rationale = list(heuristic_rationale)
        recommendation.warnings = list(warnings)

        if upstream_errors:
            recommendation.errors.extend(upstream_errors)

        heuristic_rationale_copy = list(heuristic_rationale)
        warnings_copy = list(warnings)

        # Enrich reasoning with LLM if available
        if self.llm_manager is not None:
            llm_input = {
                "request": {
                    "currency_pair": request.currency_pair,
                    "amount": request.amount,
                    "risk_tolerance": request.risk_tolerance,
                    "timeframe_days": request.timeframe_days,
                },
                "market": {
                    "summary": market.summary,
                    "bias": market.bias,
                    "regime": market.regime,
                    "confidence": market.confidence,
                    "mid_rate": market.mid_rate,
                    "ml_forecasts": market.ml_forecasts,
                },
                "economic": {
                    "summary": economic.summary,
                    "overall_bias": economic.overall_bias,
                    "high_impact_events": economic.high_impact_events,
                },
                "risk": {
                    "summary": risk.summary,
                    "risk_level": risk.risk_level,
                    "volatility": risk.volatility,
                    "var_95": risk.var_95,
                    "scenarios": risk.scenarios,
                },
                "heuristic_recommendation": {
                    "action": recommendation.action,
                    "confidence": recommendation.confidence,
                    "timeline": recommendation.timeline,
                    "rationale": heuristic_rationale_copy,
                    "warnings": warnings_copy,
                },
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert currency conversion timing advisor. "
                        "Given the data, respond with compact JSON containing keys "
                        "summary (string), rationale (array of strings), warnings (array), "
                        "timeline (string), and confidence (float between 0 and 1)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(llm_input, default=str),
                },
            ]

            try:
                llm_response = await self.llm_manager.chat(messages)
                if llm_response and llm_response.content:
                    content = llm_response.content.strip()
                    if content.startswith("```"):
                        content = content.strip("`").strip()
                        if content.lower().startswith("json"):
                            content = content[4:].strip()
                    parsed = json.loads(content)
                    summary = parsed.get("summary")
                    rationale = parsed.get("rationale")
                    timeline = parsed.get("timeline")
                    llm_confidence = parsed.get("confidence")
                    llm_warnings = parsed.get("warnings")

                    if isinstance(summary, str):
                        recommendation.summary = summary
                    if isinstance(rationale, list):
                        recommendation.rationale = [str(item) for item in rationale]
                    if isinstance(timeline, str):
                        recommendation.timeline = timeline
                    if isinstance(llm_confidence, (int, float)):
                        recommendation.confidence = float(llm_confidence)
                    if isinstance(llm_warnings, list):
                        recommendation.warnings.extend(str(item) for item in llm_warnings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM reasoning failed: %s", exc)
                recommendation.warnings.append("LLM reasoning unavailable")

        if recommendation.warnings:
            recommendation.warnings = list(dict.fromkeys(recommendation.warnings))

        return state.with_recommendation(recommendation)


async def run_decision_agent(
    state: AgentGraphState,
    agent: DecisionCoordinatorAgent | None = None,
) -> AgentGraphState:
    agent = agent or DecisionCoordinatorAgent()
    return await agent(state)
