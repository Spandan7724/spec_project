"""Decision coordination agent for LangGraph workflow."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from datetime import datetime, timezone
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
        correlation_id = state.meta.correlation_id or "n/a"
        log_extra = {"correlation_id": correlation_id}

        logger.debug("[%s] Decision coordination starting", correlation_id, extra=log_extra)

        recommendation = Recommendation()
        heuristic_rationale: List[str] = []
        warnings: List[str] = []

        try:
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

            timeframe = max(1, request.timeframe_days)
            risk_level = risk.risk_level or "unknown"
            market_bias = market.bias or "neutral"

            high_impact_events = economic.high_impact_events or []
            high_event_count = len(high_impact_events)
            event_density = high_event_count / timeframe if timeframe else 0
            now_utc = datetime.now(timezone.utc)
            upcoming_event_offsets: List[float] = []

            for event in high_impact_events:
                release_at = event.get("release_date") if isinstance(event, dict) else None
                if not release_at:
                    continue
                try:
                    event_dt = datetime.fromisoformat(release_at)
                except ValueError:
                    continue
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    event_dt = event_dt.astimezone(timezone.utc)
                offset_days = (event_dt - now_utc).total_seconds() / 86400
                if offset_days >= -0.5:
                    upcoming_event_offsets.append(offset_days)

            next_event_days = None
            if upcoming_event_offsets:
                non_negative = [value for value in upcoming_event_offsets if value >= 0]
                if non_negative:
                    next_event_days = min(non_negative)

            predictions_dict = {}
            direction_prob_dict = {}
            if isinstance(market.ml_forecasts, dict):
                predictions_dict = market.ml_forecasts.get("predictions") or {}
                direction_prob_dict = market.ml_forecasts.get("direction_probabilities") or {}

            primary_horizon = market.primary_forecast_horizon
            if primary_horizon is None:
                try:
                    primary_horizon = min(
                        (int(key) for key in predictions_dict.keys()),
                        key=lambda horizon: (abs(horizon - timeframe), horizon),
                    )
                except ValueError:
                    primary_horizon = None

            primary_key = str(primary_horizon) if primary_horizon is not None else None
            primary_forecast = market.primary_forecast if isinstance(market.primary_forecast, dict) else {}
            if (not primary_forecast) and primary_key:
                primary_forecast = predictions_dict.get(primary_key, {})
            primary_mean = None
            if isinstance(primary_forecast, dict):
                raw_mean = primary_forecast.get("mean")
                if isinstance(raw_mean, (int, float)):
                    primary_mean = raw_mean
            primary_prob_up = direction_prob_dict.get(primary_key) if primary_key else None

            if primary_horizon is None and timeframe:
                primary_horizon = timeframe

            if primary_mean is not None:
                heuristic_rationale.append(
                    f"ML {primary_horizon}d outlook mean change {primary_mean:+.2%}"
                )

            if high_event_count:
                heuristic_rationale.append(
                    f"{high_event_count} high-impact event(s) scheduled within {timeframe} day(s)"
                )

            STRONG_FORECAST = 0.002  # 0.2%
            MODERATE_FORECAST = 0.0005  # 0.05%
            forecast_strength = abs(primary_mean) if primary_mean is not None else 0.0
            ml_trend: Optional[str] = None
            if primary_mean is not None and forecast_strength >= MODERATE_FORECAST:
                if forecast_strength >= STRONG_FORECAST:
                    ml_trend = "strong_bullish" if primary_mean > 0 else "strong_bearish"
                else:
                    ml_trend = "bullish" if primary_mean > 0 else "bearish"

            wait_reasons: List[str] = []
            if risk_level == "high":
                wait_reasons.append("Risk assessment flags high volatility")
            if high_event_count and (timeframe <= 3 or event_density >= 0.25):
                wait_reasons.append("Cluster of high-impact events in this window")
            if next_event_days is not None and next_event_days <= 1.5:
                wait_reasons.append("Next high-impact event is due within 36 hours")
            if ml_trend == "strong_bearish" and risk_level != "low":
                wait_reasons.append("ML outlook projects notable downside")

            if wait_reasons:
                recommendation.action = "wait"
                base_confidence = 0.55 + (0.05 if len(wait_reasons) > 1 else 0.0)
                if ml_trend == "strong_bearish":
                    base_confidence += 0.05
                if high_event_count >= 3:
                    base_confidence += 0.05
                recommendation.confidence = min(base_confidence, 0.85)

                if next_event_days is not None:
                    horizon_days = max(1, int(next_event_days + 0.5))
                    recommendation.timeline = (
                        f"Reassess in ~{horizon_days} day(s) after the imminent events conclude"
                    )
                else:
                    recommendation.timeline = (
                        f"Reassess midway through the {timeframe}-day window"
                    )
                heuristic_rationale.extend(wait_reasons)
            else:
                supportive_trend = (
                    market_bias == "bullish" and risk_level in {"low", "moderate"}
                ) or (ml_trend in {"strong_bullish", "bullish"} and risk_level != "high")

                if supportive_trend:
                    strong_signal = ml_trend == "strong_bullish" or market_bias == "bullish"
                    recommendation.action = "convert_now" if timeframe <= 7 else "staged_conversion"
                    if recommendation.action == "convert_now":
                        recommendation.confidence = 0.62 if strong_signal else 0.57
                        window_days = min(timeframe, 2)
                        recommendation.timeline = (
                            f"Execute within next {window_days} day(s) to capture favorable momentum"
                        )
                        heuristic_rationale.append(
                            "Market momentum and ML outlook favour acting promptly"
                        )
                    else:
                        recommendation.confidence = 0.56 if strong_signal else 0.53
                        tranche_window = max(2, timeframe // 3 or 2)
                        recommendation.timeline = (
                            f"Stage conversions over ~{tranche_window} day(s) within the {timeframe}-day window"
                        )
                        heuristic_rationale.append(
                            "Longer window allows staggered execution while conditions stay supportive"
                        )
                else:
                    recommendation.action = "staged_conversion"
                    recommendation.confidence = 0.52 if ml_trend == "bearish" else 0.5
                    tranche_window = max(1, min(timeframe, (timeframe // 2) or 1))
                    recommendation.timeline = (
                        f"Split transfers across the next {tranche_window} day(s) to hedge mixed signals"
                    )
                    if ml_trend == "bearish":
                        heuristic_rationale.append(
                            "ML outlook tilts bearish; gradual execution limits downside risk"
                        )
                    else:
                        heuristic_rationale.append("Mixed signals; diversify execution timing")

            if primary_prob_up is not None:
                heuristic_rationale.append(
                    f"Direction probability (up) at {primary_horizon}d: {primary_prob_up:.0%}"
                )

            recommendation.summary = (
                f"Recommended action: {recommendation.action.replace('_', ' ')} for {request.currency_pair}."
            )
            recommendation.rationale = list(heuristic_rationale)
            recommendation.warnings = list(warnings)

            if upstream_errors:
                recommendation.errors.extend(upstream_errors)

            heuristic_rationale_copy = list(heuristic_rationale)
            warnings_copy = list(warnings)

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%s] Decision agent encountered unexpected error",
                correlation_id,
                extra=log_extra,
                exc_info=exc,
            )
            recommendation.action = recommendation.action or "wait"
            recommendation.summary = (
                f"Recommended action: {recommendation.action.replace('_', ' ')} for {request.currency_pair}."
                if recommendation.action
                else f"Recommendation unavailable for {request.currency_pair}."
            )
            recommendation.errors.append(f"Decision agent error: {exc}")
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
                logger.warning("[%s] LLM reasoning failed: %s", correlation_id, exc, extra=log_extra)
                recommendation.warnings.append("LLM reasoning unavailable")

        if recommendation.warnings:
            recommendation.warnings = list(dict.fromkeys(recommendation.warnings))

        logger.debug(
            "[%s] Decision coordination completed; action=%s, warnings=%d",
            correlation_id,
            recommendation.action,
            len(recommendation.warnings),
            extra=log_extra,
        )

        return state.with_recommendation(recommendation)


@lru_cache(maxsize=1)
def _default_decision_agent() -> DecisionCoordinatorAgent:
    """Shared default decision agent instance."""
    return DecisionCoordinatorAgent()


async def run_decision_agent(
    state: AgentGraphState,
    *,
    agent: DecisionCoordinatorAgent | None = None,
    llm_manager: LLMManager | None = None,
) -> AgentGraphState:
    if agent is None:
        if llm_manager is not None:
            agent = DecisionCoordinatorAgent(llm_manager=llm_manager)
        else:
            agent = _default_decision_agent()
    return await agent(state)
