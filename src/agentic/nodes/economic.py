"""Economic analysis agent node for LangGraph workflow."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from data_collection.economic.calendar_collector import EconomicCalendarCollector, EconomicEvent

from ..state import AgentGraphState, EconomicAnalysis

logger = logging.getLogger(__name__)


def _event_to_dict(event: EconomicEvent) -> dict:
    """Serialize EconomicEvent to a lightweight dict."""
    return {
        "event_id": event.event_id,
        "title": event.title,
        "currency": event.currency,
        "country": event.country,
        "release_date": event.release_date.isoformat(),
        "impact": event.impact.value,
        "status": event.status.value,
        "forecast": event.forecast_value,
        "previous": event.previous_value,
        "actual": event.actual_value,
        "relevance": event.relevance_score,
    }


class EconomicAnalysisAgent:
    """Evaluates upcoming economic catalysts relevant to the request."""

    def __init__(self, calendar_collector: EconomicCalendarCollector | None = None) -> None:
        self.calendar_collector = calendar_collector or EconomicCalendarCollector()

    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        request = state.request
        economic = EconomicAnalysis(event_window_days=request.timeframe_days)
        correlation_id = state.meta.correlation_id or "n/a"
        log_extra = {"correlation_id": correlation_id}

        logger.debug("[%s] Economic analysis starting", correlation_id, extra=log_extra)

        try:
            calendar = await self.calendar_collector.get_economic_calendar(
                days_ahead=request.timeframe_days
            )
            if not calendar:
                economic.errors.append("Economic calendar unavailable; check API keys and network access")
                return state.with_economic(economic)

            events_for_pair: List[EconomicEvent] = calendar.get_events_for_pair(request.currency_pair)
            high_impact = [event for event in events_for_pair if event.is_high_impact]

            economic.upcoming_events = [_event_to_dict(event) for event in events_for_pair]
            economic.high_impact_events = [_event_to_dict(event) for event in high_impact]
            economic.data_source_notes.append(
                f"Sources: {', '.join(calendar.sources) if calendar.sources else 'unknown'}"
            )

            total_events = len(events_for_pair)
            high_count = len(high_impact)

            if high_count > 0:
                economic.overall_bias = "risk_off"
                economic.summary = (
                    f"{high_count} high-impact event(s) for {request.currency_pair} within {request.timeframe_days} days;"
                    " expect elevated volatility."
                )
                economic.confidence = 0.6
            elif total_events > 0:
                economic.overall_bias = "neutral"
                economic.summary = (
                    f"{total_events} upcoming event(s) with no high-impact risk flagged for {request.currency_pair}."
                )
                economic.confidence = 0.5
            else:
                economic.overall_bias = "neutral"
                economic.summary = (
                    f"No major economic events for {request.currency_pair} over the next {request.timeframe_days} days."
                )
                economic.confidence = 0.4

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%s] Economic agent failed to gather calendar data",
                correlation_id,
                extra=log_extra,
                exc_info=exc,
            )
            economic.errors.append(f"Economic data error: {exc}")
        else:
            # Ensure unexpected exceptions surface as warnings rather than halting execution.
            pass
        finally:
            if not economic.summary and not economic.errors:
                economic.summary = "Economic analysis completed without notable events"

        logger.debug(
            "[%s] Economic analysis completed with %d warning(s)",
            correlation_id,
            len(economic.errors),
            extra=log_extra,
        )

        return state.with_economic(economic)


@lru_cache(maxsize=1)
def _default_economic_agent() -> EconomicAnalysisAgent:
    """Shared default instance for reuse across requests."""
    return EconomicAnalysisAgent()


async def run_economic_agent(
    state: AgentGraphState,
    *,
    agent: EconomicAnalysisAgent | None = None,
    calendar_collector: EconomicCalendarCollector | None = None,
) -> AgentGraphState:
    """Convenience coroutine for LangGraph nodes."""
    if agent is None:
        if calendar_collector is not None:
            agent = EconomicAnalysisAgent(calendar_collector=calendar_collector)
        else:
            agent = _default_economic_agent()
    return await agent(state)
