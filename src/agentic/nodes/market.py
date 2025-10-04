"""Market analysis agent node for LangGraph workflow."""

from __future__ import annotations

import logging

from functools import lru_cache

from data_collection.rate_collector import MultiProviderRateCollector
from data_collection.analysis.technical_indicators import TechnicalIndicatorEngine
from ml.prediction.predictor import MLPredictor
from ml.prediction.types import MLPredictionRequest

from ..state import AgentGraphState, MarketAnalysis

logger = logging.getLogger(__name__)


class MarketAnalysisAgent:
    """Generates market context using live rates and technical indicators."""

    def __init__(
        self,
        rate_collector: MultiProviderRateCollector | None = None,
        indicator_engine: TechnicalIndicatorEngine | None = None,
        ml_predictor: MLPredictor | None = None,
    ) -> None:
        self.rate_collector = rate_collector or MultiProviderRateCollector()
        self.indicator_engine = indicator_engine or TechnicalIndicatorEngine()
        if ml_predictor is not None:
            self.ml_predictor = ml_predictor
        else:
            try:
                self.ml_predictor = MLPredictor()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize ML predictor: %s", exc)
                self.ml_predictor = None

    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        request = state.request
        market = MarketAnalysis()
        notes: list[str] = []
        correlation_id = state.meta.correlation_id or "n/a"
        log_extra = {"correlation_id": correlation_id}

        logger.debug("[%s] Market analysis starting", correlation_id, extra=log_extra)

        try:
            # Fetch current rates from available providers
            try:
                rate_result = await self.rate_collector.get_rate(
                    request.base_currency,
                    request.quote_currency,
                )

                if rate_result.has_data:
                    best_rate = rate_result.best_rate or rate_result.rates[0]
                    market.mid_rate = best_rate.mid_rate
                    market.rate_timestamp = best_rate.timestamp
                    market.data_source_notes.append(
                        f"Rates from {len(rate_result.rates)} providers (success {rate_result.success_rate:.0f}%)"
                    )
                    notes.append(
                        f"Base rate {request.currency_pair} = {best_rate.rate:.5f} from {best_rate.source.value}"
                    )
                else:
                    market.errors.append("No live rates available from configured providers")
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[%s] Market agent failed to pull live rates", correlation_id, extra=log_extra, exc_info=exc
                )
                market.errors.append(f"Rate fetch error: {exc}")

            # Compute technical indicators for context
            try:
                indicators = await self.indicator_engine.calculate_indicators(request.currency_pair)
                if indicators:
                    market.technical_signals = indicators.to_dict()
                    market.indicators_used = [
                        "sma_20",
                        "sma_50",
                        "ema_12",
                        "ema_26",
                        "bollinger",
                        "rsi_14",
                        "macd",
                        "volatility",
                    ]
                    market.regime = indicators.trend_direction or "unknown"
                    if indicators.is_bullish:
                        market.bias = "bullish"
                    elif indicators.trend_direction == "down":
                        market.bias = "bearish"
                    else:
                        market.bias = "neutral"
                    market.confidence = 0.6 if indicators.is_bullish else 0.5
                    rsi_display = f"{indicators.rsi_14:.1f}" if indicators.rsi_14 is not None else "n/a"
                    notes.append(
                        f"Trend {indicators.trend_direction or 'n/a'} | RSI {rsi_display}"
                    )
                else:
                    market.errors.append("Insufficient historical data for indicators")
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[%s] Market agent failed to compute indicators", correlation_id, extra=log_extra, exc_info=exc
                )
                market.errors.append(f"Indicator error: {exc}")

            # ML forecasts via predictor
            if self.ml_predictor is not None:
                try:
                    requested_window = max(1, min(30, request.timeframe_days))
                    horizons = sorted({1, requested_window, 7, 30})
                    ml_request = MLPredictionRequest(
                        currency_pair=request.currency_pair,
                        horizons=horizons,
                        include_confidence=True,
                        include_direction_prob=True,
                        max_age_hours=1,
                    )
                    ml_response = await self.ml_predictor.predict(ml_request)
                    available_horizons = sorted({int(h) for h in ml_response.predictions.keys()})

                    if not available_horizons:
                        filtered_predictions = {}
                        filtered_direction_probabilities = {}
                        primary_horizon = requested_window
                    else:
                        primary_horizon = min(
                            available_horizons,
                            key=lambda horizon: (abs(horizon - requested_window), horizon),
                        )
                        filtered_predictions = {}
                        filtered_direction_probabilities = {}
                        for horizon in available_horizons:
                            key = str(horizon)
                            if horizon <= requested_window or horizon == primary_horizon:
                                filtered_predictions[key] = ml_response.predictions.get(key, {})
                                if ml_response.direction_probabilities and key in ml_response.direction_probabilities:
                                    filtered_direction_probabilities[key] = ml_response.direction_probabilities[key]

                    market.ml_forecasts = {
                        "model_id": ml_response.model_id,
                        "model_confidence": ml_response.model_confidence,
                        "predictions": filtered_predictions,
                        "direction_probabilities": filtered_direction_probabilities,
                        "processing_time_ms": ml_response.processing_time_ms,
                        "cached": ml_response.cached,
                        "available_horizons": available_horizons,
                    }
                    market.primary_forecast_horizon = primary_horizon
                    market.timeframe_aligned_forecasts = filtered_predictions
                    market.primary_forecast = filtered_predictions.get(str(primary_horizon), {})
                    notes.append(
                        f"ML model {ml_response.model_id} (confidence {ml_response.model_confidence:.2f})"
                    )
                    if market.confidence is None:
                        market.confidence = ml_response.model_confidence
                    else:
                        market.confidence = max(market.confidence, ml_response.model_confidence)
                    primary_mean = None
                    if isinstance(market.primary_forecast, dict):
                        primary_mean = market.primary_forecast.get("mean")
                    if primary_mean is not None:
                        notes.append(
                            f"{primary_horizon}d ML mean change {primary_mean:+.2%}"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] ML predictor unavailable: %s", correlation_id, exc, extra=log_extra)
                    market.ml_forecasts = {
                        "available": False,
                        "message": f"ML predictor unavailable: {exc}",
                    }
            else:
                market.ml_forecasts = {
                    "available": False,
                    "message": "ML predictor not initialized",
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[%s] Market agent encountered unexpected error", correlation_id, extra=log_extra, exc_info=exc
            )
            market.errors.append(f"Unexpected market agent error: {exc}")

        # Summarize findings if we gathered any data
        if market.mid_rate:
            regime = market.regime or "mixed"
            bias = market.bias or "neutral"
            market.summary = (
                f"Spot {request.currency_pair} around {market.mid_rate:.4f}; "
                f"technical regime {regime}, bias {bias}."
            )
            if market.primary_forecast_horizon and market.primary_forecast:
                mean_change = market.primary_forecast.get("mean")
                if isinstance(mean_change, (int, float)):
                    market.summary += (
                        f" ML {market.primary_forecast_horizon}d mean change {mean_change:+.2%}."
                    )
        elif not market.errors:
            market.summary = "Market data fetched without headline metrics"

        market.data_source_notes.extend(notes)

        logger.debug("[%s] Market analysis completed with %d warning(s)", correlation_id, len(market.errors), extra=log_extra)

        return state.with_market(market)


@lru_cache(maxsize=1)
def _default_market_agent() -> MarketAnalysisAgent:
    """Shared default instance so expensive collectors are reused."""
    return MarketAnalysisAgent()


async def run_market_agent(
    state: AgentGraphState,
    *,
    agent: MarketAnalysisAgent | None = None,
    rate_collector: MultiProviderRateCollector | None = None,
    indicator_engine: TechnicalIndicatorEngine | None = None,
    ml_predictor: MLPredictor | None = None,
) -> AgentGraphState:
    """Convenience coroutine for LangGraph nodes."""
    if agent is None:
        if any(dep is not None for dep in (rate_collector, indicator_engine, ml_predictor)):
            agent = MarketAnalysisAgent(
                rate_collector=rate_collector,
                indicator_engine=indicator_engine,
                ml_predictor=ml_predictor,
            )
        else:
            agent = _default_market_agent()
    return await agent(state)
