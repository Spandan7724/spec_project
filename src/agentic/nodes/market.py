"""Market analysis agent node for LangGraph workflow."""

from __future__ import annotations

import logging

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
            logger.exception("Market agent failed to pull live rates", exc_info=exc)
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
            logger.exception("Market agent failed to compute indicators", exc_info=exc)
            market.errors.append(f"Indicator error: {exc}")

        # ML forecasts via predictor
        if self.ml_predictor is not None:
            try:
                horizons = sorted({1, min(7, max(1, request.timeframe_days)), 7, 30})
                ml_request = MLPredictionRequest(
                    currency_pair=request.currency_pair,
                    horizons=horizons,
                    include_confidence=True,
                    include_direction_prob=True,
                    max_age_hours=1,
                )
                ml_response = await self.ml_predictor.predict(ml_request)
                market.ml_forecasts = {
                    "model_id": ml_response.model_id,
                    "model_confidence": ml_response.model_confidence,
                    "predictions": ml_response.predictions,
                    "direction_probabilities": ml_response.direction_probabilities,
                    "processing_time_ms": ml_response.processing_time_ms,
                    "cached": ml_response.cached,
                }
                notes.append(
                    f"ML model {ml_response.model_id} (confidence {ml_response.model_confidence:.2f})"
                )
                if market.confidence is None:
                    market.confidence = ml_response.model_confidence
                else:
                    market.confidence = max(market.confidence, ml_response.model_confidence)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML predictor unavailable: %s", exc)
                market.ml_forecasts = {
                    "available": False,
                    "message": f"ML predictor unavailable: {exc}",
                }
        else:
            market.ml_forecasts = {
                "available": False,
                "message": "ML predictor not initialized",
            }

        # Summarize findings if we gathered any data
        if market.mid_rate:
            regime = market.regime or "mixed"
            bias = market.bias or "neutral"
            market.summary = (
                f"Spot {request.currency_pair} around {market.mid_rate:.4f}; "
                f"technical regime {regime}, bias {bias}."
            )
        elif not market.errors:
            market.summary = "Market data fetched without headline metrics"

        market.data_source_notes.extend(notes)

        return state.with_market(market)


async def run_market_agent(state: AgentGraphState, agent: MarketAnalysisAgent | None = None) -> AgentGraphState:
    """Convenience coroutine for LangGraph nodes."""
    agent = agent or MarketAnalysisAgent()
    return await agent(state)
