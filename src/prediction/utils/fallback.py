import logging
from typing import Dict

import numpy as np

from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.models import (
    HorizonPrediction,
    PredictionQuality,
    PredictionRequest,
    PredictionResponse,
)


logger = logging.getLogger(__name__)


class FallbackPredictor:
    """Heuristic-based fallback when ML models are unavailable or fail.

    Uses simple technical analysis rules (MA crossovers, RSI extremes)
    to produce conservative signals with low confidence.
    """

    def __init__(self, config):
        self.config = config
        self.loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(self.config.technical_indicators)

    async def predict(
        self, request: PredictionRequest, base: str, quote: str
    ) -> PredictionResponse:
        logger.info("Using fallback heuristic predictor for %s/%s", base, quote)
        try:
            df = await self.loader.fetch_historical_data(base, quote, days=max(100, self.config.min_history_days))
            if df is None or len(df) < 30:
                return self._empty_response(request, "Insufficient data for fallback")

            close = float(df["Close"].iloc[-1])

            sma_20 = float(df["Close"].rolling(20).mean().iloc[-1])
            sma_50 = float(
                df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["Close"].rolling(20).mean().iloc[-1]
            )

            rsi = float(self._calculate_rsi(df["Close"], period=14))
            volatility = float(df["Close"].pct_change().rolling(20).std().iloc[-1] or 0.0)

            strength = float(getattr(self.config, "fallback_strength_pct", 0.15))

            predictions: Dict[int, HorizonPrediction] = {}

            # Daily horizons
            for horizon in (request.horizons or []):
                ma_signal = self._get_ma_signal(close, sma_20, sma_50, strength)
                rsi_signal = self._get_rsi_signal(rsi, strength)
                mean_change = ma_signal + rsi_signal

                direction_prob = 0.5 + (mean_change / (2 * strength))
                direction_prob = max(0.3, min(0.7, direction_prob))

                quantiles = None
                if request.include_quantiles:
                    uncertainty = volatility * 100.0 * max(1, int(horizon))
                    quantiles = {
                        "p10": mean_change - uncertainty * 1.5,
                        "p50": mean_change,
                        "p90": mean_change + uncertainty * 1.5,
                    }
                predictions[int(horizon)] = HorizonPrediction(
                    horizon=int(horizon),
                    mean_change_pct=float(mean_change),
                    quantiles=quantiles,
                    direction_probability=direction_prob if request.include_direction_probabilities else None,
                )

            # Intraday horizons (expressed in hours in request.intraday_horizons_hours)
            for horizon in (request.intraday_horizons_hours or []):
                ma_signal = self._get_ma_signal(close, sma_20, sma_50, strength)
                rsi_signal = self._get_rsi_signal(rsi, strength)
                mean_change = ma_signal + rsi_signal

                quantiles = None
                if request.include_quantiles:
                    uncertainty = volatility * 100.0 * max(1, int(horizon)) / 24.0
                    quantiles = {
                        "p10": mean_change - uncertainty * 1.5,
                        "p50": mean_change,
                        "p90": mean_change + uncertainty * 1.5,
                    }
                predictions[int(horizon)] = HorizonPrediction(
                    horizon=int(horizon),
                    mean_change_pct=float(mean_change),
                    quantiles=quantiles,
                    direction_probability=None,
                )

            quality = PredictionQuality(
                model_confidence=0.2,
                calibrated=False,
                validation_metrics={},
                notes=[
                    "Heuristic fallback based on MA crossover and RSI",
                    f"Current RSI: {rsi:.1f}",
                    f"Price vs SMA20: {((close / sma_20 - 1) * 100):+.2f}%",
                ],
            )

            return PredictionResponse(
                status="partial" if predictions else "error",
                confidence=quality.model_confidence,
                processing_time_ms=0,
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions=predictions,
                latest_close=close,
                features_used=["sma_20", "sma_50", "rsi_14", "volatility_20"],
                quality=quality,
                model_id="fallback_heuristic",
                warnings=["Using heuristic fallback - ML model unavailable"] if predictions else ["Fallback failed"],
            )
        except Exception as e:
            logger.error("Fallback predictor failed: %s", e)
            return self._empty_response(request, f"Fallback failed: {e}")

    @staticmethod
    def _calculate_rsi(series, period: int = 14) -> float:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

    @staticmethod
    def _get_ma_signal(close: float, sma_20: float, sma_50: float, strength: float) -> float:
        price_vs_sma20 = (close / sma_20) - 1.0
        sma_trend = (sma_20 / sma_50) - 1.0 if sma_50 else 0.0

        if price_vs_sma20 > 0.002 and sma_trend > 0.001:
            return strength * 0.6
        elif price_vs_sma20 < -0.002 and sma_trend < -0.001:
            return -strength * 0.6
        elif price_vs_sma20 > 0:
            return strength * 0.3
        elif price_vs_sma20 < 0:
            return -strength * 0.3
        return 0.0

    @staticmethod
    def _get_rsi_signal(rsi: float, strength: float) -> float:
        if rsi < 30.0:
            intensity = (30.0 - rsi) / 30.0
            return strength * 0.5 * float(intensity)
        elif rsi > 70.0:
            intensity = (rsi - 70.0) / 30.0
            return -strength * 0.5 * float(intensity)
        return 0.0

    @staticmethod
    def _empty_response(request: PredictionRequest, error_msg: str) -> PredictionResponse:
        return PredictionResponse(
            status="error",
            confidence=0.0,
            processing_time_ms=0,
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions={},
            latest_close=0.0,
            features_used=[],
            quality=PredictionQuality(0.0, False, {}, notes=[error_msg]),
            model_id="none",
            warnings=[error_msg],
        )

