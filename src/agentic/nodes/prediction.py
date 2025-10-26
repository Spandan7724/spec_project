"""Price Prediction LangGraph node implementation."""
from __future__ import annotations

import time
from typing import Any, Dict

from src.agentic.state import AgentState
from src.prediction.config import PredictionConfig
from src.prediction.models import PredictionRequest
from src.prediction.predictor import MLPredictor
from src.utils.decorators import timeout, log_execution
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _timeframe_to_horizons(timeframe: str, cfg: PredictionConfig) -> tuple[list[int], list[int]]:
    """Map timeframe to (daily_horizons, intraday_horizons_hours) using config preferences."""
    tf = (timeframe or "1_day").lower()
    daily = list(cfg.daily_horizons or [1, 7, 30])
    intraday = list(cfg.intraday_horizons or [1, 4, 24])
    if tf in {"immediate"}:
        return daily, intraday
    if tf in {"1_day"}:
        return daily, (intraday if cfg.include_intraday_for_1_day else [])
    if tf in {"1_week"}:
        # Keep daily horizons >= 7
        return [h for h in daily if h >= 7], []
    if tf in {"1_month"}:
        # Keep daily horizons >= 30
        return [h for h in daily if h >= 30], []
    return daily, []


@timeout(15.0)
@log_execution(log_args=False, log_result=False)
async def prediction_node(state: AgentState) -> Dict[str, Any]:
    """Generate ML-based price forecasts and update state fields.

    Returns a partial state update with keys:
    - prediction_status: "success" | "partial" | "error"
    - prediction_error: optional error message
    - price_forecast: dict payload when available
    """
    start = time.time()
    base = state.get("base_currency") or (state.get("currency_pair", "").split("/")[0] if state.get("currency_pair") else "USD")
    quote = state.get("quote_currency") or (state.get("currency_pair", "").split("/")[-1] if state.get("currency_pair") else "EUR")
    currency_pair = f"{base}/{quote}"

    try:
        # Build prediction request
        cfg = PredictionConfig.from_yaml()
        user_tf = state.get("timeframe")
        if cfg.use_user_timeframe:
            # Strictly require timeframe from user when enabled
            if not user_tf:
                logger.error("Prediction node: missing timeframe from user while use_user_timeframe=true")
                return {
                    "prediction_status": "error",
                    "prediction_error": "missing_timeframe",
                    "price_forecast": None,
                }
            timeframe = user_tf
        else:
            timeframe = cfg.default_timeframe or "1_day"
        daily_h, intraday_h = _timeframe_to_horizons(timeframe, cfg)
        req = PredictionRequest(
            currency_pair=currency_pair,
            horizons=daily_h or cfg.prediction_horizons,
            include_quantiles=True,
            include_direction_probabilities=True,
            max_age_hours=1,
            features_mode=cfg.features_mode,
            correlation_id=state.get("correlation_id", "unknown"),
            intraday_horizons_hours=intraday_h,
            include_explanations=cfg.explain_enabled,
        )

        predictor = MLPredictor(cfg)
        pred_response = await predictor.predict(req)

        # Format predictions for state
        predictions_view = {}
        for horizon, pred in pred_response.predictions.items():
            predictions_view[str(horizon)] = {
                "mean_change_pct": pred.mean_change_pct,
                "quantiles": pred.quantiles,
                "direction_prob": pred.direction_probability,
            }

        exec_ms = int((time.time() - start) * 1000)
        price_forecast = {
            "status": pred_response.status,
            "confidence": pred_response.confidence,
            "predictions": predictions_view,
            "latest_close": pred_response.latest_close,
            "model_id": pred_response.model_id,
            "cached": pred_response.cached,
            "processing_time_ms": exec_ms,
            "features_used": pred_response.features_used,
            "warnings": pred_response.warnings,
            "explanations": pred_response.explanations,
            "model_info": pred_response.model_info,
        }

        return {
            "prediction_status": pred_response.status,
            "prediction_error": None,
            "price_forecast": price_forecast,
        }

    except Exception as e:
        logger.error(f"Prediction node failed: {e}")
        return {
            "prediction_status": "error",
            "prediction_error": str(e),
            "price_forecast": None,
        }
