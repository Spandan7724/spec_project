"""
Advanced ML Predictor using pre-trained models from ml_models directory.
This predictor uses the advanced ensemble of CatBoost, XGBoost, LightGBM, and Neural Networks.
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.prediction.config import PredictionConfig
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.ml_models_feature_builder import MLModelsFeatureBuilder
from src.prediction.models import (
    HorizonPrediction,
    PredictionQuality,
    PredictionRequest,
    PredictionResponse,
)
from src.prediction.backends.advanced_ensemble_backend import AdvancedEnsembleBackend
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedMLPredictor:
    """
    Advanced prediction service using pre-trained ensemble models from ml_models.
    This provides higher accuracy predictions using CatBoost, XGBoost, LightGBM,
    and Neural Network ensembles.
    """

    def __init__(self, config: Optional[PredictionConfig] = None, ml_models_dir: str = "ml_models/models"):
        self.config = config or PredictionConfig.from_yaml()
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = MLModelsFeatureBuilder()

        # Load advanced ensemble backend
        try:
            self.backend = AdvancedEnsembleBackend(ml_models_dir=ml_models_dir)
            self.available = True
            logger.info("✓ Advanced ensemble backend loaded successfully")
            model_info = self.backend.get_model_info()
            logger.info(f"✓ Best model: {model_info['best_model']}")
            logger.info(f"✓ Test R²: {model_info['test_r2']:.6f}")
            logger.info(f"✓ MAPE: {model_info['mape']:.4f}%")
        except Exception as e:
            logger.error(f"Failed to load advanced ensemble backend: {e}")
            self.backend = None
            self.available = False

        self.cache: Dict[str, Tuple[PredictionResponse, datetime]] = {}

    def is_available(self) -> bool:
        """Check if advanced predictor is available"""
        return self.available and self.backend is not None

    def _get_cache_key(self, request: PredictionRequest) -> str:
        key_data = {
            "pair": request.currency_pair,
            "horizons": sorted(request.horizons),
            "mode": "advanced_ensemble",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make predictions using advanced ensemble models.

        Args:
            request: Prediction request with currency pair and horizons

        Returns:
            PredictionResponse with ensemble predictions
        """
        start = time.time()
        correlation_id = request.correlation_id or "unknown"

        if not self.is_available():
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000),
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(model_confidence=0.0, calibrated=False, validation_metrics={}, notes=["Advanced predictor not available"]),
                model_id="none",
                warnings=["Advanced ensemble backend not loaded"],
            )

        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            cached_resp, ts = self.cache[cache_key]
            age_h = (datetime.now() - ts).total_seconds() / 3600
            if age_h <= request.max_age_hours:
                cached_resp.cached = True
                cached_resp.processing_time_ms = int((time.time() - start) * 1000)
                logger.info(f"Returning cached prediction (age: {age_h:.2f}h)")
                return cached_resp

        # Parse currency pair
        try:
            base, quote = request.currency_pair.split("/")
        except ValueError:
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000),
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(model_confidence=0.0, calibrated=False, validation_metrics={}, notes=["Invalid currency pair format"]),
                model_id="none",
                warnings=["Invalid currency pair. Use format BASE/QUOTE (e.g., USD/EUR)"],
            )

        # Load historical data
        logger.info(f"Loading historical data for {request.currency_pair}")
        df_daily = await self.data_loader.fetch_historical_data(
            base, quote,
            days=max(self.config.max_history_days, 300),  # Need more history for advanced features
            interval="1d"
        )

        if df_daily is None or df_daily.empty:
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000),
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(model_confidence=0.0, calibrated=False, validation_metrics={}, notes=["No historical data available"]),
                model_id="none",
                warnings=["Unable to fetch historical data"],
            )

        logger.info(f"Loaded {len(df_daily)} days of historical data")

        # Build features using ml_models feature engineering
        try:
            logger.info("Building advanced features...")
            features = self.feature_builder.prepare_for_prediction(df_daily)
            logger.info(f"Built {features.shape[1]} features for prediction")
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000),
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(model_confidence=0.0, calibrated=False, validation_metrics={}, notes=[f"Feature engineering error: {str(e)}"]),
                model_id="none",
                warnings=["Feature computation failed"],
            )

        latest_close = float(df_daily["Close"].iloc[-1])

        # Make predictions using advanced ensemble
        try:
            logger.info(f"Making predictions for horizons: {request.horizons}")
            raw_predictions = self.backend.predict(
                features,
                horizons=request.horizons,
                include_quantiles=True
            )

            # Convert to HorizonPrediction objects
            predictions: Dict[int, HorizonPrediction] = {}
            for horizon, pred_dict in raw_predictions.items():
                predictions[horizon] = HorizonPrediction(
                    horizon=horizon,
                    mean_change_pct=pred_dict["mean_change"],
                    quantiles=pred_dict.get("quantiles"),
                    direction_probability=pred_dict.get("direction_prob", 0.5),
                )

            logger.info(f"Generated {len(predictions)} predictions")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000),
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=latest_close,
                features_used=list(features.columns),
                quality=PredictionQuality(model_confidence=0.0, calibrated=False, validation_metrics={}, notes=[f"Prediction error: {str(e)}"]),
                model_id="advanced_ensemble",
                warnings=["Prediction computation failed"],
            )

        # Calculate quality metrics
        confidence = self.backend.get_model_confidence()
        model_info = self.backend.get_model_info()

        quality = PredictionQuality(
            model_confidence=confidence,
            calibrated=True,
            validation_metrics={
                "test_rmse": model_info.get("test_rmse", 0.0),
                "test_mae": model_info.get("test_mae", 0.0),
                "test_r2": model_info.get("test_r2", 0.0),
                "mape": model_info.get("mape", 0.0),
                "direction_accuracy": model_info.get("direction_accuracy", 0.0),
            },
            notes=[
                f"Best model: {model_info['best_model']}",
                f"Ensemble with {len(model_info['loaded_models'])} model types"
            ]
        )

        # Build response
        response = PredictionResponse(
            status="success",
            confidence=confidence,
            processing_time_ms=int((time.time() - start) * 1000),
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions=predictions,
            latest_close=latest_close,
            features_used=list(features.columns),
            quality=quality,
            model_id="advanced_ensemble",
            warnings=[],
            cached=False,
            model_info={
                "backend": "advanced_ensemble",
                "best_model": model_info["best_model"],
                "test_metrics": {
                    "rmse": model_info.get("test_rmse"),
                    "mae": model_info.get("test_mae"),
                    "r2": model_info.get("test_r2"),
                    "mape": model_info.get("mape"),
                },
                "models_loaded": model_info.get("loaded_models"),
            }
        )

        # Cache the response
        self.cache[cache_key] = (response, datetime.now())

        logger.info(f"Prediction completed in {response.processing_time_ms}ms with confidence {confidence:.2%}")

        return response

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        if not self.is_available():
            return {"available": False}

        info = self.backend.get_model_info()
        info["available"] = True
        return info
