import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple


from src.prediction.config import PredictionConfig
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.models import (
    HorizonPrediction,
    PredictionQuality,
    PredictionRequest,
    PredictionResponse,
)
from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.backends.lstm_backend import LSTMBackend
from src.prediction.backends.catboost_backend import CatBoostBackend
from src.prediction.registry import ModelRegistry
from src.prediction.utils.fallback import FallbackPredictor
from src.prediction.utils.confidence import confidence_by_horizon
from src.prediction.explainer import PredictionExplainer
from src.utils.logging import get_logger


logger = get_logger(__name__)


class MLPredictor:
    """Main prediction service with simple caching and hybrid routing."""

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig.from_yaml()
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(self.config.technical_indicators)
        self.backend_gbm = LightGBMBackend()
        self.backend_lstm = LSTMBackend()
        self.backend_catboost = CatBoostBackend()
        self.registry = ModelRegistry(
            self.config.model_registry_path, self.config.model_storage_dir
        )
        self.fallback = FallbackPredictor(self.config)
        self.cache: Dict[str, Tuple[PredictionResponse, datetime]] = {}

    def _get_cache_key(self, request: PredictionRequest) -> str:
        key_data = {
            "pair": request.currency_pair,
            "horizons": sorted(request.horizons),
            "intraday": sorted(request.intraday_horizons_hours or []),
            "mode": request.features_mode,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _split_horizons(self, request: PredictionRequest) -> Tuple[list, list]:
        """Return (daily_horizons, intraday_horizons_hours)."""
        daily = request.horizons or []
        intraday = request.intraday_horizons_hours or []
        pref = (request.backend_preference or self.config.predictor_backend or "hybrid").lower()
        if pref == "lightgbm":
            # Ignore intraday horizons
            return daily, []
        if pref == "lstm":
            # Convert daily horizons to hours so LSTM can serve them (day -> 24h)
            intraday_ext = intraday + [int(d) * 24 for d in daily]
            return [], intraday_ext
        # hybrid: use both
        return daily, intraday

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        start = time.time()
        correlation_id = request.correlation_id or "unknown"

        # Cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            cached_resp, ts = self.cache[cache_key]
            age_h = (datetime.now() - ts).total_seconds() / 3600
            if age_h <= request.max_age_hours:
                cached_resp.cached = True
                cached_resp.processing_time_ms = int((time.time() - start) * 1000)
                return cached_resp

        # Parse pair
        try:
            base, quote = request.currency_pair.split("/")
        except ValueError:
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=0,
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(0.0, False, {}, notes=["Invalid pair"]),
                model_id="none",
                warnings=["Invalid currency pair"],
            )

        # Load daily data
        df_daily = await self.data_loader.fetch_historical_data(base, quote, days=self.config.max_history_days, interval="1d")
        if df_daily is None or df_daily.empty:
            return PredictionResponse(
                status="error",
                confidence=0.0,
                processing_time_ms=0,
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={},
                latest_close=0.0,
                features_used=[],
                quality=PredictionQuality(0.0, False, {}, notes=["No historical data"]),
                model_id="none",
                warnings=["Data unavailable"],
            )

        # Features for daily
        features_daily = self.feature_builder.build_features(df_daily, mode=request.features_mode)
        latest_close = float(df_daily["Close"].iloc[-1])

        daily_h, intraday_h = self._split_horizons(request)

        predictions: Dict[int, HorizonPrediction] = {}
        selected_daily_meta = None
        selected_lstm_meta = None
        loaded_daily_meta = None
        loaded_lstm_meta = None
        model_type_used = "none"
        model_id_used = "none"
        runtime_warnings: list[str] = []

        # Match the Models page: select the newest compatible daily model for
        # the pair. Hybrid compares LightGBM and CatBoost by the same ordering;
        # an explicit backend preference restricts the compatible model types.
        if daily_h:
            backend_preference = (
                request.backend_preference or self.config.predictor_backend or "hybrid"
            ).lower()
            if backend_preference == "lightgbm":
                daily_model_types = ["lightgbm"]
            elif backend_preference == "catboost":
                daily_model_types = ["catboost"]
            else:
                daily_model_types = ["lightgbm", "catboost"]
            selected_daily_meta = self.registry.get_latest_model(
                request.currency_pair,
                model_types=daily_model_types,
            )
            if selected_daily_meta:
                selected_type = selected_daily_meta.get("model_type")
                selected_id = selected_daily_meta.get("model_id", selected_type or "unknown")
                try:
                    model_path = self.registry.resolve_artifact_path(
                        selected_daily_meta["model_path"]
                    )
                    if selected_type == "catboost":
                        self.backend_catboost.load(model_path)
                        scaler_path = selected_daily_meta.get("scaler_path")
                        if scaler_path:
                            import joblib

                            resolved_scaler = self.registry.resolve_artifact_path(scaler_path)
                            self.backend_catboost.scaler = joblib.load(resolved_scaler)
                        self.backend_catboost.validation_metrics = (
                            selected_daily_meta.get("validation_metrics") or {}
                        )
                    elif selected_type == "lightgbm":
                        self.backend_gbm.load(model_path)
                        self.backend_gbm.validation_metrics = (
                            selected_daily_meta.get("validation_metrics") or {}
                        )
                    else:
                        raise ValueError(f"Unsupported daily model type: {selected_type}")

                    model_type_used = str(selected_type)
                    model_id_used = str(selected_id)
                    loaded_daily_meta = selected_daily_meta
                    logger.info(
                        "Loaded newest %s model %s for %s",
                        selected_type,
                        selected_id,
                        request.currency_pair,
                    )
                except Exception as exc:
                    message = f"Newest model {selected_id} could not be loaded: {exc}"
                    runtime_warnings.append(message)
                    logger.warning(message)

        # Daily predictions via CatBoost or GBM
        if daily_h:
            X_latest = features_daily.iloc[[-1]]

            # Use the single newest model selected above.
            if self.backend_catboost.models:
                raw = self.backend_catboost.predict(X_latest, horizons=daily_h)
                # Convert to expected format
                for h in daily_h:
                    if f"pred_{h}d" in raw.columns:
                        predictions[h] = HorizonPrediction(
                            horizon=h,
                            mean_change_pct=float(raw[f"pred_{h}d"].iloc[0]),
                            quantiles={
                                "p10": float(raw[f"pred_{h}d_p10"].iloc[0]) if f"pred_{h}d_p10" in raw.columns else None,
                                "p50": float(raw[f"pred_{h}d_p50"].iloc[0]) if f"pred_{h}d_p50" in raw.columns else None,
                                "p90": float(raw[f"pred_{h}d_p90"].iloc[0]) if f"pred_{h}d_p90" in raw.columns else None,
                            },
                            direction_probability=float(raw[f"direction_{h}d_proba"].iloc[0]) if f"direction_{h}d_proba" in raw.columns else 0.5,
                        )

            elif self.backend_gbm.models:
                raw = self.backend_gbm.predict(X_latest, horizons=daily_h, include_quantiles=True)
                for h, pdict in raw.items():
                    predictions[h] = HorizonPrediction(
                        horizon=h,
                        mean_change_pct=pdict["mean_change"],
                        quantiles=pdict.get("quantiles"),
                        direction_probability=pdict.get("direction_prob"),
                    )

        # Intraday via LSTM (if models available and intraday horizons provided)
        if intraday_h:
            # Load 1h data and compute features (reuse same builder)
            df_1h = await self.data_loader.fetch_historical_data(base, quote, days=min(self.config.max_history_days, 180), interval="1h")
            if df_1h is not None and len(df_1h) >= 128:
                feats_1h = self.feature_builder.build_features(df_1h, mode=request.features_mode)
                # Load LSTM model for pair from registry if available
                try:
                    selected_lstm_meta = self.registry.get_latest_model(
                        request.currency_pair, model_types=["lstm"]
                    )
                    if selected_lstm_meta:
                        lstm_path = self.registry.resolve_artifact_path(
                            selected_lstm_meta["model_path"]
                        )
                        self.backend_lstm.load(lstm_path)
                        vm = selected_lstm_meta.get("validation_metrics") or {}
                        if isinstance(vm, dict):
                            self.backend_lstm.validation_metrics = vm
                        loaded_lstm_meta = selected_lstm_meta
                        if model_id_used == "none":
                            model_type_used = "lstm"
                            model_id_used = str(selected_lstm_meta.get("model_id", "lstm"))
                except Exception as e:
                    logger.error(f"Registry load (LSTM) failed: {e}")
                    runtime_warnings.append(f"Newest LSTM model could not be loaded: {e}")

                if self.backend_lstm.models:
                    # Map hours to a faux day-key for response horizon (e.g., 1h -> horizon 1)
                    raw_lstm = self.backend_lstm.predict(feats_1h, horizons=intraday_h)
                    for h, pdict in raw_lstm.items():
                        predictions[h] = HorizonPrediction(
                            horizon=h,
                            mean_change_pct=pdict["mean_change"],
                            quantiles=pdict.get("quantiles"),
                            direction_probability=pdict.get("direction_prob"),
                        )

        # If no predictions from ML backends, attempt fallback heuristics
        fallback_response = None
        if not predictions:
            try:
                fb_resp = await self.fallback.predict(request, base, quote)
                if fb_resp and fb_resp.predictions:
                    fallback_response = fb_resp
                    predictions = fb_resp.predictions
                    model_type_used = "fallback"
                    model_id_used = fb_resp.model_id
            except Exception as e:
                logger.error(f"Fallback prediction failed: {e}")
                runtime_warnings.append(f"Fallback prediction failed: {e}")

        # Optional explanations/evidence
        explanations = None
        model_info = {}
        need_expl = bool(request.include_explanations or getattr(self.config, "explain_enabled", False))
        if need_expl:
            explanations = {"daily": {}, "intraday": {}}
            # GBM explanations: top features and optional SHAP plot
            if self.backend_gbm.models and daily_h:
                top_n = getattr(self.config, "explain_top_n", 5)
                include_plots = bool(getattr(self.config, "explain_include_plots", False))
                try:
                    X_latest = features_daily.iloc[[-1]]
                    for h in daily_h:
                        if h in predictions and h in self.backend_gbm.models:
                            tf = self.backend_gbm.get_feature_importance(h, top_n=top_n)
                            item = {"top_features": tf}
                            try:
                                expl = PredictionExplainer(self.backend_gbm.models[h], self.backend_gbm.feature_names)
                                waterfall_payload = expl.generate_waterfall(X_latest, include_plot=include_plots)
                                if waterfall_payload.get("plot_base64"):
                                    item["shap_waterfall_base64"] = waterfall_payload["plot_base64"]
                                if waterfall_payload.get("data"):
                                    item["shap_waterfall_data"] = waterfall_payload["data"]
                            except Exception as exc:  # noqa: BLE001
                                logger.error(f"Waterfall generation failed for horizon {h}: {exc}")
                            explanations["daily"][str(h)] = item
                except Exception:
                    pass
            # LSTM explanations: MC samples info
            if self.backend_lstm.models and intraday_h:
                for h in intraday_h:
                    if h in predictions:
                        explanations["intraday"][str(h)] = {"mc_samples": getattr(self.backend_lstm, "mc_samples", None)}

        # Build validation metadata from the model(s) that actually produced
        # predictions. Confidence is calculated per horizon, not from R².
        all_metrics = {}
        if self.backend_catboost.models:
            all_metrics.update(self.backend_catboost.validation_metrics)
        if self.backend_gbm.models:
            all_metrics.update(self.backend_gbm.validation_metrics)
        if self.backend_lstm.models:
            all_metrics.update(self.backend_lstm.validation_metrics)

        reliability_by_horizon = confidence_by_horizon(
            all_metrics,
            predictions.keys(),
            min_samples_required=self.config.min_samples_required,
        )
        if fallback_response is not None:
            reliability_by_horizon = {
                str(horizon): fallback_response.confidence for horizon in predictions
            }

        primary_horizon = next(
            (horizon for horizon in (daily_h + intraday_h) if horizon in predictions),
            next(iter(predictions), None),
        )
        model_confidence = (
            reliability_by_horizon.get(str(primary_horizon), 0.0)
            if primary_horizon is not None
            else 0.0
        )

        if loaded_daily_meta:
            daily_type = str(loaded_daily_meta.get("model_type", "daily"))
            model_info[daily_type] = {
                key: loaded_daily_meta.get(key)
                for key in (
                    "model_id",
                    "model_type",
                    "trained_at",
                    "validation_metrics",
                    "horizons",
                    "calibration_ok",
                )
            }
        if loaded_lstm_meta:
            model_info["lstm"] = {
                key: loaded_lstm_meta.get(key)
                for key in (
                    "model_id",
                    "model_type",
                    "trained_at",
                    "validation_metrics",
                    "horizons",
                    "calibration_ok",
                )
            }
        model_info["confidence_by_horizon"] = reliability_by_horizon
        model_info["confidence_method"] = (
            "fixed_low_confidence_fallback"
            if fallback_response is not None
            else "direction_accuracy_adjusted_for_interval_coverage_and_sample_size"
        )

        calibrated_models = [
            meta
            for meta in (loaded_daily_meta, loaded_lstm_meta)
            if meta is not None
        ]
        calibrated = (
            fallback_response is None
            and bool(calibrated_models)
            and all(bool(meta.get("calibration_ok")) for meta in calibrated_models)
        )

        quality = PredictionQuality(
            model_confidence=model_confidence,
            calibrated=calibrated,
            validation_metrics=all_metrics,
            notes=[f"Using newest {model_type_used} model"] if model_id_used != "none" else [],
        )
        response = PredictionResponse(
            status=(
                fallback_response.status
                if fallback_response is not None
                else ("success" if predictions else "partial")
            ),
            confidence=quality.model_confidence,
            processing_time_ms=int((time.time() - start) * 1000),
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions=predictions,
            latest_close=latest_close,
            features_used=self.feature_builder.indicators,
            quality=quality,
            model_id=model_id_used if model_id_used != "none" else "none",
            warnings=(
                runtime_warnings
                + (fallback_response.warnings if fallback_response is not None else [])
                + ([] if predictions else ["No models loaded; predictions may be empty"])
            ),
            explanations=explanations,
            model_info=model_info or None,
        )
        self.cache[self._get_cache_key(request)] = (response, datetime.now())
        return response
