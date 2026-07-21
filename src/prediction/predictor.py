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
        model_used = "none"

        # Try to load CatBoost model first (best performance - 98.8% R²)
        try:
            meta_catboost = self.registry.get_model(request.currency_pair, model_type="catboost")
            if meta_catboost:
                self.backend_catboost.load(meta_catboost["model_path"])

                # Load scaler if available
                scaler_path = meta_catboost.get("scaler_path")
                if scaler_path:
                    import joblib
                    self.backend_catboost.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_path}")

                vm = meta_catboost.get("validation_metrics") or {}
                if isinstance(vm, dict):
                    self.backend_catboost.validation_metrics = vm
                model_used = "catboost"
                logger.info(f"Loaded CatBoost model for {request.currency_pair}")
        except Exception as e:
            logger.debug(f"CatBoost model not available: {e}")

        # If no CatBoost model, try LightGBM
        if model_used == "none":
            try:
                meta = self.registry.get_model(request.currency_pair, model_type="lightgbm")
                if meta:
                    self.backend_gbm.load(meta["model_path"])
                    vm = meta.get("validation_metrics") or {}
                    if isinstance(vm, dict):
                        self.backend_gbm.validation_metrics = vm
                    model_used = "lightgbm"
                    logger.info(f"Loaded LightGBM model for {request.currency_pair}")
            except Exception as e:
                logger.debug(f"LightGBM model not available: {e}")

        # Daily predictions via CatBoost or GBM
        if daily_h:
            X_latest = features_daily.iloc[[-1]]

            # Use CatBoost if available (preferred)
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

            # Fallback to LightGBM if CatBoost not available
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
                    meta_lstm = self.registry.get_model(request.currency_pair, model_type="lstm")
                    if meta_lstm:
                        self.backend_lstm.load(meta_lstm["model_path"])
                        vm = meta_lstm.get("validation_metrics") or {}
                        if isinstance(vm, dict):
                            self.backend_lstm.validation_metrics = vm
                except Exception as e:
                    logger.error(f"Registry load (LSTM) failed: {e}")

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
        if not predictions:
            try:
                fb_resp = await self.fallback.predict(request, base, quote)
                if fb_resp and fb_resp.predictions:
                    # Coerce fallback PredictionResponse into current response
                    predictions = fb_resp.predictions
            except Exception as e:
                logger.error(f"Fallback prediction failed: {e}")

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

            # Model info from registry metadata
            if 'meta_catboost' in locals() and meta_catboost:
                model_info['catboost'] = {
                    k: meta_catboost.get(k)
                    for k in ("model_id", "trained_at", "validation_metrics", "horizons")
                }
            if 'meta' in locals() and meta:
                model_info['lightgbm'] = {
                    k: meta.get(k)
                    for k in ("model_id", "trained_at", "validation_metrics", "horizons")
                }
            if 'meta_lstm' in locals() and meta_lstm:
                model_info['lstm'] = {
                    k: meta_lstm.get(k)
                    for k in ("model_id", "trained_at", "validation_metrics", "horizons")
                }

        # Build response
        # Get model confidence - prioritize CatBoost if available
        all_confidences = []
        all_metrics = {}

        if self.backend_catboost.models:
            all_confidences.append(self.backend_catboost.get_model_confidence())
            all_metrics.update(self.backend_catboost.validation_metrics)
        if self.backend_gbm.models:
            all_confidences.append(self.backend_gbm.get_model_confidence())
            all_metrics.update(self.backend_gbm.validation_metrics)
        if self.backend_lstm.models:
            all_confidences.append(self.backend_lstm.get_model_confidence())
            all_metrics.update(self.backend_lstm.validation_metrics)

        max_confidence = max(all_confidences) if all_confidences else 0.0

        quality = PredictionQuality(
            model_confidence=max_confidence,
            calibrated=False,
            validation_metrics=all_metrics,
            notes=[f"Using {model_used} model"] if model_used != "none" else [],
        )
        response = PredictionResponse(
            status="success" if predictions else "partial",
            confidence=quality.model_confidence,
            processing_time_ms=int((time.time() - start) * 1000),
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions=predictions,
            latest_close=latest_close,
            features_used=self.feature_builder.indicators,
            quality=quality,
            model_id=model_used if model_used != "none" else "hybrid",
            warnings=[] if predictions else ["No models loaded; predictions may be empty"],
            explanations=explanations,
            model_info=model_info or None,
        )
        self.cache[self._get_cache_key(request)] = (response, datetime.now())
        return response
