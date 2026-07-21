from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.prediction.config import PredictionConfig
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.backends.lstm_backend import LSTMBackend
from src.prediction.backends.catboost_backend import CatBoostBackend
from src.prediction.models import ModelMetadata
from src.prediction.registry import ModelRegistry
from src.prediction.utils.calibration import check_quality_metrics
from src.utils.logging import get_logger


logger = get_logger(__name__)


async def train_and_register_lightgbm(
    currency_pair: str,
    *,
    config: Optional[PredictionConfig] = None,
    days: Optional[int] = None,
    horizons: Optional[List[int]] = None,
    version: str = "1.0",
    # LightGBM intensity knobs
    gbm_rounds: Optional[int] = None,
    gbm_patience: Optional[int] = None,
    gbm_learning_rate: Optional[float] = None,
    gbm_num_leaves: Optional[int] = None,
) -> ModelMetadata:
    """Train a LightGBM model from historical daily data and register it.

    Args:
        currency_pair: e.g., "USD/EUR"
        config: Optional PredictionConfig. If None, loads from YAML.
        days: Optional history depth in days (defaults to config.max_history_days)
        horizons: Optional list of horizons in days (defaults to config.prediction_horizons)
        version: Semantic or incremental version string

    Returns:
        ModelMetadata for the registered model
    """
    cfg = config or PredictionConfig.from_yaml()
    reg = ModelRegistry(cfg.model_registry_path, cfg.model_storage_dir)
    loader = HistoricalDataLoader()
    builder = FeatureBuilder(cfg.technical_indicators)

    # Load historical daily data
    base, quote = currency_pair.split("/")
    df = await loader.fetch_historical_data(
        base, quote, days=days or cfg.max_history_days, interval="1d"
    )
    if df is None or df.empty:
        raise RuntimeError(f"No historical data available for {currency_pair}")

    # Build features and targets
    features = builder.build_features(df, mode=cfg.features_mode)
    hz = horizons or cfg.prediction_horizons
    targets = builder.build_targets(df, hz)
    idx = features.index.intersection(targets.index)
    X, y = features.loc[idx], targets.loc[idx]

    if len(X) < 120:
        raise RuntimeError(f"Insufficient feature rows for training: {len(X)}")

    # Train LightGBM backend
    backend = LightGBMBackend()
    # Build optional parameter overrides
    lgb_params = {}
    if gbm_learning_rate is not None:
        lgb_params["learning_rate"] = gbm_learning_rate
    if gbm_num_leaves is not None:
        lgb_params["num_leaves"] = int(gbm_num_leaves)

    metrics = backend.train(
        X,
        y,
        horizons=hz,
        params=lgb_params or None,
        num_boost_round=gbm_rounds or 120,
        patience=gbm_patience or 10,
    )

    # Serialize backend state using backend.save -> load into memory -> register
    # Ensure storage directory exists
    os.makedirs(cfg.model_storage_dir, exist_ok=True)
    state_tmp_path = os.path.join(
        cfg.model_storage_dir, f"{currency_pair.replace('/', '')}_gbm_state_tmp.pkl"
    )
    backend.save(state_tmp_path)

    import pickle

    with open(state_tmp_path, "rb") as f:
        state_obj = pickle.load(f)
    try:
        os.remove(state_tmp_path)
    except OSError:
        pass

    # Build metadata
    # Basic calibration flag via quality gates
    mq = check_quality_metrics(
        metrics,
        min_samples_required=cfg.min_samples_required,
        min_validation_coverage=cfg.min_validation_coverage,
        min_directional_accuracy=cfg.min_directional_accuracy,
    )
    calibration_ok = mq.get("passed", False)

    model_id = (
        f"{currency_pair.replace('/', '').lower()}_lightgbm_"
        f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )
    meta = ModelMetadata(
        model_id=model_id,
        model_type="lightgbm",
        currency_pair=currency_pair,
        trained_at=datetime.utcnow(),
        version=version,
        validation_metrics=metrics,
        min_samples=min((m.get("n_samples", 0) for m in metrics.values()), default=0),
        calibration_ok=calibration_ok,
        features_used=list(X.columns),
        horizons=hz,
        model_path="",
    )

    # Register with state object
    reg.register_model(meta, state_obj)
    logger.info("Registered LightGBM model %s for %s", meta.model_id, currency_pair)
    return meta


async def train_and_register_lstm(
    currency_pair: str,
    *,
    config: Optional[PredictionConfig] = None,
    days: int = 180,
    interval: str = "1h",
    horizons_hours: Optional[List[int]] = None,
    version: str = "1.0",
    # LSTM intensity knobs
    lstm_epochs: int = 5,
    lstm_hidden_dim: int = 64,
    lstm_seq_len: int = 64,
    lstm_lr: float = 1e-3,
) -> ModelMetadata:
    """Train an LSTM model on intraday data and register it.

    Note: Targets are stored with the same suffix convention as GBM (target_{horizon}d)
    even though horizons here are in hours.
    """
    cfg = config or PredictionConfig.from_yaml()
    reg = ModelRegistry(cfg.model_registry_path, cfg.model_storage_dir)
    loader = HistoricalDataLoader()
    builder = FeatureBuilder(cfg.technical_indicators)

    base, quote = currency_pair.split("/")
    df = await loader.fetch_historical_data(base, quote, days=days, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No intraday data available for {currency_pair}")

    features = builder.build_features(df, mode=cfg.features_mode)
    if len(features) < 128:
        raise RuntimeError("Insufficient intraday rows for LSTM training")

    hz = horizons_hours or [1, 4, 24]
    # Build intraday targets: horizon steps ahead at 'interval' cadence
    targets = pd.DataFrame(index=features.index)
    for h in hz:
        targets[f"target_{h}d"] = (features["Close"].shift(-h) - features["Close"]) / features["Close"] * 100

    idx = features.index.intersection(targets.dropna().index)
    X, y = features.loc[idx], targets.loc[idx]

    backend = LSTMBackend(seq_len=lstm_seq_len, hidden_dim=lstm_hidden_dim)
    backend.train(X, y, horizons=hz, params={"epochs": int(lstm_epochs), "lr": float(lstm_lr)})

    # Save backend state
    os.makedirs(cfg.model_storage_dir, exist_ok=True)
    state_tmp_path = os.path.join(
        cfg.model_storage_dir, f"{currency_pair.replace('/', '')}_lstm_state_tmp.pkl"
    )
    backend.save(state_tmp_path)

    import pickle

    with open(state_tmp_path, "rb") as f:
        state_obj = pickle.load(f)
    try:
        os.remove(state_tmp_path)
    except OSError:
        pass

    model_id = (
        f"{currency_pair.replace('/', '').lower()}_lstm_"
        f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )
    meta = ModelMetadata(
        model_id=model_id,
        model_type="lstm",
        currency_pair=currency_pair,
        trained_at=datetime.utcnow(),
        version=version,
        validation_metrics=backend.validation_metrics,
        min_samples=min((m.get("n_samples", 0) for m in backend.validation_metrics.values()), default=0),
        calibration_ok=True,
        features_used=list(X.columns),
        horizons=hz,
        model_path="",
    )

    reg.register_model(meta, state_obj)
    logger.info("Registered LSTM model %s for %s", meta.model_id, currency_pair)
    return meta


async def train_and_register_catboost(
    currency_pair: str,
    *,
    config: Optional[PredictionConfig] = None,
    days: Optional[int] = None,
    horizons: Optional[List[int]] = None,
    version: str = "1.0",
    # CatBoost intensity knobs
    catboost_rounds: Optional[int] = None,
    catboost_patience: Optional[int] = None,
    catboost_learning_rate: Optional[float] = None,
    catboost_depth: Optional[int] = None,
    task_type: str = "auto",
) -> ModelMetadata:
    """Train a CatBoost model from historical daily data and register it.

    Uses the winning CatBoost_3 configuration with conservative hyperparameters
    for best generalization on forex data.

    Args:
        currency_pair: e.g., "USD/EUR"
        config: Optional PredictionConfig. If None, loads from YAML.
        days: Optional history depth in days (defaults to config.max_history_days)
        horizons: Optional list of horizons in days (defaults to config.prediction_horizons)
        version: Semantic or incremental version string
        catboost_rounds: Maximum number of boosting rounds (default: 4000)
        catboost_patience: Early stopping patience (default: 300)
        catboost_learning_rate: Learning rate (default: 0.005)
        catboost_depth: Tree depth (default: 6)
        task_type: "CPU", "GPU", or "auto" for automatic detection

    Returns:
        ModelMetadata for the registered model
    """
    cfg = config or PredictionConfig.from_yaml()
    reg = ModelRegistry(cfg.model_registry_path, cfg.model_storage_dir)
    loader = HistoricalDataLoader()
    builder = FeatureBuilder(cfg.technical_indicators)

    # Load historical daily data
    base, quote = currency_pair.split("/")
    df = await loader.fetch_historical_data(
        base, quote, days=days or cfg.max_history_days, interval="1d"
    )
    if df is None or df.empty:
        raise RuntimeError(f"No historical data available for {currency_pair}")

    # Build features and targets
    features = builder.build_features(df, mode=cfg.features_mode)
    hz = horizons or cfg.prediction_horizons
    targets = builder.build_targets(df, hz)
    idx = features.index.intersection(targets.index)
    X, y = features.loc[idx], targets.loc[idx]

    if len(X) < 120:
        raise RuntimeError(f"Insufficient feature rows for training: {len(X)}")

    # Train CatBoost backend
    backend = CatBoostBackend(task_type=task_type)

    # Build optional parameter overrides
    catboost_params = {}
    if catboost_learning_rate is not None:
        catboost_params["learning_rate"] = catboost_learning_rate
    if catboost_depth is not None:
        catboost_params["depth"] = int(catboost_depth)
    if catboost_rounds is not None:
        catboost_params["iterations"] = int(catboost_rounds)

    metrics = backend.train(
        X,
        y,
        horizons=hz,
        params=catboost_params or None,
        num_boost_round=catboost_rounds or 4000,
        patience=catboost_patience or 300,
    )

    # Serialize backend state
    os.makedirs(cfg.model_storage_dir, exist_ok=True)
    state_tmp_path = os.path.join(
        cfg.model_storage_dir, f"{currency_pair.replace('/', '')}_catboost_state_tmp.pkl"
    )
    backend.save(state_tmp_path)

    import pickle

    with open(state_tmp_path, "rb") as f:
        state_obj = pickle.load(f)
    try:
        os.remove(state_tmp_path)
    except OSError:
        pass

    # Build metadata
    mq = check_quality_metrics(
        metrics,
        min_samples_required=cfg.min_samples_required,
        min_validation_coverage=cfg.min_validation_coverage,
        min_directional_accuracy=cfg.min_directional_accuracy,
    )
    calibration_ok = mq.get("passed", False)

    model_id = (
        f"{currency_pair.replace('/', '').lower()}_catboost_"
        f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )
    meta = ModelMetadata(
        model_id=model_id,
        model_type="catboost",
        currency_pair=currency_pair,
        trained_at=datetime.utcnow(),
        version=version,
        validation_metrics=metrics,
        min_samples=min((m.get("n_samples", 0) for m in metrics.values()), default=0),
        calibration_ok=calibration_ok,
        features_used=list(X.columns),
        horizons=hz,
        model_path="",
    )

    # Register with state object
    reg.register_model(meta, state_obj)
    logger.info("Registered CatBoost model %s for %s", meta.model_id, currency_pair)
    return meta
