import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.config import PredictionConfig
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.models import ModelMetadata, PredictionRequest
from src.prediction.predictor import MLPredictor
from src.prediction.registry import ModelRegistry


def _sample_daily_training(n: int = 320):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 0.6, size=n))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, size=n),
            "High": close + np.abs(rng.normal(0, 0.2, size=n)),
            "Low": close - np.abs(rng.normal(0, 0.2, size=n)),
            "Close": close,
        },
        index=dates,
    )
    builder = FeatureBuilder(["sma_5", "sma_20", "rsi_14", "macd", "macd_signal"])
    X = builder.build_features(df, mode="price_only")
    y = builder.build_targets(df, horizons=[1])
    idx = X.index.intersection(y.index)
    return df.loc[idx], X.loc[idx], y.loc[idx]


@pytest.mark.asyncio
async def test_predictor_loads_from_registry(monkeypatch):
    # Prepare trained GBM state
    df, X, y = _sample_daily_training()
    gbm = LightGBMBackend()
    gbm.train(X, y, horizons=[1])
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "state.pkl")
        gbm.save(state_path)
        with open(state_path, "rb") as f:
            state_obj = pickle.load(f)

        # Setup registry
        registry_path = os.path.join(tmpdir, "registry.json")
        storage_dir = os.path.join(tmpdir, "models")
        reg = ModelRegistry(registry_path, storage_dir)

        meta = ModelMetadata(
            model_id="usdeur_lightgbm_test",
            model_type="lightgbm",
            currency_pair="USD/EUR",
            trained_at=pd.Timestamp.utcnow().to_pydatetime(),
            version="1.0",
            validation_metrics=gbm.validation_metrics,
            min_samples=100,
            calibration_ok=True,
            features_used=list(X.columns),
            horizons=[1],
            model_path="",
        )
        reg.register_model(meta, state_obj)

        # Predictor with registry-config
        cfg = PredictionConfig(
            model_registry_path=registry_path,
            model_storage_dir=storage_dir,
        )
        predictor = MLPredictor(cfg)

        # Patch data loader and features to use the same data
        async def fake_fetch(base, quote, days=365, interval="1d"):
            return df

        monkeypatch.setattr(predictor.data_loader, "fetch_historical_data", fake_fetch)
        monkeypatch.setattr(predictor.feature_builder, "build_features", lambda df_, mode="price_only": X)

        req = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
        result = await predictor.predict(req)

        assert result.predictions
        assert 1 in result.predictions
        assert isinstance(result.predictions[1].mean_change_pct, float)

