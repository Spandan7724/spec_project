import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.training import train_and_register_lightgbm
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest


def _synthetic_daily_df(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, size=n),
            "High": close + np.abs(rng.normal(0, 0.2, size=n)),
            "Low": close - np.abs(rng.normal(0, 0.2, size=n)),
            "Close": close,
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_train_and_register_lightgbm(monkeypatch):
    # Create temp registry storage
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PredictionConfig(
            model_registry_path=os.path.join(tmpdir, "registry.json"),
            model_storage_dir=os.path.join(tmpdir, "models"),
        )

        # Mock data loader to avoid network
        df = _synthetic_daily_df()

        async def fake_fetch(self, base, quote, days=365, interval="1d"):
            return df

        from src.prediction import training as training_mod

        monkeypatch.setattr(
            training_mod.HistoricalDataLoader, "fetch_historical_data", fake_fetch
        )

        # Train and register
        meta = await train_and_register_lightgbm("USD/EUR", config=cfg, horizons=[1])
        assert meta.model_id

        # Now predictor should load model from registry and predict
        predictor = MLPredictor(cfg)

        async def fake_fetch_pred(base, quote, days=365, interval="1d"):
            return df

        monkeypatch.setattr(
            predictor.data_loader, "fetch_historical_data", fake_fetch_pred
        )

        req = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
        result = await predictor.predict(req)
        assert result.predictions and 1 in result.predictions
