import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.training import train_and_register_lstm
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest


def _synthetic_hourly_df(n: int = 240):
    rng = np.random.default_rng(101)
    dates = pd.date_range("2024-01-01", periods=n, freq="H")
    close = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.05, size=n),
            "High": close + np.abs(rng.normal(0, 0.1, size=n)),
            "Low": close - np.abs(rng.normal(0, 0.1, size=n)),
            "Close": close,
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_predictor_loads_lstm_from_registry(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PredictionConfig(
            model_registry_path=os.path.join(tmpdir, "registry.json"),
            model_storage_dir=os.path.join(tmpdir, "models"),
        )

        df_1h = _synthetic_hourly_df()
        builder = FeatureBuilder(cfg.technical_indicators)

        async def fake_fetch(self, base, quote, days=180, interval="1h"):
            return df_1h

        # Train and register LSTM on synthetic 1h data
        from src.prediction import training as training_mod

        monkeypatch.setattr(
            training_mod.HistoricalDataLoader, "fetch_historical_data", fake_fetch
        )
        await train_and_register_lstm(
            "USD/EUR", config=cfg, days=120, interval="1h", horizons_hours=[1]
        )

        predictor = MLPredictor(cfg)

        # Monkeypatch predictor loader:
        async def fake_fetch_daily(base, quote, days=365, interval="1d"):
            # derive a simple daily df from hourly by resampling
            dfd = df_1h.resample("D").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna()
            return dfd

        async def fake_fetch_intraday(base, quote, days=180, interval="1h"):
            return df_1h

        monkeypatch.setattr(
            predictor.data_loader, "fetch_historical_data", fake_fetch_daily
        )
        # For intraday path, construct a side object to call the loader directly
        # We'll replace it at runtime before the intraday call
        async def fetch_selector(base, quote, days, interval="1d"):
            if interval == "1h":
                return await fake_fetch_intraday(base, quote, days, interval)
            return await fake_fetch_daily(base, quote, days, interval)

        monkeypatch.setattr(
            predictor.data_loader, "fetch_historical_data", fetch_selector
        )

        req = PredictionRequest(
            currency_pair="USD/EUR", horizons=[], intraday_horizons_hours=[1]
        )
        result = await predictor.predict(req)
        assert result.predictions and 1 in result.predictions
        assert isinstance(result.predictions[1].mean_change_pct, float)
