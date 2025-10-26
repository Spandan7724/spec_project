import pandas as pd
import numpy as np
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.models import PredictionRequest
from src.prediction.utils.fallback import FallbackPredictor


def _sample_daily_df(n: int = 120):
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 0.4, size=n))
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
async def test_fallback_predict_basic(monkeypatch):
    cfg = PredictionConfig()
    fb = FallbackPredictor(cfg)

    df = _sample_daily_df()

    async def fake_fetch(self, base, quote, days=365, interval="1d"):
        return df

    monkeypatch.setattr(HistoricalDataLoader, "fetch_historical_data", fake_fetch)

    req = PredictionRequest(currency_pair="USD/EUR", horizons=[1, 7, 30])
    resp = await fb.predict(req, "USD", "EUR")
    assert resp.status in {"partial", "success"}
    assert 1 in resp.predictions
    assert isinstance(resp.predictions[1].mean_change_pct, float)


@pytest.mark.asyncio
async def test_fallback_insufficient_data(monkeypatch):
    cfg = PredictionConfig()
    fb = FallbackPredictor(cfg)

    async def fake_fetch(self, base, quote, days=365, interval="1d"):
        return pd.DataFrame({"Close": [100] * 10})

    monkeypatch.setattr(HistoricalDataLoader, "fetch_historical_data", fake_fetch)

    req = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
    resp = await fb.predict(req, "USD", "EUR")
    assert resp.status == "error"
    assert "Insufficient data" in resp.warnings[0]
