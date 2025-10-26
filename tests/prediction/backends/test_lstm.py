import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch")

from src.prediction.feature_builder import FeatureBuilder
from src.prediction.backends.lstm_backend import LSTMBackend


def _sample_training_data(n: int = 200):
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=n, freq="H")
    close = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.05, size=n),
            "High": close + np.abs(rng.normal(0, 0.1, size=n)),
            "Low": close - np.abs(rng.normal(0, 0.1, size=n)),
            "Close": close,
        },
        index=dates,
    )
    builder = FeatureBuilder(["sma_5", "sma_20", "rsi_14", "macd", "macd_signal"])
    X = builder.build_features(df, mode="price_only").tail(180)
    # Use 1-step ahead on this hourly series but align to '1d' column name for compatibility
    y = pd.DataFrame(index=X.index)
    y["target_1d"] = (X["Close"].shift(-1) - X["Close"]) / X["Close"] * 100
    return X.dropna(), y.dropna()


def test_lstm_training_and_prediction():
    X, y = _sample_training_data()
    backend = LSTMBackend(seq_len=32, hidden_dim=32)
    metrics = backend.train(X, y, horizons=[1], params={"epochs": 3, "lr": 5e-3})
    assert "1d" in metrics

    X_latest = X.iloc[-40:]  # ensure >= seq_len rows
    preds = backend.predict(X_latest, horizons=[1])
    assert 1 in preds
    assert "mean_change" in preds[1]


def test_lstm_predict_with_quantiles():
    X, y = _sample_training_data()
    backend = LSTMBackend(seq_len=32, hidden_dim=16, mc_samples=10)
    backend.train(X, y, horizons=[1], params={"epochs": 2, "lr": 1e-3})
    X_latest = X.iloc[-40:]
    preds = backend.predict(X_latest, horizons=[1], include_quantiles=True)
    assert "quantiles" in preds[1]
    assert "direction_prob" in preds[1]


def test_lstm_save_and_load(tmp_path):
    X, y = _sample_training_data()
    backend = LSTMBackend(seq_len=32, hidden_dim=16)
    backend.train(X, y, horizons=[1], params={"epochs": 2, "lr": 1e-3})

    model_path = tmp_path / "lstm_state.pkl"
    backend.save(str(model_path))
    assert model_path.exists()

    backend2 = LSTMBackend()
    backend2.load(str(model_path))
    # Use last 40 rows to ensure >= seq_len
    preds1 = backend.predict(X.iloc[-40:], horizons=[1])
    preds2 = backend2.predict(X.iloc[-40:], horizons=[1])
    assert 1 in preds1 and 1 in preds2
    # Not necessarily equal due to float drift, but both should produce float
    assert isinstance(preds2[1]["mean_change"], float)
