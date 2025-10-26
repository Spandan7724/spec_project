import numpy as np
import pandas as pd

from src.prediction.feature_builder import FeatureBuilder
from src.prediction.backends.lightgbm_backend import LightGBMBackend


def _sample_training_data(n: int = 320):
    rng = np.random.default_rng(42)
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
    y = builder.build_targets(df, horizons=[1, 7])
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]


def test_lightgbm_training_and_prediction(tmp_path):
    X, y = _sample_training_data()
    backend = LightGBMBackend()
    metrics = backend.train(X, y, horizons=[1, 7])

    assert "1d" in metrics and "7d" in metrics
    assert metrics["1d"]["rmse"] > 0
    assert 0 <= metrics["1d"]["directional_accuracy"] <= 1

    X_latest = X.iloc[[-1]]
    preds = backend.predict(X_latest, horizons=[1, 7], include_quantiles=True)
    assert 1 in preds and 7 in preds
    assert "mean_change" in preds[1]
    assert "quantiles" in preds[1]
    assert "direction_prob" in preds[1]

    # feature importance
    fi = backend.get_feature_importance(1, top_n=3)
    assert len(fi) <= 3

    # save and load
    model_path = tmp_path / "gbm.pkl"
    backend.save(str(model_path))
    assert model_path.exists()

    backend2 = LightGBMBackend()
    backend2.load(str(model_path))
    preds2 = backend2.predict(X_latest, horizons=[1])
    assert abs(preds2[1]["mean_change"] - preds[1]["mean_change"]) < 1e-3

