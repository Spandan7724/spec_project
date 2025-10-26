import pandas as pd
import numpy as np

from src.prediction.feature_builder import FeatureBuilder


def _sample_ohlc(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, size=n),
            "High": close + np.abs(rng.normal(0, 0.2, size=n)),
            "Low": close - np.abs(rng.normal(0, 0.2, size=n)),
            "Close": close,
        },
        index=dates,
    )
    return df


def test_build_features_price_only():
    df = _sample_ohlc(120)
    indicators = [
        "sma_5",
        "sma_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr_14",
        "volatility_20",
    ]
    builder = FeatureBuilder(indicators)
    features = builder.build_features(df, mode="price_only")

    assert len(features) > 0
    for col in [
        "sma_5",
        "sma_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr_14",
        "volatility_20",
    ]:
        assert col in features.columns


def test_calculate_rsi_bounds():
    df = _sample_ohlc(60)
    rsi = FeatureBuilder._calculate_rsi(df["Close"], period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_build_targets_columns():
    df = _sample_ohlc(120)
    builder = FeatureBuilder(["sma_5"])  # indicators list not used by build_targets
    horizons = [1, 7, 30]
    targets = builder.build_targets(df, horizons)
    for h in horizons:
        assert f"target_{h}d" in targets.columns
        assert f"direction_{h}d" in targets.columns

