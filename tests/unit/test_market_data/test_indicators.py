import pandas as pd
import numpy as np
from datetime import datetime

from src.data_collection.market_data.indicators import calculate_indicators


def make_ohlc(days: int = 60, start: float = 1.0, step: float = 0.001) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(), periods=days, freq='D')
    close = np.array([start + i * step for i in range(days)])
    open_ = close * (1 - 0.0005)
    high = close * (1 + 0.001)
    low = close * (1 - 0.001)
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
    }, index=idx)
    return df


def test_indicators_basic():
    df = make_ohlc()
    ind = calculate_indicators(df)

    # Most should exist (since we have 60 days)
    assert ind.sma_20 is not None
    assert ind.sma_50 is not None
    assert ind.ema_12 is not None
    assert ind.ema_26 is not None
    assert ind.rsi_14 is None or (0 <= ind.rsi_14 <= 100)
    assert ind.bb_position is None or (0.0 <= ind.bb_position <= 1.0)


def test_indicators_insufficient_history():
    df = make_ohlc(days=10)
    ind = calculate_indicators(df)
    # Should return None fields due to insufficient history
    assert all(getattr(ind, f) is None for f in ind.__dict__.keys())

