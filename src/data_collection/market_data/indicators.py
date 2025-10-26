"""Technical indicators for market data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Indicators:
    sma_20: Optional[float]
    sma_50: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    rsi_14: Optional[float]  # [0,100]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    bb_middle: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_position: Optional[float]  # [0,1]
    atr_14: Optional[float]
    realized_vol_30d: Optional[float]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(series: pd.Series, period: int = 20, std_dev: int = 2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_indicators(historical: pd.DataFrame, min_history_days: int = 50) -> Indicators:
    """
    Calculate indicator values from OHLC DataFrame (index datetime).

    Returns single-latest values for each indicator; if insufficient data, fields are None.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if historical is None or historical.empty or not required_cols.issubset(historical.columns):
        return Indicators(*([None] * 14))

    df = historical.copy()
    df = df.sort_index()

    # If insufficient rows
    if len(df) < min_history_days:
        return Indicators(*([None] * 14))

    close = df['Close']

    # SMAs
    sma_20_s = close.rolling(window=20).mean()
    sma_50_s = close.rolling(window=50).mean()

    # EMAs
    ema_12_s = _ema(close, 12)
    ema_26_s = _ema(close, 26)

    # RSI
    rsi_s = _rsi(close, 14)
    rsi_val = rsi_s.iloc[-1]
    if pd.isna(rsi_val):
        rsi_val = None
    elif rsi_val < 0 or rsi_val > 100:
        rsi_val = None

    # MACD
    macd_s, macd_signal_s, macd_hist_s = _macd(close)

    # Bollinger
    bb_upper_s, bb_middle_s, bb_lower_s = _bollinger(close)
    # Position in band
    if not pd.isna(bb_upper_s.iloc[-1]) and not pd.isna(bb_lower_s.iloc[-1]):
        if bb_upper_s.iloc[-1] != bb_lower_s.iloc[-1]:
            bb_pos = float((close.iloc[-1] - bb_lower_s.iloc[-1]) / (bb_upper_s.iloc[-1] - bb_lower_s.iloc[-1]))
            bb_pos = max(0.0, min(1.0, bb_pos))
        else:
            bb_pos = None
    else:
        bb_pos = None

    # ATR
    atr_s = _atr(df, 14)

    # Realized volatility 30d
    vol_30 = close.pct_change().rolling(window=30).std()

    ind = Indicators(
        sma_20=float(sma_20_s.iloc[-1]) if not pd.isna(sma_20_s.iloc[-1]) else None,
        sma_50=float(sma_50_s.iloc[-1]) if not pd.isna(sma_50_s.iloc[-1]) else None,
        ema_12=float(ema_12_s.iloc[-1]) if not pd.isna(ema_12_s.iloc[-1]) else None,
        ema_26=float(ema_26_s.iloc[-1]) if not pd.isna(ema_26_s.iloc[-1]) else None,
        rsi_14=float(rsi_val) if rsi_val is not None else None,
        macd=float(macd_s.iloc[-1]) if not pd.isna(macd_s.iloc[-1]) else None,
        macd_signal=float(macd_signal_s.iloc[-1]) if not pd.isna(macd_signal_s.iloc[-1]) else None,
        macd_histogram=float(macd_hist_s.iloc[-1]) if not pd.isna(macd_hist_s.iloc[-1]) else None,
        bb_middle=float(bb_middle_s.iloc[-1]) if not pd.isna(bb_middle_s.iloc[-1]) else None,
        bb_upper=float(bb_upper_s.iloc[-1]) if not pd.isna(bb_upper_s.iloc[-1]) else None,
        bb_lower=float(bb_lower_s.iloc[-1]) if not pd.isna(bb_lower_s.iloc[-1]) else None,
        bb_position=float(bb_pos) if bb_pos is not None else None,
        atr_14=float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else None,
        realized_vol_30d=float(vol_30.iloc[-1]) if not pd.isna(vol_30.iloc[-1]) else None,
    )

    return ind

