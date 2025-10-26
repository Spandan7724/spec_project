"""Regime classification based on indicators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .indicators import Indicators


@dataclass
class Regime:
    trend_direction: Optional[str]  # "up" | "down" | "sideways" | None
    bias: Optional[str]            # "bullish" | "bearish" | "neutral" | None


def classify_regime(latest_price: Optional[float], ind: Indicators) -> Regime:
    """Classify trend direction and bias using indicators."""
    trend: Optional[str] = None
    bias: Optional[str] = None

    # Trend direction rules
    if latest_price is not None and ind.sma_50 is not None and ind.sma_20 is not None:
        if latest_price > ind.sma_50 and ind.sma_20 > ind.sma_50:
            trend = "up"
        elif latest_price < ind.sma_50 and ind.sma_20 < ind.sma_50:
            trend = "down"
        else:
            trend = "sideways"

    # Bias rules using RSI and MACD
    if ind.rsi_14 is not None and ind.macd is not None:
        if ind.rsi_14 > 60 and ind.macd > 0:
            bias = "bullish"
        elif ind.rsi_14 < 40 and ind.macd < 0:
            bias = "bearish"
        else:
            bias = "neutral"

    return Regime(trend_direction=trend, bias=bias)

