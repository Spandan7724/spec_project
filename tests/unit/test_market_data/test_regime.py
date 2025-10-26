from src.data_collection.market_data.regime import classify_regime
from src.data_collection.market_data.indicators import Indicators


def test_regime_up_bullish():
    ind = Indicators(
        sma_20=1.05,
        sma_50=1.00,
        ema_12=1.04,
        ema_26=1.02,
        rsi_14=65.0,
        macd=0.01,
        macd_signal=0.009,
        macd_histogram=0.001,
        bb_middle=1.02,
        bb_upper=1.06,
        bb_lower=0.98,
        bb_position=0.6,
        atr_14=0.005,
        realized_vol_30d=0.01,
    )
    regime = classify_regime(latest_price=1.06, ind=ind)
    assert regime.trend_direction == "up"
    assert regime.bias == "bullish"


def test_regime_down_bearish():
    ind = Indicators(
        sma_20=0.95,
        sma_50=1.00,
        ema_12=0.96,
        ema_26=0.98,
        rsi_14=35.0,
        macd=-0.01,
        macd_signal=-0.009,
        macd_histogram=-0.001,
        bb_middle=0.98,
        bb_upper=1.02,
        bb_lower=0.94,
        bb_position=0.4,
        atr_14=0.005,
        realized_vol_30d=0.01,
    )
    regime = classify_regime(latest_price=0.94, ind=ind)
    assert regime.trend_direction == "down"
    assert regime.bias == "bearish"

