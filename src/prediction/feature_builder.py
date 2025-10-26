import pandas as pd
from typing import Dict, List, Optional

from src.utils.logging import get_logger


logger = get_logger(__name__)


class FeatureBuilder:
    """Build ML features from OHLC data and optional market intelligence."""

    def __init__(self, indicators: List[str]):
        self.indicators = indicators

    def build_features(
        self,
        df: pd.DataFrame,
        mode: str = "price_only",
        market_intel: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Build feature set from OHLC data.

        Args:
            df: OHLC DataFrame (expects Open, High, Low, Close)
            mode: "price_only" or "price_plus_intel"
            market_intel: Optional market intelligence data
        Returns:
            DataFrame with engineered features (NaNs dropped)
        """
        features = df.copy()

        # Price-based features
        features = self._add_technical_indicators(features)

        # Market intelligence features (if enabled)
        if mode == "price_plus_intel" and market_intel:
            features = self._add_market_intel_features(features, market_intel)

        return features.dropna()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to a copy of df and return it."""
        df = df.copy()

        # Simple Moving Averages
        if "sma_5" in self.indicators:
            df["sma_5"] = df["Close"].rolling(window=5).mean()
        if "sma_20" in self.indicators:
            df["sma_20"] = df["Close"].rolling(window=20).mean()
        if "sma_50" in self.indicators:
            df["sma_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Averages
        if "ema_12" in self.indicators:
            df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        if "ema_26" in self.indicators:
            df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # RSI
        if "rsi_14" in self.indicators:
            df["rsi_14"] = self._calculate_rsi(df["Close"], 14)

        # MACD (line and signal)
        if "macd" in self.indicators or "macd_signal" in self.indicators:
            macd, signal = self._calculate_macd(df["Close"]) 
            if "macd" in self.indicators:
                df["macd"] = macd
            if "macd_signal" in self.indicators:
                df["macd_signal"] = signal

        # Bollinger Bands
        if any(ind in self.indicators for ind in ["bb_upper", "bb_lower", "bb_middle"]):
            upper, middle, lower = self._calculate_bollinger_bands(df["Close"]) 
            if "bb_upper" in self.indicators:
                df["bb_upper"] = upper
            if "bb_middle" in self.indicators:
                df["bb_middle"] = middle
            if "bb_lower" in self.indicators:
                df["bb_lower"] = lower

        # ATR
        if "atr_14" in self.indicators:
            df["atr_14"] = self._calculate_atr(df, 14)

        # Realized volatility
        if "volatility_20" in self.indicators:
            df["volatility_20"] = df["Close"].pct_change().rolling(window=20).std()

        return df

    def _add_market_intel_features(
        self, df: pd.DataFrame, market_intel: Dict
    ) -> pd.DataFrame:
        df = df.copy()
        # Example features from market intelligence payload
        if "news" in market_intel:
            df["news_pair_bias"] = market_intel["news"].get("pair_bias", 0.0)
            df["news_confidence"] = self._confidence_to_numeric(
                market_intel["news"].get("confidence", "low")
            )
        if (
            "calendar" in market_intel
            and market_intel["calendar"].get("next_high_event") is not None
        ):
            next_event = market_intel["calendar"]["next_high_event"]
            df["next_event_hours"] = next_event.get("proximity_minutes", 10000) / 60.0
        else:
            df["next_event_hours"] = 10000.0
        return df

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    @staticmethod
    def _calculate_bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ):
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def _confidence_to_numeric(confidence: str) -> float:
        mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
        return mapping.get(confidence, 0.5)

    def build_targets(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """Build target variables (future percentage returns and direction flags)."""
        targets = pd.DataFrame(index=df.index)
        for horizon in horizons:
            targets[f"target_{horizon}d"] = (
                (df["Close"].shift(-horizon) - df["Close"]) / df["Close"] * 100.0
            )
            targets[f"direction_{horizon}d"] = (
                df["Close"].shift(-horizon) > df["Close"]
            ).astype(int)
        return targets

