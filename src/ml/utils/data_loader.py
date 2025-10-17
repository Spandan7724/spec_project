"""Data loader integration between Layer 1 collectors and the ML stack."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import logging

from src.data_collection.analysis.historical_data import HistoricalDataCollector
from src.data_collection.economic.calendar_collector import EconomicCalendarCollector

logger = logging.getLogger(__name__)


class Layer1DataLoader:
    """Fetches time-series data required for ML training and inference."""

    def __init__(self, *, min_quality: float = 0.5) -> None:
        self._historical_collector = HistoricalDataCollector()
        self._calendar_collector = EconomicCalendarCollector()
        self._min_quality = min_quality

    async def fetch_historical_rates(
        self,
        currency_pair: str,
        days: int,
    ) -> pd.DataFrame:
        """Return OHLCV frame with returns for the requested window."""

        dataset = await self._historical_collector.get_historical_data(
            currency_pair=currency_pair,
            days=days,
            force_refresh=False,
        )

        if not dataset or not dataset.data:
            raise ValueError(f"No historical data available for {currency_pair}")

        if dataset.data_quality is not None and dataset.data_quality < self._min_quality:
            raise ValueError(
                f"Historical data quality too low for {currency_pair}: {dataset.data_quality:.2%}"
            )

        df = dataset.to_dataframe().copy()
        df = df.sort_index()
        if df.empty:
            raise ValueError(f"Historical dataframe empty for {currency_pair}")

        df["returns"] = df["close"].pct_change(fill_method=None)
        df["log_returns"] = np.log(df["close"].div(df["close"].shift(1)))
        df.dropna(inplace=True)

        if df.empty:
            raise ValueError(f"Historical data after preprocessing empty for {currency_pair}")

        return df

    async def fetch_economic_events(
        self,
        currency_pair: str,
        days: int,
    ) -> pd.DataFrame:
        """Return a frame of upcoming events affecting the currency pair."""

        try:
            calendar = await self._calendar_collector.get_economic_calendar(days_ahead=days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Economic calendar fetch failed: %s", exc)
            return pd.DataFrame()
        if not calendar:
            return pd.DataFrame()

        events = calendar.get_events_for_pair(currency_pair)
        if not events:
            return pd.DataFrame()

        records = []
        for event in events:
            records.append(
                {
                    "date": event.release_date,
                    "indicator": event.title,
                    "currency": event.currency,
                    "impact": event.impact.value if hasattr(event.impact, "value") else str(event.impact),
                    "forecast": event.forecast_value,
                    "previous": event.previous_value,
                    "actual": event.actual_value,
                }
            )

        df = pd.DataFrame.from_records(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    async def fetch_combined_dataset(
        self,
        currency_pair: str,
        days: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return price, indicator, and economic frames for downstream usage."""

        prices = await self.fetch_historical_rates(currency_pair, days)
        indicators = self._compute_indicators(prices)

        base, quote = currency_pair.split("/")
        events = await self.fetch_economic_events(currency_pair, days)
        if not events.empty:
            events = events[(events["currency"] == base) | (events["currency"] == quote)]

        return prices, indicators, events

    def load_combined_dataset(
        self,
        currency_pair: str,
        days: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Synchronous wrapper for use in training scripts."""

        return asyncio.run(self.fetch_combined_dataset(currency_pair, days))

    def load_historical_rates(self, currency_pair: str, days: int) -> pd.DataFrame:
        """Convenience synchronous wrapper for historical prices."""

        return asyncio.run(self.fetch_historical_rates(currency_pair, days))

    def load_economic_events(self, currency_pair: str, days: int) -> pd.DataFrame:
        """Convenience synchronous wrapper for economic calendar data."""

        return asyncio.run(self.fetch_economic_events(currency_pair, days))

    @staticmethod
    def _compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
        """Derive technical indicators directly from price history."""

        indicators = pd.DataFrame(index=prices.index.copy())
        closes = prices["close"]

        indicators["sma_20"] = closes.rolling(20).mean()
        indicators["sma_50"] = closes.rolling(50).mean()
        indicators["sma_200"] = closes.rolling(200).mean()

        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain.div(loss.replace(0, np.nan))
        indicators["rsi"] = 100 - (100 / (1 + rs))

        mid = closes.rolling(20).mean()
        std = closes.rolling(20).std()
        indicators["bb_middle"] = mid
        indicators["bb_upper"] = mid + 2 * std
        indicators["bb_lower"] = mid - 2 * std
        indicators["bb_width"] = indicators["bb_upper"] - indicators["bb_lower"]
        indicators["bb_position"] = (closes - indicators["bb_lower"]).div(
            indicators["bb_width"].replace(0, np.nan)
        )

        indicators["volatility"] = prices["returns"].rolling(20).std()

        macd_fast = closes.ewm(span=12, adjust=False).mean()
        macd_slow = closes.ewm(span=26, adjust=False).mean()
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        indicators["macd"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_line - macd_signal

        indicators = indicators.replace([np.inf, -np.inf], np.nan)

        # Drop any indicator columns that never produced a value in the
        # requested window (e.g., long moving averages without enough history).
        empty_cols = [col for col in indicators.columns if indicators[col].isna().all()]
        if empty_cols:
            logger.debug("Dropping empty indicator columns due to insufficient history: %s", empty_cols)
            indicators = indicators.drop(columns=empty_cols)

        indicators = indicators.ffill().dropna()
        return indicators
