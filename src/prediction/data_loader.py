import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger


logger = get_logger(__name__)


class HistoricalDataLoader:
    """Load historical OHLC data for prediction."""

    @staticmethod
    def get_yahoo_symbol(base: str, quote: str) -> str:
        """Convert currency pair to Yahoo Finance FX symbol."""
        return f"{base}{quote}=X"

    # Note: For LSTM/intraday support, extend to accept an `interval` parameter
    # (e.g., '1h', '30m') with corresponding yfinance calls and history limits.
    async def fetch_historical_data(
        self,
        base: str,
        quote: str,
        days: int = 365,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from yfinance.

        Returns DataFrame with columns: Open, High, Low, Close
        Index: DatetimeIndex
        """
        symbol = self.get_yahoo_symbol(base, quote)

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            def _download():
                ticker = yf.Ticker(symbol)
                # For daily data, start/end is fine. For intraday, use period + interval.
                if interval == "1d":
                    return ticker.history(start=start_date, end=end_date)

                # Intraday intervals: map to safe period limits
                intraday_max_days = {
                    "4h": 730,
                    "1h": 730,
                    "30m": 60,
                    "15m": 60,
                    "5m": 60,
                    "1m": 7,
                }
                max_days = intraday_max_days.get(interval, min(days, 730))
                period_days = min(days, max_days)
                period_str = f"{period_days}d"
                return ticker.history(period=period_str, interval=interval)

            df = await asyncio.to_thread(_download)

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return None

            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df[required_cols]

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
