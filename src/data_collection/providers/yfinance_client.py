"""yfinance provider implementation for FX pairs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from src.data_collection.providers.base import BaseProvider, ProviderRate
from src.utils.decorators import log_execution, retry
from src.utils.errors import DataProviderError
from src.utils.logging import get_logger


logger = get_logger(__name__)


class YFinanceClient(BaseProvider):
    NAME = "yfinance"

    @staticmethod
    def get_symbol(base: str, quote: str) -> str:
        return f"{base}{quote}=X"

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    @log_execution(log_args=False, log_result=False)
    async def get_rate(self, base: str, quote: str) -> ProviderRate:
        self.validate_pair(base, quote)
        symbol = self.get_symbol(base, quote)

        try:
            # yfinance is sync, but our interface is async – run as blocking
            # Keep it simple: direct call; tests will mock Ticker
            ticker = yf.Ticker(symbol)

            # Prefer fast_info if available
            bid: Optional[float] = None
            ask: Optional[float] = None
            mid: Optional[float] = None

            try:
                fi = getattr(ticker, "fast_info", None)
                if fi:
                    bid = float(fi.get("bid", 0) or 0) or None
                    ask = float(fi.get("ask", 0) or 0) or None
                    last = float(fi.get("last_price", 0) or 0) or None
                    if bid and ask and bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                    elif last and last > 0:
                        mid = last
            except Exception:
                # fall back
                pass

            if mid is None:
                # Fallback: try info dict
                try:
                    info = getattr(ticker, "info", {}) or {}
                    bid = float(info.get("bid", 0) or 0) or bid
                    ask = float(info.get("ask", 0) or 0) or ask
                    last = float(info.get("regularMarketPrice", 0) or 0) or None
                    if bid and ask and bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                    elif last and last > 0:
                        mid = last
                except Exception:
                    pass

            if mid is None or mid <= 0:
                raise DataProviderError(f"No price available for {symbol}")

            pr = ProviderRate(
                source=self.NAME,
                rate=float(mid),
                bid=float(bid) if bid else None,
                ask=float(ask) if ask else None,
                timestamp=datetime.now(timezone.utc),
                notes=[],
            )
            pr.validate()
            return pr

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            raise DataProviderError(str(e))

    async def health_check(self) -> bool:
        try:
            _ = await self.get_rate("USD", "EUR")
            return True
        except Exception:
            return False

