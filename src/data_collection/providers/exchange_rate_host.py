"""ExchangeRate.host provider implementation."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from src.config import get_config, load_config
from src.data_collection.providers.base import BaseProvider, ProviderRate
from src.utils.decorators import retry, log_execution
from src.utils.errors import DataProviderError
from src.utils.logging import get_logger


logger = get_logger(__name__)


class ExchangeRateHostClient(BaseProvider):
    NAME = "exchange_rate_host"

    def __init__(self) -> None:
        # Ensure configuration is available in test and runtime contexts
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()
        self.base_url: str = (
            cfg.get("api.exchange_rate_host.base_url", "https://api.exchangerate.host")
        )
        self.timeout: float = float(cfg.get("api.exchange_rate_host.timeout", 10))
        # Get API key from environment
        import os
        self.api_key: str = os.getenv("EXCHANGE_RATE_HOST_API_KEY", "")

    @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError, httpx.TimeoutException,))
    @log_execution(log_args=False, log_result=False)
    async def get_rate(self, base: str, quote: str) -> ProviderRate:
        self.validate_pair(base, quote)

        url = f"{self.base_url}/live"
        params = {"currencies": quote}
        if self.api_key:
            params["access_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json() or {}
        except Exception as e:
            logger.error(f"ExchangeRate.host request failed: {e}")
            raise DataProviderError(str(e))

        # Parse response
        try:
            # Check if API returned an error
            if not data.get("success", True):
                error_info = data.get("error", {})
                error_msg = f"API error: {error_info.get('type', 'unknown')} - {error_info.get('info', 'no details')}"
                logger.error(f"ExchangeRate.host API error: {error_msg}")
                raise DataProviderError(error_msg)
            
            # Parse the quotes format: {"quotes": {"USDEUR": 0.860104}}
            quotes = data.get("quotes") or {}
            quote_key = f"{base}{quote}"
            value = quotes.get(quote_key)
            
            if value is None:
                logger.error(f"ExchangeRate.host response missing rate for {quote_key}. Available quotes: {list(quotes.keys())}")
                raise DataProviderError(f"No rate found for {quote_key} in response")
            
            rate = float(value)
            # Parse timestamp from Unix timestamp
            ts = data.get("timestamp")
            if ts:
                timestamp = datetime.fromtimestamp(ts, timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            pr = ProviderRate(
                source=self.NAME,
                rate=rate,
                bid=None,
                ask=None,
                timestamp=timestamp,
                notes=[],
            )
            pr.validate()
            return pr
        except Exception as e:
            logger.error(f"Failed to parse ExchangeRate.host response: {e}")
            raise DataProviderError("Invalid response from ExchangeRate.host")

    async def health_check(self) -> bool:
        try:
            _ = await self.get_rate("USD", "EUR")
            return True
        except Exception:
            return False
