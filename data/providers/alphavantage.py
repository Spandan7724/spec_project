"""
Alpha Vantage FX data provider implementation.

Alpha Vantage offers free tier: 25 requests per day, 5 requests per minute.
API Documentation: https://www.alphavantage.co/documentation/#fx
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import httpx

from .base import FXDataProvider, FXRateData

logger = logging.getLogger(__name__)


class AlphaVantageProvider(FXDataProvider):
    """
    Alpha Vantage FX data provider.
    
    Provides real-time and historical foreign exchange rates.
    Free tier limitations: 25 requests/day, 5 requests/minute
    """
    
    def __init__(self, api_key: str):
        super().__init__(name="AlphaVantage", api_key=api_key)
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting
        self.daily_limit = 25
        self.minute_limit = 5
        self.daily_requests = 0
        self.minute_requests = 0
        self.last_reset_day: Optional[datetime] = None
        self.last_reset_minute: Optional[datetime] = None
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def _reset_counters_if_needed(self):
        """Reset request counters based on time periods."""
        now = datetime.utcnow()
        
        # Reset daily counter
        if (not self.last_reset_day or 
            (now - self.last_reset_day).days >= 1):
            self.daily_requests = 0
            self.last_reset_day = now
        
        # Reset minute counter  
        if (not self.last_reset_minute or
            (now - self.last_reset_minute).seconds >= 60):
            self.minute_requests = 0
            self.last_reset_minute = now
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits."""
        self._reset_counters_if_needed()
        return (self.daily_requests < self.daily_limit and 
                self.minute_requests < self.minute_limit)
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return (self.api_key is not None and 
                self.api_key.strip() != "" and
                self._can_make_request())
    
    def get_rate_limit_info(self) -> Dict[str, any]:
        """Return rate limit information."""
        self._reset_counters_if_needed()
        return {
            "daily_limit": self.daily_limit,
            "daily_used": self.daily_requests,
            "daily_remaining": self.daily_limit - self.daily_requests,
            "minute_limit": self.minute_limit,
            "minute_used": self.minute_requests,
            "minute_remaining": self.minute_limit - self.minute_requests,
        }
    
    async def fetch_rate(self, from_currency: str, to_currency: str) -> Optional[FXRateData]:
        """
        Fetch exchange rate for a single currency pair from Alpha Vantage.
        
        Uses the FX_EXCHANGE_RATE function which provides real-time rates.
        """
        if not self.is_available():
            logger.warning("AlphaVantage provider not available (rate limited or no API key)")
            return None
        
        params = {
            "function": "FX_EXCHANGE_RATE",
            "from_symbol": from_currency.upper(),
            "to_symbol": to_currency.upper(), 
            "apikey": self.api_key
        }
        
        try:
            self.request_count += 1
            self.daily_requests += 1
            self.minute_requests += 1
            self.last_request_time = datetime.utcnow()
            
            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"AlphaVantage API error: {data['Error Message']}")
                self.error_count += 1
                return None
            
            if "Note" in data:
                logger.warning(f"AlphaVantage rate limit warning: {data['Note']}")
                return None
            
            # Parse the response
            rate_data = data.get("Realtime Currency Exchange Rate", {})
            if not rate_data:
                logger.error("No rate data in AlphaVantage response")
                self.error_count += 1
                return None
            
            # Extract rate information
            rate = Decimal(rate_data.get("5. Exchange Rate", "0"))
            bid_price = rate_data.get("8. Bid Price")
            ask_price = rate_data.get("9. Ask Price")
            
            # Validate the rate
            currency_pair = f"{from_currency}/{to_currency}"
            if not self._validate_rate(rate, currency_pair):
                self.error_count += 1
                return None
            
            return FXRateData(
                currency_pair=currency_pair,
                rate=rate,
                bid=Decimal(bid_price) if bid_price else None,
                ask=Decimal(ask_price) if ask_price else None,
                timestamp=datetime.utcnow(),
                provider=self.name
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"AlphaVantage HTTP error: {e}")
            self.error_count += 1
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"AlphaVantage data parsing error: {e}")
            self.error_count += 1
            return None
            
        except Exception as e:
            logger.error(f"AlphaVantage unexpected error: {e}")
            self.error_count += 1
            return None
    
    async def fetch_multiple_rates(self, currency_pairs: List[str]) -> Dict[str, Optional[FXRateData]]:
        """
        Fetch multiple currency pairs with rate limiting.
        
        Note: Alpha Vantage doesn't have a bulk endpoint, so we make individual
        requests with delays to respect rate limits.
        """
        results = {}
        
        for pair in currency_pairs:
            if not self._can_make_request():
                logger.warning(f"Rate limit reached, skipping remaining pairs")
                # Mark remaining pairs as failed
                for remaining_pair in currency_pairs[currency_pairs.index(pair):]:
                    results[remaining_pair] = None
                break
            
            try:
                from_currency, to_currency = self._parse_currency_pair(pair)
                result = await self.fetch_rate(from_currency, to_currency)
                results[pair] = result
                
                # Add delay between requests to avoid hitting per-minute limit
                if len(currency_pairs) > 1:
                    await asyncio.sleep(12)  # 5 requests per minute = 12 seconds between requests
                    
            except ValueError as e:
                logger.error(f"Invalid currency pair {pair}: {e}")
                results[pair] = None
        
        return results
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup when provider is destroyed."""
        try:
            asyncio.create_task(self.close())
        except RuntimeError:
            # Event loop might be closed already
            pass