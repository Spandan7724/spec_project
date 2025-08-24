"""
Fixer.io FX data provider implementation.

Fixer.io offers free tier: 100 requests per month, 1000 requests per hour.
API Documentation: https://fixer.io/documentation
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import httpx

from .base import FXDataProvider, FXRateData

logger = logging.getLogger(__name__)


class FixerIOProvider(FXDataProvider):
    """
    Fixer.io FX data provider.
    
    Provides real-time foreign exchange rates with EUR as base currency.
    Free tier limitations: 100 requests/month, 1000 requests/hour
    """
    
    def __init__(self, api_key: str):
        super().__init__(name="FixerIO", api_key=api_key)
        self.base_url = "http://data.fixer.io/api"
        
        # Rate limiting (very generous for free tier)
        self.monthly_limit = 100
        self.hourly_limit = 1000
        self.monthly_requests = 0
        self.hourly_requests = 0
        self.last_reset_month: Optional[datetime] = None
        self.last_reset_hour: Optional[datetime] = None
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(timeout=10.0)
    
    def _reset_counters_if_needed(self):
        """Reset request counters based on time periods."""
        now = datetime.utcnow()
        
        # Reset monthly counter (simplified - reset every 30 days)
        if (not self.last_reset_month or 
            (now - self.last_reset_month).days >= 30):
            self.monthly_requests = 0
            self.last_reset_month = now
        
        # Reset hourly counter
        if (not self.last_reset_hour or
            (now - self.last_reset_hour).seconds >= 3600):
            self.hourly_requests = 0
            self.last_reset_hour = now
    
    def _can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits.""" 
        self._reset_counters_if_needed()
        return (self.monthly_requests < self.monthly_limit and 
                self.hourly_requests < self.hourly_limit)
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return (self.api_key is not None and 
                self.api_key.strip() != "" and
                self._can_make_request())
    
    def get_rate_limit_info(self) -> Dict[str, any]:
        """Return rate limit information."""
        self._reset_counters_if_needed()
        return {
            "monthly_limit": self.monthly_limit,
            "monthly_used": self.monthly_requests,
            "monthly_remaining": self.monthly_limit - self.monthly_requests,
            "hourly_limit": self.hourly_limit,
            "hourly_used": self.hourly_requests,
            "hourly_remaining": self.hourly_limit - self.hourly_requests,
        }
    
    async def fetch_rate(self, from_currency: str, to_currency: str) -> Optional[FXRateData]:
        """
        Fetch exchange rate for a single currency pair from Fixer.io.
        
        Note: Fixer.io uses EUR as base currency on free tier, so we may need
        to calculate cross rates for non-EUR pairs.
        """
        if not self.is_available():
            logger.warning("FixerIO provider not available (rate limited or no API key)")
            return None
        
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # For free tier, we can only use EUR as base currency
        # So we need to handle different scenarios:
        if from_currency == "EUR":
            return await self._fetch_eur_based_rate(to_currency)
        elif to_currency == "EUR":
            return await self._fetch_to_eur_rate(from_currency)
        else:
            return await self._fetch_cross_rate(from_currency, to_currency)
    
    async def _fetch_eur_based_rate(self, to_currency: str) -> Optional[FXRateData]:
        """Fetch EUR to target currency rate."""
        params = {
            "access_key": self.api_key,
            "symbols": to_currency
        }
        
        return await self._make_request("latest", params, "EUR", to_currency)
    
    async def _fetch_to_eur_rate(self, from_currency: str) -> Optional[FXRateData]:
        """Fetch from currency to EUR rate (inverse of EUR to from_currency)."""
        params = {
            "access_key": self.api_key,
            "symbols": from_currency  
        }
        
        # Get EUR/from_currency rate and invert it
        response_data = await self._make_raw_request("latest", params)
        if not response_data:
            return None
        
        rates = response_data.get("rates", {})
        eur_to_from_rate = rates.get(from_currency)
        
        if not eur_to_from_rate:
            logger.error(f"No rate found for EUR/{from_currency}")
            return None
        
        # Invert the rate to get from_currency/EUR
        inverted_rate = Decimal("1") / Decimal(str(eur_to_from_rate))
        
        currency_pair = f"{from_currency}/EUR"
        if not self._validate_rate(inverted_rate, currency_pair):
            return None
        
        return FXRateData(
            currency_pair=currency_pair,
            rate=inverted_rate,
            timestamp=datetime.utcnow(),
            provider=self.name
        )
    
    async def _fetch_cross_rate(self, from_currency: str, to_currency: str) -> Optional[FXRateData]:
        """Fetch cross rate by using EUR as intermediate currency."""
        params = {
            "access_key": self.api_key,
            "symbols": f"{from_currency},{to_currency}"
        }
        
        response_data = await self._make_raw_request("latest", params)
        if not response_data:
            return None
        
        rates = response_data.get("rates", {})
        eur_to_from = rates.get(from_currency)
        eur_to_to = rates.get(to_currency)
        
        if not eur_to_from or not eur_to_to:
            logger.error(f"Missing rates for cross calculation: {from_currency}={eur_to_from}, {to_currency}={eur_to_to}")
            return None
        
        # Calculate cross rate: (EUR/to_currency) / (EUR/from_currency) = from_currency/to_currency
        cross_rate = Decimal(str(eur_to_to)) / Decimal(str(eur_to_from))
        
        currency_pair = f"{from_currency}/{to_currency}"
        if not self._validate_rate(cross_rate, currency_pair):
            return None
        
        return FXRateData(
            currency_pair=currency_pair,
            rate=cross_rate,
            timestamp=datetime.utcnow(),
            provider=self.name
        )
    
    async def _make_request(self, endpoint: str, params: dict, from_currency: str, to_currency: str) -> Optional[FXRateData]:
        """Make API request and parse response into FXRateData."""
        response_data = await self._make_raw_request(endpoint, params)
        if not response_data:
            return None
        
        rates = response_data.get("rates", {})
        rate_value = rates.get(to_currency)
        
        if not rate_value:
            logger.error(f"No rate found for {to_currency} in Fixer.io response")
            return None
        
        rate = Decimal(str(rate_value))
        currency_pair = f"{from_currency}/{to_currency}"
        
        if not self._validate_rate(rate, currency_pair):
            return None
        
        return FXRateData(
            currency_pair=currency_pair,
            rate=rate,
            timestamp=datetime.utcnow(),
            provider=self.name
        )
    
    async def _make_raw_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make raw HTTP request to Fixer.io API."""
        if not self._can_make_request():
            return None
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            self.request_count += 1
            self.monthly_requests += 1
            self.hourly_requests += 1
            self.last_request_time = datetime.utcnow()
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if not data.get("success", False):
                error_info = data.get("error", {})
                logger.error(f"Fixer.io API error: {error_info}")
                self.error_count += 1
                return None
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Fixer.io HTTP error: {e}")
            self.error_count += 1
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Fixer.io data parsing error: {e}")
            self.error_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Fixer.io unexpected error: {e}")
            self.error_count += 1
            return None
    
    async def fetch_multiple_rates(self, currency_pairs: List[str]) -> Dict[str, Optional[FXRateData]]:
        """
        Fetch multiple currency pairs efficiently.
        
        Fixer.io allows fetching multiple currencies in a single request,
        which is much more efficient than individual requests.
        """
        if not self.is_available():
            return {pair: None for pair in currency_pairs}
        
        results = {}
        
        # Group pairs by base currency to optimize requests
        eur_pairs = []  # Pairs with EUR as base
        to_eur_pairs = []  # Pairs converting to EUR
        cross_pairs = []  # Pairs not involving EUR
        
        for pair in currency_pairs:
            try:
                from_curr, to_curr = self._parse_currency_pair(pair)
                if from_curr == "EUR":
                    eur_pairs.append((pair, to_curr))
                elif to_curr == "EUR": 
                    to_eur_pairs.append((pair, from_curr))
                else:
                    cross_pairs.append((pair, from_curr, to_curr))
            except ValueError as e:
                logger.error(f"Invalid currency pair {pair}: {e}")
                results[pair] = None
        
        # Fetch EUR-based pairs in bulk
        if eur_pairs:
            symbols = ",".join([to_curr for _, to_curr in eur_pairs])
            params = {
                "access_key": self.api_key,
                "symbols": symbols
            }
            
            response_data = await self._make_raw_request("latest", params)
            if response_data:
                rates = response_data.get("rates", {})
                for pair, to_curr in eur_pairs:
                    if to_curr in rates:
                        rate = Decimal(str(rates[to_curr]))
                        if self._validate_rate(rate, pair):
                            results[pair] = FXRateData(
                                currency_pair=pair,
                                rate=rate,
                                timestamp=datetime.utcnow(),
                                provider=self.name
                            )
                        else:
                            results[pair] = None
                    else:
                        results[pair] = None
            else:
                for pair, _ in eur_pairs:
                    results[pair] = None
        
        # Handle other pair types individually (for simplicity in MVP)
        for pair, from_curr in to_eur_pairs:
            results[pair] = await self.fetch_rate(from_curr, "EUR")
        
        for pair, from_curr, to_curr in cross_pairs:
            results[pair] = await self.fetch_rate(from_curr, to_curr)
        
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