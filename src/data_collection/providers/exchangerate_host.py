"""
ExchangeRate.host API provider implementation.
"""
from typing import Optional, Dict
from datetime import datetime
import logging

from .base import BaseRateProvider, DataProviderError
from ..models import ExchangeRate, DataSource


logger = logging.getLogger(__name__)


class ExchangeRateHostProvider(BaseRateProvider):
    """
    Provider for ExchangeRate.host API.
    
    API Documentation: https://exchangerate.host/
    Free tier: 1000 requests/month
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._max_requests_per_window = 50  # Conservative limit for free tier
    
    @property
    def source(self) -> DataSource:
        return DataSource.EXCHANGE_RATE_HOST
    
    @property 
    def base_url(self) -> str:
        return "https://api.exchangerate.host"
    
    def _get_headers(self) -> Dict[str, str]:
        # ExchangeRate.host uses query parameters, not headers for API key
        return super()._get_headers()
    
    async def _fetch_rate(self, base_currency: str, quote_currency: str) -> ExchangeRate:
        """
        Fetch exchange rate from ExchangeRate.host API.
        
        API endpoint: GET /live?access_key=YOUR_KEY&source=USD&currencies=EUR
        According to docs: https://exchangerate.host/documentation
        """
        if not self._client:
            raise DataProviderError("HTTP client not initialized")
        
        params = {
            'source': base_currency,  # Source currency 
            'currencies': quote_currency,  # Target currencies (comma-separated)
        }
        
        # Add API key if available
        if self.api_key:
            params['access_key'] = self.api_key
        
        try:
            url = f"{self.base_url}/live"  # Use /live endpoint per docs
            response = await self._client.get(url, params=params)
            
            if response.status_code == 429:
                raise DataProviderError("Rate limit exceeded")
            elif response.status_code == 401:
                raise DataProviderError("Invalid API key")
            elif response.status_code != 200:
                raise DataProviderError(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Check if API returned success
            if not data.get('success', False):
                error_info = data.get('error', {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get('info', 'Unknown API error')
                else:
                    error_msg = str(error_info)
                raise DataProviderError(f"API error: {error_msg}")
            
            # Extract the rate from quotes object
            # Format: {"quotes": {"USDEUR": 0.8592}}
            quotes = data.get('quotes', {})
            currency_pair_key = f"{base_currency}{quote_currency}"
            
            if currency_pair_key not in quotes:
                raise DataProviderError(f"Rate for {currency_pair_key} not found in quotes")
            
            rate_value = float(quotes[currency_pair_key])
            
            # Validate the rate
            if not self._validate_rate(rate_value, base_currency, quote_currency):
                raise DataProviderError(f"Invalid rate value: {rate_value}")
            
            # Parse timestamp
            timestamp_unix = data.get('timestamp', None)
            try:
                # ExchangeRate.host returns Unix timestamp in seconds
                if timestamp_unix:
                    timestamp = datetime.fromtimestamp(timestamp_unix)
                else:
                    timestamp = datetime.utcnow()
            except (ValueError, TypeError):
                timestamp = datetime.utcnow()
                logger.warning(f"Could not parse timestamp: {timestamp_unix}")
            
            return ExchangeRate(
                base_currency=base_currency,
                quote_currency=quote_currency,
                rate=rate_value,
                timestamp=timestamp,
                source=self.source,
                raw_data=data
            )
            
        except Exception as e:
            if isinstance(e, DataProviderError):
                raise
            raise DataProviderError(f"Failed to fetch rate from ExchangeRate.host: {str(e)}")
    
    async def get_supported_currencies(self) -> list[str]:
        """
        Get list of supported currencies.
        
        Returns:
            List of 3-letter currency codes
        """
        if not self._client:
            raise DataProviderError("HTTP client not initialized")
        
        try:
            url = f"{self.base_url}/symbols"
            response = await self._client.get(url)
            
            if response.status_code != 200:
                raise DataProviderError(f"HTTP {response.status_code}")
            
            data = response.json()
            symbols = data.get('symbols', {})
            return list(symbols.keys())
            
        except Exception as e:
            logger.error(f"Failed to fetch supported currencies: {e}")
            return []