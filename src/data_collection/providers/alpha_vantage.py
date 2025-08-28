"""
Alpha Vantage API provider implementation.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from .base import BaseRateProvider, DataProviderError, RateLimitError
from ..models import ExchangeRate, DataSource


logger = logging.getLogger(__name__)


class AlphaVantageProvider(BaseRateProvider):
    """
    Provider for Alpha Vantage FX API.
    
    API Documentation: https://www.alphavantage.co/documentation/#fx
    Free tier: 25 requests/day, 5 requests/minute
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._max_requests_per_window = 4  # Conservative limit (5/min limit)
        
        if not api_key:
            raise ValueError("Alpha Vantage requires an API key")
    
    @property
    def source(self) -> DataSource:
        return DataSource.ALPHA_VANTAGE
    
    @property
    def base_url(self) -> str:
        return "https://www.alphavantage.co/query"
    
    async def _fetch_rate(self, base_currency: str, quote_currency: str) -> ExchangeRate:
        """
        Fetch exchange rate from Alpha Vantage API.
        
        API endpoint: GET /query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=EUR&apikey=KEY
        """
        if not self._client:
            raise DataProviderError("HTTP client not initialized")
        
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': base_currency,
            'to_currency': quote_currency,
            'apikey': self.api_key
        }
        
        try:
            response = await self._client.get(self.base_url, params=params)
            
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code != 200:
                raise DataProviderError(f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                raise DataProviderError(f"API error: {data['Error Message']}")
            
            if 'Note' in data:
                # Alpha Vantage returns this when rate limit is hit
                raise RateLimitError(f"API note: {data['Note']}")
            
            # Extract rate data
            rate_data = data.get('Realtime Currency Exchange Rate', {})
            if not rate_data:
                raise DataProviderError("No exchange rate data found in response")
            
            # Parse the rate value
            rate_key = '5. Exchange Rate'
            if rate_key not in rate_data:
                raise DataProviderError("Exchange rate not found in response")
            
            rate_value = float(rate_data[rate_key])
            
            # Validate the rate
            if not self._validate_rate(rate_value, base_currency, quote_currency):
                raise DataProviderError(f"Invalid rate value: {rate_value}")
            
            # Parse timestamp
            timestamp_key = '6. Last Refreshed'
            timestamp_str = rate_data.get(timestamp_key, '')
            try:
                # Alpha Vantage format: "2023-08-28 15:30:01"
                if timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.utcnow()
            except ValueError:
                timestamp = datetime.utcnow()
                logger.warning(f"Could not parse timestamp: {timestamp_str}")
            
            # Extract bid/ask if available
            bid_key = '8. Bid Price'
            ask_key = '9. Ask Price'
            
            bid = None
            ask = None
            spread = None
            
            try:
                if bid_key in rate_data and rate_data[bid_key]:
                    bid = float(rate_data[bid_key])
                if ask_key in rate_data and rate_data[ask_key]:
                    ask = float(rate_data[ask_key])
                
                if bid and ask:
                    spread = ask - bid
            except (ValueError, TypeError):
                logger.debug("Could not parse bid/ask prices")
            
            return ExchangeRate(
                base_currency=base_currency,
                quote_currency=quote_currency,
                rate=rate_value,
                timestamp=timestamp,
                source=self.source,
                bid=bid,
                ask=ask,
                spread=spread,
                raw_data=data
            )
            
        except Exception as e:
            if isinstance(e, (DataProviderError, RateLimitError)):
                raise
            raise DataProviderError(f"Failed to fetch rate from Alpha Vantage: {str(e)}")
    
    async def get_intraday_data(self, base_currency: str, quote_currency: str, 
                              interval: str = "5min") -> Dict[str, Any]:
        """
        Get intraday FX data (for future use in volatility analysis).
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code  
            interval: 1min, 5min, 15min, 30min, 60min
            
        Returns:
            Raw intraday data from API
        """
        if not self._client:
            raise DataProviderError("HTTP client not initialized")
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': base_currency,
            'to_symbol': quote_currency,
            'interval': interval,
            'apikey': self.api_key
        }
        
        try:
            response = await self._client.get(self.base_url, params=params)
            
            if response.status_code != 200:
                raise DataProviderError(f"HTTP {response.status_code}")
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                raise DataProviderError(f"API error: {data['Error Message']}")
            
            if 'Note' in data:
                raise RateLimitError(f"Rate limit: {data['Note']}")
            
            return data
            
        except Exception as e:
            if isinstance(e, (DataProviderError, RateLimitError)):
                raise
            raise DataProviderError(f"Failed to fetch intraday data: {str(e)}")