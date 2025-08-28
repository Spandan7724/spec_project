"""
Base classes for exchange rate data providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict
import logging
import httpx
from datetime import datetime, timedelta

from ..models import ExchangeRate, DataSource


logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


class DataProviderError(Exception):
    """Raised when data provider encounters an error."""
    pass


class BaseRateProvider(ABC):
    """
    Abstract base class for exchange rate providers.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limit_window = timedelta(minutes=1)
        self._max_requests_per_window = 60  # Default conservative limit
        self._request_timestamps: list[datetime] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers=self._get_headers(),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    @property
    @abstractmethod
    def source(self) -> DataSource:
        """Return the data source enum for this provider."""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the API."""
        pass
    
    @abstractmethod
    async def _fetch_rate(self, base_currency: str, quote_currency: str) -> ExchangeRate:
        """
        Fetch exchange rate from the provider's API.
        
        Args:
            base_currency: Base currency code (e.g., 'USD')
            quote_currency: Quote currency code (e.g., 'EUR')
            
        Returns:
            ExchangeRate object with the fetched data
            
        Raises:
            RateLimitError: If rate limit is exceeded
            DataProviderError: If API call fails or returns invalid data
        """
        pass
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            'User-Agent': 'CurrencyTimingAdvisor/1.0',
            'Accept': 'application/json',
        }
    
    def _check_rate_limit(self) -> None:
        """Check if we're within rate limits."""
        now = datetime.utcnow()
        # Remove old timestamps outside the window
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if now - ts < self._rate_limit_window
        ]
        
        if len(self._request_timestamps) >= self._max_requests_per_window:
            raise RateLimitError(f"Rate limit exceeded for {self.source.value}")
        
        self._request_timestamps.append(now)
    
    async def get_rate(self, base_currency: str, quote_currency: str) -> Optional[ExchangeRate]:
        """
        Get exchange rate with error handling and rate limiting.
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            
        Returns:
            ExchangeRate object or None if failed
        """
        try:
            self._check_rate_limit()
            rate = await self._fetch_rate(base_currency.upper(), quote_currency.upper())
            logger.info(f"Successfully fetched {rate.currency_pair} from {self.source.value}")
            return rate
            
        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {self.source.value}: {e}")
            return None
            
        except DataProviderError as e:
            logger.error(f"Data provider error for {self.source.value}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching rate from {self.source.value}: {e}")
            return None
    
    def _validate_rate(self, rate: float, base: str, quote: str) -> bool:
        """
        Basic validation for exchange rates.
        
        Args:
            rate: The exchange rate value
            base: Base currency
            quote: Quote currency
            
        Returns:
            True if rate seems reasonable, False otherwise
        """
        if rate <= 0:
            return False
        
        # Very basic sanity checks (these could be more sophisticated)
        if rate > 1000000:  # No currency pair should be this high
            return False
        
        # For major pairs, rates should be in reasonable ranges
        major_pairs_ranges = {
            'USD/EUR': (0.5, 2.0),
            'USD/GBP': (0.5, 2.0), 
            'USD/JPY': (50, 200),
            'EUR/GBP': (0.5, 1.5),
            'EUR/USD': (0.5, 2.0),
            'GBP/USD': (0.5, 2.0),
        }
        
        pair = f"{base}/{quote}"
        if pair in major_pairs_ranges:
            min_rate, max_rate = major_pairs_ranges[pair]
            if not (min_rate <= rate <= max_rate):
                logger.warning(f"Rate {rate} for {pair} outside expected range {min_rate}-{max_rate}")
                return False
        
        return True
    
    async def test_connection(self) -> bool:
        """
        Test if the provider API is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a common currency pair
            rate = await self.get_rate('USD', 'EUR')
            return rate is not None
        except Exception as e:
            logger.error(f"Connection test failed for {self.source.value}: {e}")
            return False