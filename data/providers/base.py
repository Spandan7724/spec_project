"""
Base interface for FX data providers.

Defines the standard interface that all external API providers must implement
for fetching foreign exchange rates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class FXRateData:
    """Standard data structure for FX rates from any provider."""
    currency_pair: str
    rate: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    timestamp: datetime = None
    provider: str = ""
    volume: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FXDataProvider(ABC):
    """
    Abstract base class for all FX data providers.
    
    Defines the standard interface that providers like Alpha Vantage,
    Fixer.io, etc. must implement.
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.request_count = 0
        self.error_count = 0
        self.last_request_time: Optional[datetime] = None
        
    @abstractmethod
    async def fetch_rate(self, from_currency: str, to_currency: str) -> Optional[FXRateData]:
        """
        Fetch exchange rate for a single currency pair.
        
        Args:
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'EUR')
            
        Returns:
            FXRateData object or None if fetch failed
        """
        pass
    
    @abstractmethod
    async def fetch_multiple_rates(self, currency_pairs: List[str]) -> Dict[str, Optional[FXRateData]]:
        """
        Fetch exchange rates for multiple currency pairs.
        
        Args:
            currency_pairs: List of currency pairs (e.g., ['USD/EUR', 'USD/GBP'])
            
        Returns:
            Dictionary mapping currency pairs to FXRateData objects
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (has API key, not rate limited, etc.)"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, any]:
        """Return information about API rate limits."""
        pass
    
    async def fetch_with_retry(
        self, 
        from_currency: str, 
        to_currency: str, 
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[FXRateData]:
        """
        Fetch rate with automatic retry logic.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code  
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            FXRateData object or None if all attempts failed
        """
        for attempt in range(max_retries + 1):
            try:
                result = await self.fetch_rate(from_currency, to_currency)
                if result:
                    return result
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                self.error_count += 1
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"All {max_retries + 1} attempts failed for {self.name}")
        return None
    
    def _parse_currency_pair(self, currency_pair: str) -> tuple[str, str]:
        """
        Parse currency pair string into from/to currencies.
        
        Args:
            currency_pair: String like 'USD/EUR' or 'USDEUR'
            
        Returns:
            Tuple of (from_currency, to_currency)
        """
        if '/' in currency_pair:
            return currency_pair.split('/')
        elif len(currency_pair) == 6:
            return currency_pair[:3], currency_pair[3:]
        else:
            raise ValueError(f"Invalid currency pair format: {currency_pair}")
    
    def _validate_rate(self, rate: Decimal, currency_pair: str) -> bool:
        """
        Validate that a rate is reasonable (not obviously wrong).
        
        Args:
            rate: Exchange rate to validate
            currency_pair: Currency pair for context
            
        Returns:
            True if rate seems valid, False otherwise
        """
        # Basic sanity checks
        if rate <= 0:
            return False
        
        # Most exchange rates should be between 0.001 and 1000
        if rate < Decimal('0.001') or rate > Decimal('1000'):
            logger.warning(f"Suspicious rate {rate} for {currency_pair}")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, any]:
        """Get provider statistics."""
        success_rate = 0
        if self.request_count > 0:
            success_rate = ((self.request_count - self.error_count) / self.request_count) * 100
        
        return {
            "name": self.name,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": f"{success_rate:.1f}%",
            "last_request": self.last_request_time,
            "is_available": self.is_available()
        }