"""
Data models for exchange rate information.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class DataSource(Enum):
    """Enum for different data sources."""
    EXCHANGE_RATE_HOST = "exchange_rate_host"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class ExchangeRate:
    """
    Represents a currency exchange rate at a specific point in time.
    """
    base_currency: str  # e.g., "USD"
    quote_currency: str  # e.g., "EUR"
    rate: float  # How much quote currency you get for 1 unit of base currency
    timestamp: datetime
    source: DataSource
    bid: Optional[float] = None  # Best bid price (what buyers are willing to pay)
    ask: Optional[float] = None  # Best ask price (what sellers want)
    spread: Optional[float] = None  # Difference between ask and bid
    raw_data: Optional[Dict[str, Any]] = None  # Original API response for debugging
    
    @property
    def currency_pair(self) -> str:
        """Returns currency pair in standard format: BASE/QUOTE"""
        return f"{self.base_currency}/{self.quote_currency}"
    
    @property
    def mid_rate(self) -> float:
        """Returns mid-market rate (average of bid/ask if available, otherwise rate)"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.rate
    
    @property
    def spread_bps(self) -> Optional[int]:
        """Returns spread in basis points (1 basis point = 0.01%)"""
        if self.spread:
            return int(self.spread / self.rate * 10000)
        elif self.bid and self.ask:
            spread = self.ask - self.bid
            return int(spread / self.rate * 10000)
        return None
    
    def __str__(self) -> str:
        return f"{self.currency_pair}: {self.rate:.4f} ({self.source.value})"


@dataclass 
class RateCollectionResult:
    """
    Result of collecting rates from multiple sources.
    """
    currency_pair: str
    rates: list[ExchangeRate]
    best_rate: Optional[ExchangeRate] = None  # Rate with highest confidence/reliability
    timestamp: datetime = None
    errors: list[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.errors is None:
            self.errors = []
        
        # Determine best rate (prefer sources in order: Alpha Vantage, ExchangeRate.host, Yahoo)
        if self.rates:
            source_priority = {
                DataSource.ALPHA_VANTAGE: 3,
                DataSource.EXCHANGE_RATE_HOST: 2,
                DataSource.YAHOO_FINANCE: 1
            }
            self.best_rate = max(self.rates, key=lambda r: source_priority.get(r.source, 0))
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful data source calls"""
        total_sources = 3  # We have 3 providers
        successful_sources = len(self.rates)
        return (successful_sources / total_sources) * 100
    
    @property
    def has_data(self) -> bool:
        """Whether we successfully got any rate data"""
        return len(self.rates) > 0