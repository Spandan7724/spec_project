"""
Main FX data collector that orchestrates multiple providers.

Handles provider failover, data validation, and storage coordination
for real-time foreign exchange rate collection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from .providers.base import FXDataProvider, FXRateData
from .providers.alphavantage import AlphaVantageProvider
from .providers.fixer_io import FixerIOProvider

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    """Result of a data collection cycle."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    successful_pairs: Set[str] = field(default_factory=set)
    failed_pairs: Set[str] = field(default_factory=set)
    provider_stats: Dict[str, dict] = field(default_factory=dict)
    rates_collected: Dict[str, FXRateData] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class FXDataCollector:
    """
    Main orchestrator for collecting FX data from multiple providers.
    
    Features:
    - Multi-provider redundancy with automatic failover
    - Rate limiting respect across all providers
    - Data validation and outlier detection
    - Simple in-memory storage for MVP
    - Provider health monitoring
    """
    
    def __init__(self, currency_pairs: List[str]):
        self.currency_pairs = currency_pairs
        self.providers: List[FXDataProvider] = []
        
        # In-memory storage for MVP (replace with database later)
        self.latest_rates: Dict[str, FXRateData] = {}
        self.rate_history: Dict[str, List[FXRateData]] = {pair: [] for pair in currency_pairs}
        
        # Collection statistics
        self.total_collections = 0
        self.successful_collections = 0
        self.last_collection_time: Optional[datetime] = None
        self.collection_errors: List[str] = []
        
        # Data validation settings
        self.max_rate_change_percent = 5.0  # Maximum allowed rate change between collections
        self.max_history_size = 1000  # Keep last 1000 rates per pair in memory
    
    def add_provider(self, provider: FXDataProvider):
        """Add a data provider to the collector."""
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.name}")
    
    def setup_default_providers(self, alphavantage_key: Optional[str] = None, fixer_key: Optional[str] = None):
        """Set up default providers with API keys."""
        if alphavantage_key:
            self.add_provider(AlphaVantageProvider(alphavantage_key))
        
        if fixer_key:
            self.add_provider(FixerIOProvider(fixer_key))
        
        if not self.providers:
            logger.warning("No API keys provided - collector will not be able to fetch real data")
    
    async def collect_all_rates(self) -> CollectionResult:
        """
        Collect rates for all configured currency pairs.
        
        Uses provider failover - tries each provider until successful or all fail.
        """
        result = CollectionResult()
        self.total_collections += 1
        
        logger.info(f"Starting collection cycle for {len(self.currency_pairs)} pairs")
        
        for pair in self.currency_pairs:
            rate_data = await self._collect_single_pair(pair)
            
            if rate_data:
                result.successful_pairs.add(pair)
                result.rates_collected[pair] = rate_data
                
                # Store in memory
                self.latest_rates[pair] = rate_data
                self._add_to_history(pair, rate_data)
                
                logger.debug(f"Successfully collected {pair}: {rate_data.rate}")
            else:
                result.failed_pairs.add(pair)
                error_msg = f"Failed to collect rate for {pair} from all providers"
                result.errors.append(error_msg)
                logger.warning(error_msg)
        
        # Update statistics
        if result.successful_pairs:
            self.successful_collections += 1
        
        self.last_collection_time = datetime.utcnow()
        
        # Collect provider statistics
        for provider in self.providers:
            result.provider_stats[provider.name] = provider.get_stats()
        
        logger.info(f"Collection completed: {len(result.successful_pairs)}/{len(self.currency_pairs)} pairs successful")
        return result
    
    async def _collect_single_pair(self, currency_pair: str) -> Optional[FXRateData]:
        """
        Collect rate for a single currency pair using provider failover.
        
        Tries providers in order until one succeeds or all fail.
        """
        from_currency, to_currency = self._parse_currency_pair(currency_pair)
        
        for provider in self.providers:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} not available, skipping")
                continue
            
            try:
                rate_data = await provider.fetch_with_retry(from_currency, to_currency)
                
                if rate_data and self._validate_new_rate(rate_data):
                    logger.debug(f"Successfully fetched {currency_pair} from {provider.name}")
                    return rate_data
                else:
                    logger.warning(f"Invalid rate data from {provider.name} for {currency_pair}")
                    
            except Exception as e:
                logger.error(f"Error fetching {currency_pair} from {provider.name}: {e}")
                continue
        
        return None
    
    def _validate_new_rate(self, new_rate: FXRateData) -> bool:
        """
        Validate a new rate against historical data and sanity checks.
        
        Checks for:
        - Reasonable rate values
        - Sudden large changes from previous rates
        - Provider consistency
        """
        pair = new_rate.currency_pair
        
        # Basic sanity check
        if new_rate.rate <= 0:
            logger.warning(f"Invalid rate value {new_rate.rate} for {pair}")
            return False
        
        # Check against previous rate if available
        if pair in self.latest_rates:
            previous_rate = self.latest_rates[pair]
            change_percent = abs((new_rate.rate - previous_rate.rate) / previous_rate.rate) * 100
            
            if change_percent > self.max_rate_change_percent:
                logger.warning(
                    f"Large rate change detected for {pair}: {change_percent:.2f}% "
                    f"(from {previous_rate.rate} to {new_rate.rate})"
                )
                # For MVP, we'll log but still accept - in production might reject
        
        return True
    
    def _add_to_history(self, currency_pair: str, rate_data: FXRateData):
        """Add rate data to historical storage with size limiting."""
        history = self.rate_history[currency_pair]
        history.append(rate_data)
        
        # Keep only the most recent rates to limit memory usage
        if len(history) > self.max_history_size:
            history.pop(0)
    
    def _parse_currency_pair(self, currency_pair: str) -> tuple[str, str]:
        """Parse currency pair string into from/to currencies."""
        if '/' in currency_pair:
            return currency_pair.split('/')
        elif len(currency_pair) == 6:
            return currency_pair[:3], currency_pair[3:]
        else:
            raise ValueError(f"Invalid currency pair format: {currency_pair}")
    
    def get_latest_rate(self, currency_pair: str) -> Optional[FXRateData]:
        """Get the latest rate for a currency pair."""
        return self.latest_rates.get(currency_pair)
    
    def get_rate_history(self, currency_pair: str, limit: int = 100) -> List[FXRateData]:
        """Get historical rates for a currency pair."""
        history = self.rate_history.get(currency_pair, [])
        return history[-limit:] if limit else history
    
    def get_all_latest_rates(self) -> Dict[str, FXRateData]:
        """Get latest rates for all currency pairs."""
        return self.latest_rates.copy()
    
    def get_collection_stats(self) -> Dict[str, any]:
        """Get overall collection statistics."""
        success_rate = 0
        if self.total_collections > 0:
            success_rate = (self.successful_collections / self.total_collections) * 100
        
        provider_health = {}
        for provider in self.providers:
            provider_health[provider.name] = {
                **provider.get_stats(),
                "rate_limits": provider.get_rate_limit_info()
            }
        
        return {
            "total_collections": self.total_collections,
            "successful_collections": self.successful_collections, 
            "success_rate": f"{success_rate:.1f}%",
            "last_collection": self.last_collection_time,
            "currency_pairs_configured": len(self.currency_pairs),
            "pairs_with_data": len(self.latest_rates),
            "providers": provider_health,
            "recent_errors": self.collection_errors[-10:],  # Last 10 errors
        }
    
    async def health_check(self) -> Dict[str, any]:
        """Perform health check on all providers and data freshness."""
        health_status = {
            "overall_healthy": True,
            "providers": {},
            "data_freshness": {},
            "issues": []
        }
        
        # Check each provider
        for provider in self.providers:
            provider_healthy = provider.is_available()
            health_status["providers"][provider.name] = {
                "healthy": provider_healthy,
                "stats": provider.get_stats(),
                "rate_limits": provider.get_rate_limit_info()
            }
            
            if not provider_healthy:
                health_status["overall_healthy"] = False
                health_status["issues"].append(f"Provider {provider.name} is not available")
        
        # Check data freshness
        now = datetime.utcnow()
        for pair, rate_data in self.latest_rates.items():
            age_minutes = (now - rate_data.timestamp).total_seconds() / 60
            is_fresh = age_minutes < 5  # Data should be less than 5 minutes old
            
            health_status["data_freshness"][pair] = {
                "fresh": is_fresh,
                "age_minutes": age_minutes,
                "last_update": rate_data.timestamp
            }
            
            if not is_fresh:
                health_status["overall_healthy"] = False
                health_status["issues"].append(f"Stale data for {pair} ({age_minutes:.1f} minutes old)")
        
        return health_status
    
    async def cleanup(self):
        """Clean up resources."""
        for provider in self.providers:
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    logger.warning(f"Error closing provider {provider.name}: {e}")
        
        logger.info("Data collector cleanup completed")