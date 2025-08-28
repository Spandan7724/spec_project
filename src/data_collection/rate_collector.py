"""
Main exchange rate collector that coordinates multiple providers.
"""
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from .models import ExchangeRate, RateCollectionResult
from .providers.exchangerate_host import ExchangeRateHostProvider
from .providers.alpha_vantage import AlphaVantageProvider  
from .providers.yahoo_finance import YahooFinanceProvider
from .providers.base import BaseRateProvider

# Load environment variables
load_dotenv()



# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MultiProviderRateCollector:
    """
    Collects exchange rates from multiple providers with failover logic.
    
    Provides high reliability through:
    - Multiple data sources
    - Async concurrent fetching 
    - Error handling and graceful degradation
    - Rate validation and quality scoring
    """
    
    def __init__(self):
        self.providers: List[BaseRateProvider] = []
        self._initialize_providers()
        
    def _initialize_providers(self) -> None:
        """Initialize available providers based on API keys."""
        
        # ExchangeRate.host
        exchangerate_key = os.getenv('EXCHANGE_RATE_HOST_API_KEY')
        if exchangerate_key:
            try:
                provider = ExchangeRateHostProvider(exchangerate_key)
                self.providers.append(provider)
                logger.info("Initialized ExchangeRate.host provider")
            except Exception as e:
                logger.error(f"Failed to initialize ExchangeRate.host: {e}")
        else:
            logger.warning("No ExchangeRate.host API key found")
        
        # Alpha Vantage
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY') 
        if alpha_key:
            try:
                provider = AlphaVantageProvider(alpha_key)
                self.providers.append(provider)
                logger.info("Initialized Alpha Vantage provider")
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage: {e}")
        else:
            logger.warning("No Alpha Vantage API key found")
        
        # Yahoo Finance (always available, no API key needed)
        try:
            provider = YahooFinanceProvider()
            self.providers.append(provider)
            logger.info("Initialized Yahoo Finance provider")
        except Exception as e:
            logger.error(f"Failed to initialize Yahoo Finance: {e}")
        
        if not self.providers:
            raise RuntimeError("No exchange rate providers could be initialized")
        
        logger.info(f"Initialized {len(self.providers)} exchange rate providers")
    
    async def get_rate(self, base_currency: str, quote_currency: str, 
                      timeout: float = 30.0) -> RateCollectionResult:
        """
        Get exchange rate from all available providers.
        
        Args:
            base_currency: Base currency code (e.g., 'USD')
            quote_currency: Quote currency code (e.g., 'EUR')
            timeout: Total timeout for all provider calls
            
        Returns:
            RateCollectionResult with rates from successful providers
        """
        base_currency = base_currency.upper()
        quote_currency = quote_currency.upper()
        currency_pair = f"{base_currency}/{quote_currency}"
        
        logger.info(f"Fetching rates for {currency_pair} from {len(self.providers)} providers")
        
        # Create async tasks for all providers
        tasks = []
        provider_names = []
        
        for provider in self.providers:
            task = self._fetch_from_provider(provider, base_currency, quote_currency)
            tasks.append(task)
            provider_names.append(provider.source.value)
        
        # Execute all tasks concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout after {timeout}s fetching rates for {currency_pair}")
            results = [None] * len(tasks)
        
        # Process results
        successful_rates = []
        errors = []
        
        for i, result in enumerate(results):
            provider_name = provider_names[i]
            
            if isinstance(result, ExchangeRate):
                successful_rates.append(result)
                logger.debug(f"Successfully got rate from {provider_name}: {result.rate}")
            elif isinstance(result, Exception):
                error_msg = f"{provider_name}: {str(result)}"
                errors.append(error_msg)
                logger.warning(f"Provider {provider_name} failed: {result}")
            else:
                error_msg = f"{provider_name}: No data returned"
                errors.append(error_msg)
                logger.warning(f"Provider {provider_name} returned no data")
        
        result = RateCollectionResult(
            currency_pair=currency_pair,
            rates=successful_rates,
            errors=errors
        )
        
        if result.has_data:
            logger.info(f"Successfully collected {len(successful_rates)} rates for {currency_pair} "
                       f"(success rate: {result.success_rate:.1f}%)")
            
            # Log rate comparison if we have multiple sources
            if len(successful_rates) > 1:
                rates_str = ", ".join([f"{r.source.value}: {r.rate:.4f}" for r in successful_rates])
                logger.info(f"Rate comparison - {rates_str}")
                
                # Check for significant rate differences (potential data quality issues)
                min_rate = min(r.rate for r in successful_rates)
                max_rate = max(r.rate for r in successful_rates)
                diff_pct = ((max_rate - min_rate) / min_rate) * 100
                
                if diff_pct > 1.0:  # More than 1% difference
                    logger.warning(f"Large rate difference detected: {diff_pct:.2f}% "
                                 f"(min: {min_rate:.4f}, max: {max_rate:.4f})")
        else:
            logger.error(f"Failed to get any rates for {currency_pair}")
        
        return result
    
    async def _fetch_from_provider(self, provider: BaseRateProvider, 
                                 base_currency: str, quote_currency: str) -> Optional[ExchangeRate]:
        """
        Fetch rate from a single provider with proper context management.
        
        Args:
            provider: The provider to fetch from
            base_currency: Base currency code
            quote_currency: Quote currency code
            
        Returns:
            ExchangeRate object or None if failed
        """
        try:
            async with provider:
                rate = await provider.get_rate(base_currency, quote_currency)
                return rate
        except Exception as e:
            logger.error(f"Error fetching from {provider.source.value}: {e}")
            return None
    
    async def test_all_providers(self) -> Dict[str, bool]:
        """
        Test connectivity to all providers.
        
        Returns:
            Dictionary mapping provider name to success status
        """
        logger.info("Testing connectivity to all providers...")
        
        results = {}
        tasks = []
        provider_names = []
        
        for provider in self.providers:
            task = self._test_provider(provider)
            tasks.append(task)
            provider_names.append(provider.source.value)
        
        try:
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(test_results):
                provider_name = provider_names[i]
                success = isinstance(result, bool) and result
                results[provider_name] = success
                
                if success:
                    logger.info(f"✅ {provider_name}: Connection OK")
                else:
                    logger.error(f"❌ {provider_name}: Connection failed - {result}")
                    
        except Exception as e:
            logger.error(f"Error testing providers: {e}")
            for name in provider_names:
                results[name] = False
        
        return results
    
    async def _test_provider(self, provider: BaseRateProvider) -> bool:
        """Test a single provider connection."""
        try:
            async with provider:
                return await provider.test_connection()
        except Exception as e:
            logger.error(f"Test failed for {provider.source.value}: {e}")
            return False
    
    def get_provider_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all initialized providers.
        
        Returns:
            List of provider information dictionaries
        """
        provider_info = []
        
        for provider in self.providers:
            info = {
                'name': provider.source.value,
                'source': provider.source,
                'base_url': provider.base_url,
                'has_api_key': provider.api_key is not None,
                'rate_limit': provider._max_requests_per_window,
            }
            provider_info.append(info)
        
        return provider_info