"""
Unified Data Integration Layer for Real Market Intelligence.

Combines news, economic calendar, market data, and provider rates 
into a single interface for the multi-agent system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .news_providers import fetch_real_news_data
from .economic_calendar import fetch_real_economic_calendar  
from .market_data import fetch_real_market_data
from .provider_rates import fetch_real_provider_rates

logger = logging.getLogger(__name__)


class RealDataProvider:
    """
    Unified provider for all real market data sources.
    Replaces mock data with actual market intelligence.
    """
    
    def __init__(self, 
                 newsapi_key: Optional[str] = None,
                 fred_api_key: Optional[str] = None):
        """
        Initialize real data provider.
        
        Args:
            newsapi_key: Optional NewsAPI key for enhanced news coverage
            fred_api_key: Optional FRED API key for US economic data
        """
        self.newsapi_key = newsapi_key
        self.fred_api_key = fred_api_key
        
        # Cache for rate limiting
        self.cache = {}
        self.cache_ttl = {
            'news': 1800,      # 30 minutes
            'calendar': 3600,   # 1 hour  
            'market': 300,      # 5 minutes
            'providers': 600    # 10 minutes
        }
    
    async def get_comprehensive_market_intelligence(self,
                                                  currency_pair: str,
                                                  amount: float,
                                                  timeframe_days: int = 7) -> Dict[str, Any]:
        """
        Fetch all market intelligence data for agent analysis.
        
        Args:
            currency_pair: Currency pair to analyze
            amount: Conversion amount for cost calculations
            timeframe_days: Analysis timeframe
            
        Returns:
            Comprehensive market intelligence data
        """
        logger.info(f"Fetching real market intelligence for {currency_pair}")
        
        # Extract currencies for economic calendar
        base_currency, quote_currency = currency_pair.split('/')
        target_currencies = [base_currency, quote_currency]
        
        # Fetch all data sources concurrently for performance
        results = await asyncio.gather(
            self._get_cached_or_fetch('news', 
                lambda: fetch_real_news_data(currency_pair, 48, self.newsapi_key)),
            self._get_cached_or_fetch('calendar',
                lambda: fetch_real_economic_calendar(target_currencies, timeframe_days, self.fred_api_key)),
            self._get_cached_or_fetch('market',
                lambda: fetch_real_market_data(currency_pair, 30)),
            self._get_cached_or_fetch('providers',
                lambda: fetch_real_provider_rates(currency_pair, amount)),
            return_exceptions=True
        )
        
        news_data, calendar_data, market_data, provider_data = results
        
        # Handle any fetch failures gracefully
        if isinstance(news_data, Exception):
            logger.error(f"News data fetch failed: {news_data}")
            news_data = []
        
        if isinstance(calendar_data, Exception):
            logger.error(f"Calendar data fetch failed: {calendar_data}")
            calendar_data = []
        
        if isinstance(market_data, Exception):
            logger.error(f"Market data fetch failed: {market_data}")
            market_data = {}
        
        if isinstance(provider_data, Exception):
            logger.error(f"Provider data fetch failed: {provider_data}")
            provider_data = []
        
        # Compile comprehensive intelligence
        intelligence = {
            "currency_pair": currency_pair,
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat(),
            "data_sources": {
                "news": {
                    "articles": news_data,
                    "count": len(news_data) if isinstance(news_data, list) else 0,
                    "source": "rss_feeds" + ("+newsapi" if self.newsapi_key else "")
                },
                "economic_calendar": {
                    "events": calendar_data,
                    "count": len(calendar_data) if isinstance(calendar_data, list) else 0,
                    "currencies": target_currencies,
                    "source": "fred" + ("+ecb+boe" if self.fred_api_key else "_mock")
                },
                "market_data": {
                    "price_data": market_data.get('price_data', []),
                    "technical_indicators": market_data.get('technical_indicators', {}),
                    "market_regime": market_data.get('market_regime', {}),
                    "data_quality": market_data.get('data_quality', {}),
                    "source": "yahoo_finance"
                },
                "provider_rates": {
                    "providers": provider_data,
                    "count": len(provider_data) if isinstance(provider_data, list) else 0,
                    "source": "estimated_from_public_info"
                }
            },
            "data_quality_score": self._calculate_data_quality_score(news_data, calendar_data, market_data, provider_data)
        }
        
        logger.info(f"Market intelligence compiled: {intelligence['data_quality_score']:.2f} quality score")
        return intelligence
    
    async def _get_cached_or_fetch(self, cache_key: str, fetch_func) -> Any:
        """Get data from cache or fetch if expired."""
        cache_entry = self.cache.get(cache_key)
        
        if cache_entry:
            data, timestamp = cache_entry
            ttl = self.cache_ttl.get(cache_key, 300)
            
            if (datetime.utcnow() - timestamp).total_seconds() < ttl:
                logger.debug(f"Using cached data for {cache_key}")
                return data
        
        # Fetch new data
        logger.debug(f"Fetching fresh data for {cache_key}")
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                data = await fetch_func()
            else:
                data = await fetch_func()
            
            # Cache the result
            self.cache[cache_key] = (data, datetime.utcnow())
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {cache_key}: {e}")
            # Return cached data even if expired, or empty data
            if cache_entry:
                return cache_entry[0]
            return [] if cache_key in ['news', 'calendar', 'providers'] else {}
    
    def _calculate_data_quality_score(self, 
                                    news_data: Any,
                                    calendar_data: Any, 
                                    market_data: Any,
                                    provider_data: Any) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # News quality
        if isinstance(news_data, list) and len(news_data) > 0:
            news_score = min(1.0, len(news_data) / 10.0)  # Up to 10 articles = 1.0
            scores.append(news_score)
        else:
            scores.append(0.0)
        
        # Calendar quality  
        if isinstance(calendar_data, list) and len(calendar_data) > 0:
            calendar_score = min(1.0, len(calendar_data) / 5.0)  # Up to 5 events = 1.0
            scores.append(calendar_score)
        else:
            scores.append(0.0)
        
        # Market data quality
        if isinstance(market_data, dict) and market_data.get('data_points', 0) > 0:
            market_score = market_data.get('data_quality', {}).get('completeness', 0.0)
            scores.append(market_score)
        else:
            scores.append(0.0)
        
        # Provider data quality
        if isinstance(provider_data, list) and len(provider_data) > 0:
            provider_score = min(1.0, len(provider_data) / 5.0)  # Up to 5 providers = 1.0
            scores.append(provider_score)
        else:
            scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status for monitoring."""
        status = {}
        
        for key in ['news', 'calendar', 'market', 'providers']:
            cache_entry = self.cache.get(key)
            if cache_entry:
                _, timestamp = cache_entry
                age_seconds = (datetime.utcnow() - timestamp).total_seconds()
                ttl = self.cache_ttl.get(key, 300)
                
                status[key] = {
                    "cached": True,
                    "age_seconds": age_seconds,
                    "ttl_seconds": ttl,
                    "expired": age_seconds > ttl
                }
            else:
                status[key] = {"cached": False}
        
        return status


# Integration with existing agent tools
def replace_mock_data_in_market_tools():
    """
    Function to patch market_tools.py to use real data instead of mock data.
    This will be called to upgrade the agents to use real data.
    """
    
    async def real_comprehensive_market_analysis(currency_pair: str,
                                               amount: float = 10000.0,
                                               timeframe_days: int = 7) -> Dict[str, Any]:
        """Replace the mock comprehensive_market_analysis with real data version."""
        
        # Initialize real data provider
        newsapi_key = os.getenv('NEWSAPI_KEY')
        fred_key = os.getenv('FRED_API_KEY') 
        
        data_provider = RealDataProvider(newsapi_key, fred_key)
        
        # Get real market intelligence
        intelligence = await data_provider.get_comprehensive_market_intelligence(
            currency_pair, amount, timeframe_days
        )
        
        # Format for existing agent consumption
        return {
            "currency_pair": currency_pair,
            "analysis_timestamp": intelligence["timestamp"],
            "overall_confidence": intelligence["data_quality_score"],
            "components": {
                "news": {
                    "total_items": intelligence["data_sources"]["news"]["count"],
                    "recent_items": intelligence["data_sources"]["news"]["articles"][:5],
                    "confidence": min(1.0, intelligence["data_sources"]["news"]["count"] / 10.0)
                },
                "economic_events": {
                    "total_events": intelligence["data_sources"]["economic_calendar"]["count"],
                    "events_by_currency": {
                        curr: [e for e in intelligence["data_sources"]["economic_calendar"]["events"] 
                              if e.get("currency") == curr]
                        for curr in currency_pair.split('/')
                    },
                    "confidence": min(1.0, intelligence["data_sources"]["economic_calendar"]["count"] / 5.0)
                },
                "market_regime": intelligence["data_sources"]["market_data"]["market_regime"],
                "rsi": {"value": intelligence["data_sources"]["market_data"]["technical_indicators"].get("rsi", 50)},
                "moving_averages": {
                    "signal": "bullish" if (
                        intelligence["data_sources"]["market_data"]["technical_indicators"].get("ma_20", 0) >
                        intelligence["data_sources"]["market_data"]["technical_indicators"].get("ma_50", 0)
                    ) else "bearish"
                }
            }
        }
    
    return real_comprehensive_market_analysis


# Convenience function for agents
async def get_real_market_intelligence_for_agents(currency_pair: str,
                                                 amount: float,
                                                 timeframe_days: int = 7) -> Dict[str, Any]:
    """
    Get real market intelligence formatted for agent consumption.
    
    This function replaces the mock data generators in agent tools.
    """
    # Get API keys from environment
    newsapi_key = os.getenv('NEWSAPI_KEY')
    fred_key = os.getenv('FRED_API_KEY')
    
    # Initialize provider
    provider = RealDataProvider(newsapi_key, fred_key)
    
    # Fetch comprehensive intelligence
    intelligence = await provider.get_comprehensive_market_intelligence(
        currency_pair, amount, timeframe_days
    )
    
    return {
        "news_data": intelligence["data_sources"]["news"]["articles"],
        "calendar_data": intelligence["data_sources"]["economic_calendar"]["events"],
        "price_data": intelligence["data_sources"]["market_data"]["price_data"],
        "provider_data": intelligence["data_sources"]["provider_rates"]["providers"],
        "market_context": {
            "regime": intelligence["data_sources"]["market_data"]["market_regime"],
            "technical_indicators": intelligence["data_sources"]["market_data"]["technical_indicators"],
            "data_quality": intelligence["data_quality_score"]
        }
    }


if __name__ == "__main__":
    # Test complete data integration
    async def test_data_integration():
        print("🌐 Testing Complete Real Data Integration...")
        print("=" * 50)
        
        # Test comprehensive intelligence
        provider = RealDataProvider()
        intelligence = await provider.get_comprehensive_market_intelligence("USD/EUR", 15000.0, 7)
        
        print(f"Data Quality Score: {intelligence['data_quality_score']:.2f}")
        print(f"News Articles: {intelligence['data_sources']['news']['count']}")
        print(f"Economic Events: {intelligence['data_sources']['economic_calendar']['count']}")  
        print(f"Market Data Points: {len(intelligence['data_sources']['market_data']['price_data'])}")
        print(f"Provider Options: {intelligence['data_sources']['provider_rates']['count']}")
        
        # Test agent-formatted data
        agent_data = await get_real_market_intelligence_for_agents("USD/EUR", 15000.0)
        print(f"\nAgent Data Format:")
        print(f"  News items: {len(agent_data['news_data'])}")
        print(f"  Calendar events: {len(agent_data['calendar_data'])}")
        print(f"  Price data points: {len(agent_data['price_data'])}")
        print(f"  Provider options: {len(agent_data['provider_data'])}")
        print(f"  Data quality: {agent_data['market_context']['data_quality']:.2f}")
    
    asyncio.run(test_data_integration())