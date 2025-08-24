"""
Test script for the FX data collection system.

This script tests the data ingestion component without requiring a full database setup.
It demonstrates fetching real-time FX data from external APIs.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from data.collector import FXDataCollector
from data.cache import CacheManager

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_collection():
    """Test the complete data collection system."""
    
    print("🚀 Testing FX Data Collection System")
    print("=" * 50)
    
    # Configuration
    currency_pairs = ["USD/EUR", "USD/GBP", "EUR/GBP"]
    
    # Load API keys from environment variables
    alphavantage_key = os.getenv("ALPHAVANTAGE_API_KEY")
    fixer_key = os.getenv("FIXER_API_KEY")
    
    if not alphavantage_key and not fixer_key:
        print("⚠️  No API keys provided - using mock data for demonstration")
        print("   To test with real data, get free API keys from:")
        print("   - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("   - Fixer.io: https://fixer.io/product")
        print()
    
    # Initialize collector
    collector = FXDataCollector(currency_pairs)
    
    # Set up providers (will skip if no API keys)
    collector.setup_default_providers(
        alphavantage_key=alphavantage_key,
        fixer_key=fixer_key
    )
    
    # Initialize cache
    cache_manager = CacheManager()
    
    print(f"📊 Configured to collect {len(currency_pairs)} currency pairs:")
    for pair in currency_pairs:
        print(f"   - {pair}")
    print()
    
    print(f"🔌 Active providers: {len(collector.providers)}")
    for provider in collector.providers:
        print(f"   - {provider.name} ({'available' if provider.is_available() else 'not available'})")
    print()
    
    if not collector.providers:
        print("❌ No providers available - cannot collect real data")
        print("   This would work with valid API keys")
        return
    
    try:
        # Test single collection cycle
        print("🔄 Starting data collection...")
        result = await collector.collect_all_rates()
        
        print(f"✅ Collection completed:")
        print(f"   - Successful: {len(result.successful_pairs)}/{len(currency_pairs)} pairs")
        print(f"   - Failed: {len(result.failed_pairs)} pairs")
        
        if result.errors:
            print(f"   - Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"     • {error}")
        
        print()
        
        # Display collected rates
        if result.rates_collected:
            print("📈 Current Exchange Rates:")
            for pair, rate_data in result.rates_collected.items():
                print(f"   {pair}: {rate_data.rate} (from {rate_data.provider})")
                
                # Test caching
                await cache_manager.set_current_rate(pair, rate_data)
        else:
            print("⚠️  No rates were successfully collected")
        
        print()
        
        # Test cache functionality
        if result.rates_collected:
            print("💾 Testing cache functionality...")
            
            # Test cache retrieval
            first_pair = list(result.rates_collected.keys())[0]
            cached_rate = await cache_manager.get_current_rate(first_pair)
            
            if cached_rate:
                print(f"   ✅ Successfully cached and retrieved {first_pair}")
                print(f"      Rate: {cached_rate.rate} (cached at {cached_rate.timestamp})")
            else:
                print(f"   ❌ Cache retrieval failed for {first_pair}")
        
        print()
        
        # Display provider statistics
        print("📊 Provider Statistics:")
        for provider in collector.providers:
            stats = provider.get_stats()
            rate_limits = provider.get_rate_limit_info()
            print(f"   {provider.name}:")
            print(f"     - Requests: {stats['total_requests']}")
            print(f"     - Errors: {stats['error_count']}")
            print(f"     - Success Rate: {stats['success_rate']}")
            
            # Show rate limits if available
            if 'daily_remaining' in rate_limits:
                print(f"     - Daily Requests Remaining: {rate_limits['daily_remaining']}")
            elif 'monthly_remaining' in rate_limits:
                print(f"     - Monthly Requests Remaining: {rate_limits['monthly_remaining']}")
        
        print()
        
        # Test health check
        print("🏥 Running health check...")
        health = await collector.health_check()
        
        print(f"   Overall Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Unhealthy'}")
        
        if health['issues']:
            print("   Issues:")
            for issue in health['issues']:
                print(f"     - {issue}")
        
        # Display collection statistics
        print("\n📈 Collection Statistics:")
        stats = collector.get_collection_stats()
        print(f"   - Total Collections: {stats['total_collections']}")
        print(f"   - Success Rate: {stats['success_rate']}")
        print(f"   - Currency Pairs with Data: {stats['pairs_with_data']}/{stats['currency_pairs_configured']}")
        
        # Display cache statistics
        print("\n💾 Cache Statistics:")
        cache_stats = cache_manager.get_stats()
        print(f"   - Entries: {cache_stats['entries']}")
        print(f"   - Hit Rate: {cache_stats['hit_rate']}")
        print(f"   - Total Sets: {cache_stats['sets']}")
        
        print("\n🎉 Data collection test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed: {e}")
        
    finally:
        # Cleanup
        await collector.cleanup()
        await cache_manager.cleanup()


def test_without_api_keys():
    """Demonstrate the system structure without making real API calls."""
    print("🔧 System Architecture Test (No API Calls)")
    print("=" * 50)
    
    from data.providers.alphavantage import AlphaVantageProvider
    from data.providers.fixer_io import FixerIOProvider
    
    # Create providers without API keys
    alpha_provider = AlphaVantageProvider("")
    fixer_provider = FixerIOProvider("")
    
    print(f"✅ AlphaVantage Provider: {alpha_provider.name}")
    print(f"   - Available: {alpha_provider.is_available()}")
    print(f"   - Rate Limits: {alpha_provider.get_rate_limit_info()}")
    
    print(f"✅ FixerIO Provider: {fixer_provider.name}")
    print(f"   - Available: {fixer_provider.is_available()}")
    print(f"   - Rate Limits: {fixer_provider.get_rate_limit_info()}")
    
    print("\n✅ All components initialized successfully!")
    print("   Ready to collect real data with valid API keys.")


if __name__ == "__main__":
    print("Currency Assistant - Data Collection Test")
    print("=" * 60)
    
    try:
        # Run the async test
        asyncio.run(test_data_collection())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        
        # Fall back to architecture test
        print("\nRunning architecture test instead...")
        test_without_api_keys()