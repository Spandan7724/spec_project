"""
Comprehensive test script for exchange rate collection system.
"""
import asyncio
import logging
from datetime import datetime
from src.data_collection.rate_collector import MultiProviderRateCollector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_single_pair(collector: MultiProviderRateCollector, 
                          base: str, quote: str) -> None:
    """Test fetching rates for a single currency pair."""
    print(f"\n🔄 Testing {base}/{quote}")
    print("=" * 40)
    
    try:
        result = await collector.get_rate(base, quote)
        
        print(f"Success Rate: {result.success_rate:.1f}%")
        print(f"Rates Retrieved: {len(result.rates)}")
        
        if result.has_data:
            print(f"Best Rate: {result.best_rate.rate:.4f} ({result.best_rate.source.value})")
            
            print("\nAll Rates:")
            for rate in result.rates:
                bid_ask = ""
                if rate.bid and rate.ask:
                    bid_ask = f" (bid: {rate.bid:.4f}, ask: {rate.ask:.4f}, spread: {rate.spread_bps}bps)"
                
                print(f"  • {rate.source.value}: {rate.rate:.4f}{bid_ask}")
                print(f"    Timestamp: {rate.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("❌ No rates retrieved!")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def test_provider_connectivity(collector: MultiProviderRateCollector) -> None:
    """Test connectivity to all providers."""
    print("\n🔌 Testing Provider Connectivity")
    print("=" * 40)
    
    results = await collector.test_all_providers()
    
    for provider_name, success in results.items():
        status = "✅ OK" if success else "❌ Failed"
        print(f"  {provider_name}: {status}")


async def test_multiple_pairs(collector: MultiProviderRateCollector) -> None:
    """Test multiple currency pairs."""
    pairs = [
        ("USD", "EUR"),
        ("USD", "GBP"), 
        ("EUR", "GBP"),
        ("USD", "JPY"),
        ("GBP", "USD"),  # Test reverse pair
    ]
    
    print(f"\n📊 Testing {len(pairs)} Currency Pairs")
    print("=" * 50)
    
    results_summary = []
    
    for base, quote in pairs:
        try:
            result = await collector.get_rate(base, quote)
            results_summary.append({
                'pair': f"{base}/{quote}",
                'success_rate': result.success_rate,
                'rate_count': len(result.rates),
                'best_rate': result.best_rate.rate if result.best_rate else None,
                'has_data': result.has_data
            })
            
            # Brief delay to respect rate limits
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to test {base}/{quote}: {e}")
            results_summary.append({
                'pair': f"{base}/{quote}",
                'success_rate': 0,
                'rate_count': 0,
                'best_rate': None,
                'has_data': False
            })
    
    # Print summary
    print("\nSummary Results:")
    print("-" * 60)
    print(f"{'Pair':<10} {'Success%':<10} {'Sources':<8} {'Rate':<12} {'Status'}")
    print("-" * 60)
    
    for result in results_summary:
        status = "✅" if result['has_data'] else "❌"
        rate_str = f"{result['best_rate']:.4f}" if result['best_rate'] else "N/A"
        
        print(f"{result['pair']:<10} {result['success_rate']:<10.1f} "
              f"{result['rate_count']:<8} {rate_str:<12} {status}")


async def test_provider_info(collector: MultiProviderRateCollector) -> None:
    """Test provider information retrieval."""
    print("\n📋 Provider Information")
    print("=" * 40)
    
    providers = collector.get_provider_info()
    
    for provider in providers:
        print(f"\n{provider['name']}:")
        print(f"  URL: {provider['base_url']}")
        print(f"  API Key: {'✅ Present' if provider['has_api_key'] else '❌ Missing'}")
        print(f"  Rate Limit: {provider['rate_limit']} requests/minute")


async def main():
    """Main test function."""
    print("🧪 Exchange Rate Collection System Test")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize collector
        print("\n🚀 Initializing Rate Collector...")
        collector = MultiProviderRateCollector()
        
        # Test provider info
        await test_provider_info(collector)
        
        # Test connectivity
        await test_provider_connectivity(collector)
        
        # Test single pair in detail
        await test_single_pair(collector, "USD", "EUR")
        
        # Test multiple pairs
        await test_multiple_pairs(collector)
        
        print(f"\n✅ All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.exception("Test suite error")


if __name__ == "__main__":
    asyncio.run(main())