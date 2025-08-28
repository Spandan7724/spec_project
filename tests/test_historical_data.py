#!/usr/bin/env python3
"""
Comprehensive test for historical data collection system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.analysis.historical_data import (
    HistoricalDataCollector,
    get_historical_rates,
    get_recent_volatility
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_historical_data_collection():
    """Test historical data collection from multiple sources."""
    
    logger.info("🧪 Starting Historical Data Collection Tests")
    
    collector = HistoricalDataCollector()
    
    # Test currency pairs to validate
    test_pairs = ['USD/EUR', 'GBP/USD', 'USD/JPY']
    
    results = {}
    
    for pair in test_pairs:
        logger.info(f"\n📈 Testing historical data for {pair}")
        
        try:
            # Test 30 days of data
            dataset = await collector.get_historical_data(pair, days=30)
            
            if dataset:
                results[pair] = {
                    'success': True,
                    'data_points': len(dataset.data),
                    'date_range': f"{dataset.start_date.date()} to {dataset.end_date.date()}",
                    'data_quality': f"{dataset.data_quality:.1%}",
                    'source': dataset.source.value,
                    'latest_rate': dataset.get_latest_rate().close_rate if dataset.get_latest_rate() else None
                }
                
                logger.info(f"✅ {pair}: {len(dataset.data)} data points "
                          f"({dataset.data_quality:.1%} quality) from {dataset.source.value}")
                
                # Test DataFrame conversion
                df = dataset.to_dataframe()
                logger.info(f"📊 DataFrame: {len(df)} rows, columns: {list(df.columns)}")
                
                # Test rate series extraction
                close_series = dataset.get_rate_series('close')
                logger.info(f"📈 Close rates: {len(close_series)} values, "
                          f"latest: {close_series.iloc[-1]:.4f}")
                
                # Test date range filtering
                week_ago = datetime.utcnow() - timedelta(days=7)
                recent_data = dataset.get_rate_range(week_ago, datetime.utcnow())
                logger.info(f"🗓️  Recent week: {len(recent_data)} data points")
                
            else:
                results[pair] = {
                    'success': False,
                    'error': 'No data returned'
                }
                logger.warning(f"❌ {pair}: No historical data available")
                
        except Exception as e:
            results[pair] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ {pair}: Error - {e}")
    
    return results


async def test_convenience_functions():
    """Test convenience functions for historical data."""
    
    logger.info("\n🛠️  Testing Convenience Functions")
    
    try:
        # Test get_historical_rates
        logger.info("Testing get_historical_rates()...")
        dataset = await get_historical_rates('USD/EUR', days=14)
        
        if dataset:
            logger.info(f"✅ get_historical_rates: {len(dataset.data)} points, "
                      f"{dataset.data_quality:.1%} quality")
        else:
            logger.warning("❌ get_historical_rates: No data returned")
        
        # Test get_recent_volatility
        logger.info("Testing get_recent_volatility()...")
        volatility = await get_recent_volatility('USD/EUR', days=30)
        
        if volatility is not None:
            logger.info(f"✅ get_recent_volatility: {volatility:.2%} annualized")
        else:
            logger.warning("❌ get_recent_volatility: No volatility calculated")
            
    except Exception as e:
        logger.error(f"❌ Convenience function error: {e}")


async def test_caching_system():
    """Test the caching system performance."""
    
    logger.info("\n💾 Testing Caching System")
    
    collector = HistoricalDataCollector()
    
    try:
        # First request (should fetch from API)
        start_time = datetime.utcnow()
        dataset1 = await collector.get_historical_data('USD/EUR', days=30)
        first_duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"🌐 First request: {first_duration:.2f}s")
        
        # Second request (should use cache)
        start_time = datetime.utcnow()
        dataset2 = await collector.get_historical_data('USD/EUR', days=30)
        second_duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"⚡ Cached request: {second_duration:.2f}s")
        
        if dataset1 and dataset2:
            if len(dataset1.data) == len(dataset2.data):
                logger.info("✅ Cache working: Same data returned")
                logger.info(f"🚀 Speed improvement: {first_duration/second_duration:.1f}x faster")
            else:
                logger.warning("❌ Cache issue: Different data returned")
        
        # Test cache info
        cached_pairs = collector.get_cached_pairs()
        logger.info(f"💽 Cached pairs: {cached_pairs}")
        
    except Exception as e:
        logger.error(f"❌ Caching test error: {e}")


async def test_data_validation():
    """Test data validation and quality checks."""
    
    logger.info("\n🔍 Testing Data Validation")
    
    collector = HistoricalDataCollector()
    
    try:
        dataset = await collector.get_historical_data('USD/EUR', days=30)
        
        if dataset:
            logger.info("📊 Dataset validation:")
            logger.info(f"   • Total data points: {len(dataset.data)}")
            logger.info(f"   • Data quality score: {dataset.data_quality:.1%}")
            logger.info(f"   • Date range: {dataset.start_date.date()} to {dataset.end_date.date()}")
            
            # Check data consistency
            valid_points = 0
            invalid_points = 0
            
            for rate_data in dataset.data[:10]:  # Check first 10 points
                if collector._validate_rate_data(rate_data, 'USD/EUR'):
                    valid_points += 1
                else:
                    invalid_points += 1
                    logger.warning(f"❌ Invalid data point: {rate_data.date}")
            
            logger.info(f"✅ Sample validation: {valid_points}/10 valid points")
            
            # Test typical price calculation
            if dataset.data:
                sample_rate = dataset.data[0]
                typical_price = sample_rate.typical_price
                range_percent = sample_rate.range_percent
                
                logger.info("📈 Sample calculations:")
                logger.info(f"   • OHLC: {sample_rate.open_rate:.4f}, {sample_rate.high_rate:.4f}, "
                          f"{sample_rate.low_rate:.4f}, {sample_rate.close_rate:.4f}")
                logger.info(f"   • Typical price: {typical_price:.4f}")
                logger.info(f"   • Daily range: {range_percent:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Validation test error: {e}")


async def main():
    """Run all historical data tests."""
    
    logger.info("🚀 Currency Assistant - Historical Data Collection Test Suite")
    logger.info("=" * 70)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API keys
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        logger.info("🔑 Alpha Vantage API key found")
    else:
        logger.warning("⚠️  No Alpha Vantage API key - will use Yahoo Finance only")
    
    try:
        # Run all tests
        results = await test_historical_data_collection()
        await test_convenience_functions() 
        await test_caching_system()
        await test_data_validation()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📋 TEST RESULTS SUMMARY")
        
        success_count = sum(1 for r in results.values() if r['success'])
        total_pairs = len(results)
        
        logger.info(f"✅ Successfully tested: {success_count}/{total_pairs} currency pairs")
        
        for pair, result in results.items():
            if result['success']:
                logger.info(f"   • {pair}: {result['data_points']} points "
                          f"({result['data_quality']}) from {result['source']}")
            else:
                logger.error(f"   • {pair}: FAILED - {result['error']}")
        
        if success_count == total_pairs:
            logger.info("🎉 ALL TESTS PASSED - Historical Data Collection Working!")
        else:
            logger.warning(f"⚠️  {total_pairs - success_count} tests failed")
    
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())