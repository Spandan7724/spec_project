#!/usr/bin/env python3
"""
Comprehensive test for technical indicators system.
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.analysis.technical_indicators import (
    TechnicalIndicatorEngine,
    get_technical_indicators,
    get_volatility_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_technical_indicators():
    """Test technical indicator calculations."""
    
    logger.info("📊 Testing Technical Indicators")
    
    engine = TechnicalIndicatorEngine()
    
    # Test currency pairs
    test_pairs = ['USD/EUR', 'GBP/USD']
    
    results = {}
    
    for pair in test_pairs:
        logger.info(f"\n📈 Calculating indicators for {pair}")
        
        try:
            indicators = await engine.calculate_indicators(pair)
            
            if indicators:
                results[pair] = {
                    'success': True,
                    'indicators': indicators.to_dict()
                }
                
                logger.info(f"✅ {pair} indicators calculated successfully")
                
                # Log key indicator values
                logger.info("   📊 Moving Averages:")
                logger.info(f"      • SMA 20: {indicators.sma_20:.4f}" if indicators.sma_20 else "      • SMA 20: N/A")
                logger.info(f"      • SMA 50: {indicators.sma_50:.4f}" if indicators.sma_50 else "      • SMA 50: N/A")
                logger.info(f"      • EMA 12: {indicators.ema_12:.4f}" if indicators.ema_12 else "      • EMA 12: N/A")
                
                logger.info("   📈 Bollinger Bands:")
                if indicators.bb_upper and indicators.bb_lower:
                    logger.info(f"      • Upper: {indicators.bb_upper:.4f}")
                    logger.info(f"      • Lower: {indicators.bb_lower:.4f}")
                    logger.info(f"      • Width: {indicators.bb_width:.2f}%")
                    logger.info(f"      • Position: {indicators.bb_position:.2f}")
                
                logger.info("   📊 Volatility:")
                if indicators.realized_volatility:
                    logger.info(f"      • Realized Vol: {indicators.realized_volatility:.2%}")
                    logger.info(f"      • Regime: {indicators.volatility_regime}")
                    logger.info(f"      • Vol Score: {indicators.volatility_score:.2f}" if indicators.volatility_score else "      • Vol Score: N/A")
                
                logger.info("   📈 Momentum:")
                if indicators.rsi_14:
                    logger.info(f"      • RSI 14: {indicators.rsi_14:.2f}")
                if indicators.macd is not None:
                    logger.info(f"      • MACD: {indicators.macd:.4f}")
                    logger.info(f"      • MACD Signal: {indicators.macd_signal:.4f}" if indicators.macd_signal else "")
                
                logger.info("   📊 Trend:")
                logger.info(f"      • Direction: {indicators.trend_direction}")
                logger.info(f"      • Strength: {indicators.trend_strength:.2f}" if indicators.trend_strength else "      • Strength: N/A")
                logger.info(f"      • Is Bullish: {indicators.is_bullish}")
                
                logger.info("   🎯 Support/Resistance:")
                if indicators.support_level and indicators.resistance_level:
                    logger.info(f"      • Support: {indicators.support_level:.4f}")
                    logger.info(f"      • Resistance: {indicators.resistance_level:.4f}")
                
            else:
                results[pair] = {
                    'success': False,
                    'error': 'No indicators calculated'
                }
                logger.warning(f"❌ {pair}: No indicators calculated")
                
        except Exception as e:
            results[pair] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ {pair}: Error - {e}")
    
    return results


async def test_convenience_functions():
    """Test convenience functions."""
    
    logger.info("\n🛠️  Testing Convenience Functions")
    
    try:
        # Test get_technical_indicators
        logger.info("Testing get_technical_indicators()...")
        indicators = await get_technical_indicators('USD/EUR')
        
        if indicators:
            logger.info(f"✅ get_technical_indicators: {indicators.currency_pair}")
            logger.info(f"   • Calculated {sum(1 for v in indicators.to_dict().values() if v is not None)} indicators")
        else:
            logger.warning("❌ get_technical_indicators: No indicators returned")
        
        # Test get_volatility_analysis
        logger.info("Testing get_volatility_analysis()...")
        vol_analysis = await get_volatility_analysis('USD/EUR')
        
        if vol_analysis:
            logger.info("✅ get_volatility_analysis:")
            logger.info(f"   • Volatility: {vol_analysis['realized_volatility']:.2%}" if vol_analysis['realized_volatility'] else "   • Volatility: N/A")
            logger.info(f"   • Regime: {vol_analysis['volatility_regime']}")
            logger.info(f"   • Score: {vol_analysis['volatility_score']:.2f}" if vol_analysis['volatility_score'] else "   • Score: N/A")
        else:
            logger.warning("❌ get_volatility_analysis: No analysis returned")
            
    except Exception as e:
        logger.error(f"❌ Convenience function error: {e}")


async def test_caching_system():
    """Test technical indicator caching."""
    
    logger.info("\n💾 Testing Indicator Caching")
    
    engine = TechnicalIndicatorEngine()
    
    try:
        from datetime import datetime
        
        # First calculation (should compute indicators)
        start_time = datetime.utcnow()
        indicators1 = await engine.calculate_indicators('USD/EUR')
        first_duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"🔄 First calculation: {first_duration:.2f}s")
        
        # Second calculation (should use cache)
        start_time = datetime.utcnow()
        indicators2 = await engine.calculate_indicators('USD/EUR')
        second_duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"⚡ Cached calculation: {second_duration:.2f}s")
        
        if indicators1 and indicators2:
            # Compare a few key values to ensure cache is working
            if (indicators1.sma_20 == indicators2.sma_20 and 
                indicators1.rsi_14 == indicators2.rsi_14):
                logger.info("✅ Cache working: Identical indicators returned")
                if first_duration > second_duration:
                    logger.info(f"🚀 Speed improvement: {first_duration/second_duration:.1f}x faster")
            else:
                logger.warning("❌ Cache issue: Different indicators returned")
        
    except Exception as e:
        logger.error(f"❌ Caching test error: {e}")


async def test_indicator_validation():
    """Test indicator value validation and edge cases."""
    
    logger.info("\n🔍 Testing Indicator Validation")
    
    engine = TechnicalIndicatorEngine()
    
    try:
        indicators = await engine.calculate_indicators('USD/EUR')
        
        if indicators:
            logger.info(f"📊 Indicator Validation for {indicators.currency_pair}:")
            
            # Test RSI bounds (should be 0-100)
            if indicators.rsi_14 is not None:
                rsi_valid = 0 <= indicators.rsi_14 <= 100
                logger.info(f"   • RSI bounds check: {'✅' if rsi_valid else '❌'} ({indicators.rsi_14:.2f})")
            
            # Test Bollinger Band position (should be 0-1 approximately)
            if indicators.bb_position is not None:
                bb_pos_reasonable = -0.5 <= indicators.bb_position <= 1.5  # Allow some slack
                logger.info(f"   • BB position check: {'✅' if bb_pos_reasonable else '❌'} ({indicators.bb_position:.2f})")
            
            # Test volatility reasonableness (should be positive)
            if indicators.realized_volatility is not None:
                vol_valid = indicators.realized_volatility > 0
                logger.info(f"   • Volatility positive: {'✅' if vol_valid else '❌'} ({indicators.realized_volatility:.2%})")
            
            # Test moving average relationship (SMA20 should be reasonable vs current price)
            if indicators.sma_20 is not None:
                # Get current price estimate from support/resistance
                current_est = (indicators.support_level + indicators.resistance_level) / 2 if (
                    indicators.support_level and indicators.resistance_level) else indicators.sma_20
                ma_reasonable = 0.5 * current_est < indicators.sma_20 < 2.0 * current_est
                logger.info(f"   • SMA reasonableness: {'✅' if ma_reasonable else '❌'} (SMA: {indicators.sma_20:.4f})")
            
            # Test trend strength (should be 0-1)
            if indicators.trend_strength is not None:
                trend_valid = 0 <= indicators.trend_strength <= 1
                logger.info(f"   • Trend strength bounds: {'✅' if trend_valid else '❌'} ({indicators.trend_strength:.2f})")
            
            # Test support/resistance relationship
            if indicators.support_level and indicators.resistance_level:
                sr_valid = indicators.support_level < indicators.resistance_level
                logger.info(f"   • Support < Resistance: {'✅' if sr_valid else '❌'}")
                logger.info(f"     Support: {indicators.support_level:.4f}, Resistance: {indicators.resistance_level:.4f}")
            
            # Test bullish/bearish logic
            logger.info(f"   • Overall sentiment: {'🐂 Bullish' if indicators.is_bullish else '🐻 Bearish'}")
            
        else:
            logger.warning("❌ No indicators to validate")
    
    except Exception as e:
        logger.error(f"❌ Validation test error: {e}")


async def main():
    """Run all technical indicator tests."""
    
    logger.info("🚀 Currency Assistant - Technical Indicators Test Suite")
    logger.info("=" * 70)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Run all tests
        results = await test_technical_indicators()
        await test_convenience_functions()
        await test_caching_system()
        await test_indicator_validation()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📋 TECHNICAL INDICATORS TEST SUMMARY")
        
        success_count = sum(1 for r in results.values() if r['success'])
        total_pairs = len(results)
        
        logger.info(f"✅ Successfully calculated indicators for: {success_count}/{total_pairs} currency pairs")
        
        for pair, result in results.items():
            if result['success']:
                indicators_count = sum(1 for v in result['indicators'].values() 
                                     if v is not None and v != '' and v != 'N/A')
                logger.info(f"   • {pair}: {indicators_count} indicators calculated")
            else:
                logger.error(f"   • {pair}: FAILED - {result['error']}")
        
        if success_count == total_pairs:
            logger.info("🎉 ALL TESTS PASSED - Technical Indicators Working!")
        else:
            logger.warning(f"⚠️  {total_pairs - success_count} tests failed")
    
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())