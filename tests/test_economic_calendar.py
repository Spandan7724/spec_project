#!/usr/bin/env python3
"""
Comprehensive test for economic calendar integration.
"""

import asyncio
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.economic.calendar_collector import (
    EconomicCalendarCollector,
    EventImpact,
    get_upcoming_events,
    get_events_by_impact
)

from data_collection.economic.fred_provider import FREDProvider
from data_collection.economic.ecb_provider import ECBProvider
from data_collection.economic.boe_provider import BOEProvider
from data_collection.economic.rbi_provider import RBIProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_individual_providers():
    """Test each economic data provider individually."""
    
    logger.info("📊 Testing Individual Economic Data Providers")
    
    results = {}
    
    # Test FRED Provider (US)
    logger.info("\n🇺🇸 Testing FRED Provider (US Economic Data)")
    try:
        fred_api_key = os.getenv('FRED_API_KEY')
        if fred_api_key:
            async with FREDProvider(fred_api_key) as fred_provider:
                fred_events = await fred_provider.get_upcoming_releases(days_ahead=21)
                
                if fred_events:
                    results['FRED'] = {
                        'success': True,
                        'event_count': len(fred_events),
                        'sample_events': [e.to_dict() for e in fred_events[:3]]
                    }
                    logger.info(f"✅ FRED: {len(fred_events)} upcoming US economic events")
                    
                    # Show sample events
                    for event in fred_events[:3]:
                        logger.info(f"   • {event.title}: {event.release_date.date()} "
                                  f"({event.impact.value} impact)")
                else:
                    results['FRED'] = {'success': False, 'error': 'No events returned'}
                    logger.warning("❌ FRED: No events returned")
        else:
            results['FRED'] = {'success': False, 'error': 'No API key'}
            logger.warning("⚠️  FRED: No API key available")
            
    except Exception as e:
        results['FRED'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ FRED: Error - {e}")
    
    # Test ECB Provider (Europe)
    logger.info("\n🇪🇺 Testing ECB Provider (European Economic Data)")
    try:
        async with ECBProvider() as ecb_provider:
            ecb_events = await ecb_provider.get_upcoming_releases(days_ahead=21)
            
            if ecb_events:
                results['ECB'] = {
                    'success': True,
                    'event_count': len(ecb_events),
                    'sample_events': [e.to_dict() for e in ecb_events[:3]]
                }
                logger.info(f"✅ ECB: {len(ecb_events)} upcoming European economic events")
                
                for event in ecb_events[:3]:
                    logger.info(f"   • {event.title}: {event.release_date.date()} "
                              f"({event.impact.value} impact)")
            else:
                results['ECB'] = {'success': False, 'error': 'No events returned'}
                logger.warning("❌ ECB: No events returned")
                
    except Exception as e:
        results['ECB'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ ECB: Error - {e}")
    
    # Test BOE Provider (UK)
    logger.info("\n🇬🇧 Testing BOE Provider (UK Economic Data)")
    try:
        async with BOEProvider() as boe_provider:
            boe_events = await boe_provider.get_upcoming_releases(days_ahead=21)
            
            if boe_events:
                results['BOE'] = {
                    'success': True,
                    'event_count': len(boe_events),
                    'sample_events': [e.to_dict() for e in boe_events[:3]]
                }
                logger.info(f"✅ BOE: {len(boe_events)} upcoming UK economic events")
                
                for event in boe_events[:3]:
                    logger.info(f"   • {event.title}: {event.release_date.date()} "
                              f"({event.impact.value} impact)")
            else:
                results['BOE'] = {'success': False, 'error': 'No events returned'}
                logger.warning("❌ BOE: No events returned")
                
    except Exception as e:
        results['BOE'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ BOE: Error - {e}")
    
    # Test RBI Provider (India)
    logger.info("\n🇮🇳 Testing RBI Provider (Indian Economic Data)")
    try:
        async with RBIProvider() as rbi_provider:
            rbi_events = await rbi_provider.get_upcoming_releases(days_ahead=21)
            
            if rbi_events:
                results['RBI'] = {
                    'success': True,
                    'event_count': len(rbi_events),
                    'sample_events': [e.to_dict() for e in rbi_events[:3]]
                }
                logger.info(f"✅ RBI: {len(rbi_events)} upcoming Indian economic events")
                
                for event in rbi_events[:3]:
                    logger.info(f"   • {event.title}: {event.release_date.date()} "
                              f"({event.impact.value} impact)")
            else:
                results['RBI'] = {'success': False, 'error': 'No events returned'}
                logger.warning("❌ RBI: No events returned")
                
    except Exception as e:
        results['RBI'] = {'success': False, 'error': str(e)}
        logger.error(f"❌ RBI: Error - {e}")
    
    return results


async def test_economic_calendar():
    """Test the integrated economic calendar system."""
    
    logger.info("\n📅 Testing Integrated Economic Calendar")
    
    collector = EconomicCalendarCollector()
    
    try:
        # Get comprehensive calendar
        calendar = await collector.get_economic_calendar(days_ahead=21)
        
        if calendar:
            logger.info("✅ Economic Calendar created successfully")
            logger.info(f"   • Total events: {calendar.event_count}")
            logger.info(f"   • Upcoming events: {calendar.upcoming_count}")
            logger.info(f"   • High impact events: {calendar.high_impact_count}")
            logger.info(f"   • Data sources: {', '.join(calendar.sources)}")
            logger.info(f"   • Period: {calendar.start_date.date()} to {calendar.end_date.date()}")
            
            # Test filtering methods
            logger.info("\n📊 Testing Calendar Filtering:")
            
            # High impact events
            high_impact = calendar.get_events_by_impact(EventImpact.HIGH)
            logger.info(f"   • High impact events: {len(high_impact)}")
            
            # Events by currency
            usd_events = calendar.get_events_by_currency('USD')
            eur_events = calendar.get_events_by_currency('EUR')
            gbp_events = calendar.get_events_by_currency('GBP')
            inr_events = calendar.get_events_by_currency('INR')
            
            logger.info(f"   • USD events: {len(usd_events)}")
            logger.info(f"   • EUR events: {len(eur_events)}")
            logger.info(f"   • GBP events: {len(gbp_events)}")
            logger.info(f"   • INR events: {len(inr_events)}")
            
            # Events for specific currency pairs
            usd_eur_events = calendar.get_events_for_pair('USD/EUR')
            gbp_usd_events = calendar.get_events_for_pair('GBP/USD')
            usd_inr_events = calendar.get_events_for_pair('USD/INR')
            
            logger.info(f"   • USD/EUR relevant events: {len(usd_eur_events)}")
            logger.info(f"   • GBP/USD relevant events: {len(gbp_usd_events)}")
            logger.info(f"   • USD/INR relevant events: {len(usd_inr_events)}")
            
            # Upcoming high impact events
            upcoming_high = calendar.get_high_impact_upcoming(days_ahead=14)
            logger.info(f"   • Upcoming high impact (14 days): {len(upcoming_high)}")
            
            # Show sample high impact events
            if upcoming_high:
                logger.info("\n🎯 Sample High Impact Upcoming Events:")
                for event in upcoming_high[:5]:
                    logger.info(f"   • {event.title} ({event.country})")
                    logger.info(f"     Date: {event.release_date.date()}")
                    logger.info(f"     Impact: {event.impact.value}")
                    if event.previous_value:
                        logger.info(f"     Previous: {event.previous_value}")
                    logger.info(f"     Relevance: {event.relevance_score:.2f}")
            
            return {
                'success': True,
                'total_events': calendar.event_count,
                'upcoming_events': calendar.upcoming_count,
                'high_impact_events': calendar.high_impact_count,
                'sources': calendar.sources
            }
        else:
            logger.warning("❌ Economic Calendar: Failed to create calendar")
            return {'success': False, 'error': 'Calendar creation failed'}
            
    except Exception as e:
        logger.error(f"❌ Economic Calendar: Error - {e}")
        return {'success': False, 'error': str(e)}


async def test_convenience_functions():
    """Test convenience functions for economic events."""
    
    logger.info("\n🛠️  Testing Convenience Functions")
    
    try:
        # Test get_upcoming_events
        logger.info("Testing get_upcoming_events()...")
        upcoming = await get_upcoming_events(days_ahead=14)
        
        if upcoming:
            logger.info(f"✅ get_upcoming_events: {len(upcoming)} events")
            
            # Test with currency pair filter
            usd_eur_upcoming = await get_upcoming_events('USD/EUR', days_ahead=14)
            if usd_eur_upcoming:
                logger.info(f"✅ USD/EUR upcoming events: {len(usd_eur_upcoming)}")
        else:
            logger.warning("❌ get_upcoming_events: No events returned")
        
        # Test get_events_by_impact
        logger.info("Testing get_events_by_impact()...")
        high_impact = await get_events_by_impact(EventImpact.HIGH, days_ahead=21)
        
        if high_impact:
            logger.info(f"✅ get_events_by_impact: {len(high_impact)} high impact events")
        else:
            logger.warning("❌ get_events_by_impact: No events returned")
            
    except Exception as e:
        logger.error(f"❌ Convenience function error: {e}")


async def test_event_validation():
    """Test economic event data validation and properties."""
    
    logger.info("\n🔍 Testing Event Data Validation")
    
    collector = EconomicCalendarCollector()
    
    try:
        calendar = await collector.get_economic_calendar(days_ahead=14)
        
        if calendar and calendar.events:
            logger.info(f"📊 Validating {len(calendar.events)} events:")
            
            valid_events = 0
            invalid_events = 0
            
            for event in calendar.events[:10]:  # Check first 10 events
                issues = []
                
                # Check required fields
                if not event.title:
                    issues.append("Missing title")
                if not event.country:
                    issues.append("Missing country")
                if not event.currency:
                    issues.append("Missing currency")
                if not event.release_date:
                    issues.append("Missing release date")
                
                # Check date logic
                if event.release_date and event.release_date <= event.release_date.replace(year=2020):
                    issues.append("Invalid release date")
                
                # Check impact classification
                if event.impact not in [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH]:
                    issues.append("Invalid impact level")
                
                # Check relevance score
                if event.relevance_score < 0 or event.relevance_score > 1:
                    issues.append(f"Invalid relevance score: {event.relevance_score}")
                
                if issues:
                    invalid_events += 1
                    logger.debug(f"   ❌ {event.title}: {', '.join(issues)}")
                else:
                    valid_events += 1
                
                # Log sample event details
                if valid_events <= 3:
                    logger.info(f"   ✅ {event.title}:")
                    logger.info(f"      Country: {event.country}, Currency: {event.currency}")
                    logger.info(f"      Date: {event.release_date.date()}, Impact: {event.impact.value}")
                    logger.info(f"      Relevance: {event.relevance_score:.2f}")
                    if event.previous_value:
                        logger.info(f"      Previous: {event.previous_value}")
                    logger.info(f"      Is upcoming: {event.is_upcoming}")
            
            logger.info(f"✅ Validation complete: {valid_events} valid, {invalid_events} invalid")
            
        else:
            logger.warning("❌ No events to validate")
    
    except Exception as e:
        logger.error(f"❌ Validation test error: {e}")


async def main():
    """Run all economic calendar tests."""
    
    logger.info("🚀 Currency Assistant - Economic Calendar Test Suite")
    logger.info("=" * 70)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API keys
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key:
        logger.info("🔑 FRED API key found")
    else:
        logger.warning("⚠️  No FRED API key - US economic data will be limited")
    
    try:
        # Run all tests
        provider_results = await test_individual_providers()
        calendar_result = await test_economic_calendar()
        await test_convenience_functions()
        await test_event_validation()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📋 ECONOMIC CALENDAR TEST SUMMARY")
        
        # Provider summary
        working_providers = sum(1 for r in provider_results.values() if r['success'])
        total_providers = len(provider_results)
        
        logger.info(f"🌐 Data Providers: {working_providers}/{total_providers} working")
        
        for provider, result in provider_results.items():
            if result['success']:
                logger.info(f"   ✅ {provider}: {result['event_count']} events")
            else:
                logger.error(f"   ❌ {provider}: {result['error']}")
        
        # Calendar summary
        if calendar_result['success']:
            logger.info("📅 Economic Calendar: ✅ Working")
            logger.info(f"   • Total events: {calendar_result['total_events']}")
            logger.info(f"   • Upcoming events: {calendar_result['upcoming_events']}")
            logger.info(f"   • High impact events: {calendar_result['high_impact_events']}")
            logger.info(f"   • Data sources: {', '.join(calendar_result['sources'])}")
        else:
            logger.error(f"📅 Economic Calendar: ❌ {calendar_result['error']}")
        
        if working_providers > 0 and calendar_result['success']:
            logger.info("🎉 ECONOMIC CALENDAR INTEGRATION WORKING!")
        else:
            logger.warning(f"⚠️  Partial functionality - {working_providers} providers working")
    
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())