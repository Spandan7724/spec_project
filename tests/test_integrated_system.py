#!/usr/bin/env python3
"""
Test integrated economic calendar with RBI scraper and news system
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.economic.calendar_collector import EconomicCalendarCollector
from data_collection.news.news_scraper import FinancialNewsScraper

async def test_integrated_calendar():
    """Test economic calendar with integrated RBI scraper"""
    print("=== Testing Integrated Economic Calendar ===")
    
    collector = EconomicCalendarCollector()
    
    try:
        calendar = await collector.get_economic_calendar(days_ahead=14)
        
        if calendar:
            print(f"\nIntegrated Calendar Results:")
            print(f"Total events: {calendar.event_count}")
            print(f"Upcoming events: {calendar.upcoming_count}")
            print(f"High impact events: {calendar.high_impact_count}")
            print(f"Sources: {calendar.sources}")
            
            print(f"\nAll Events:")
            for i, event in enumerate(calendar.events, 1):
                print(f"{i}. {event.title} ({event.source})")
                print(f"   Date: {event.release_date}")
                print(f"   Impact: {event.impact}")
                print(f"   Country: {event.country}")
                print()
        else:
            print("No calendar data retrieved")
            
    except Exception as e:
        print(f"Error testing integrated calendar: {e}")
        import traceback
        traceback.print_exc()

async def test_news_system():
    """Test news scraping system"""
    print("\n=== Testing News System ===")
    
    async with FinancialNewsScraper() as scraper:
        try:
            articles = await scraper.get_latest_news(hours_back=24)
            
            if articles:
                print(f"\nNews System Results:")
                print(f"Articles found: {len(articles)}")
                
                print(f"\nSample Articles:")
                for i, article in enumerate(articles[:5], 1):
                    print(f"{i}. {article.title}")
                    print(f"   Source: {article.source}")
                    print(f"   Sentiment: {article.sentiment}")
                    print(f"   Currencies: {article.affected_currencies}")
                    print()
            else:
                print("No news articles found")
                
        except Exception as e:
            print(f"Error testing news system: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run both integrated tests"""
    await test_integrated_calendar()
    await test_news_system()

if __name__ == "__main__":
    asyncio.run(main())