#!/usr/bin/env python3
"""
Test RBI scraper to verify it extracts real economic events
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.economic.rbi_scraper import RBIScraper

async def test_rbi_scraper():
    """Test RBI scraper with real data extraction"""
    print("=== Testing RBI Scraper ===")
    
    async with RBIScraper() as scraper:
        try:
            events = await scraper.get_upcoming_releases(days_ahead=30)
            
            print(f"\nRBI Scraper Results:")
            print(f"Number of events found: {len(events) if events else 0}")
            
            if events:
                print("\nExtracted Events:")
                for i, event in enumerate(events, 1):
                    print(f"{i}. {event.title}")
                    print(f"   Date: {event.release_date}")
                    print(f"   Impact: {event.impact}")
                    print(f"   Country: {event.country}")
                    print(f"   Source: {event.source}")
                    print(f"   Category: {event.category}")
                    if event.description:
                        print(f"   Description: {event.description}")
                    print()
            else:
                print("No events extracted")
                
        except Exception as e:
            print(f"Error testing RBI scraper: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rbi_scraper())