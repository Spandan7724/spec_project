#!/usr/bin/env python3
"""
Test alternative financial news sources for scraping
"""
import asyncio
from crawl4ai import AsyncWebCrawler

async def test_alternative_news_sources():
    """Test scraping alternative financial news sites"""
    print("=== Testing Alternative Financial News Sources ===")
    
    # Alternative news sites that might be more scraping-friendly
    test_sites = [
        ("FXStreet", "https://www.fxstreet.com/news"),
        ("Investing.com", "https://www.investing.com/news/forex-news"),
        ("ForexFactory", "https://www.forexfactory.com/news"),
        ("DailyFX", "https://www.dailyfx.com/news"),
        ("Yahoo Finance", "https://finance.yahoo.com/topic/currencies/"),
    ]
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        for name, url in test_sites:
            print(f"\n--- Testing {name} ---")
            try:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10
                )
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                if result.status_code == 200:
                    print(f"Sample content:\n{result.markdown[:300]}...")
                    print(f"✅ {name} - ACCESSIBLE")
                else:
                    print(f"❌ {name} - BLOCKED (Status {result.status_code})")
            except Exception as e:
                print(f"❌ {name} - ERROR: {e}")
            
            await asyncio.sleep(1)  # Be respectful to servers

if __name__ == "__main__":
    asyncio.run(test_alternative_news_sources())