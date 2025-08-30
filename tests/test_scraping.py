#!/usr/bin/env python3
"""
Test crawl4ai scraping capabilities for RBI and financial news data
"""
import asyncio
from crawl4ai import AsyncWebCrawler

async def test_rbi_scraping():
    """Test scraping RBI economic calendar and press releases"""
    print("=== Testing RBI Website Scraping ===")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Test 1: RBI Data Releases page
        print("\n1. Testing RBI Data Releases...")
        try:
            result = await crawler.arun(
                url="https://www.rbi.org.in/scripts/statistics.aspx",
                word_count_threshold=10
            )
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            print(f"First 500 chars:\n{result.markdown[:500]}")
        except Exception as e:
            print(f"Error scraping RBI statistics: {e}")
        
        # Test 2: RBI Press Releases
        print("\n2. Testing RBI Press Releases...")
        try:
            result = await crawler.arun(
                url="https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
                word_count_threshold=10
            )
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            print(f"First 500 chars:\n{result.markdown[:500]}")
        except Exception as e:
            print(f"Error scraping RBI press releases: {e}")

async def test_financial_news_scraping():
    """Test scraping financial news sites for forex/currency news"""
    print("\n=== Testing Financial News Scraping ===")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Test 1: Reuters Currency news
        print("\n1. Testing Reuters Currencies...")
        try:
            result = await crawler.arun(
                url="https://www.reuters.com/markets/currencies/",
                word_count_threshold=10
            )
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            print(f"First 500 chars:\n{result.markdown[:500]}")
        except Exception as e:
            print(f"Error scraping Reuters currencies: {e}")
        
        # Test 2: MarketWatch FX
        print("\n2. Testing MarketWatch FX...")
        try:
            result = await crawler.arun(
                url="https://www.marketwatch.com/markets/currencies",
                word_count_threshold=10
            )
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            print(f"First 500 chars:\n{result.markdown[:500]}")
        except Exception as e:
            print(f"Error scraping MarketWatch FX: {e}")

async def main():
    """Run all scraping tests"""
    await test_rbi_scraping()
    await test_financial_news_scraping()
    print("\n=== Scraping Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())