#!/usr/bin/env python3
"""
Test financial news scraper
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.news.news_scraper import FinancialNewsScraper

async def test_news_scraper():
    """Test news scraper with real data extraction"""
    print("=== Testing Financial News Scraper ===")
    
    async with FinancialNewsScraper() as scraper:
        try:
            articles = await scraper.get_latest_news(hours_back=48)
            
            print(f"\nNews Scraper Results:")
            print(f"Number of articles found: {len(articles) if articles else 0}")
            
            if articles:
                print("\nExtracted Articles:")
                for i, article in enumerate(articles, 1):
                    print(f"{i}. {article.title}")
                    print(f"   Source: {article.source}")
                    print(f"   URL: {article.url}")
                    print(f"   Sentiment: {article.sentiment}")
                    print(f"   Affected Currencies: {article.affected_currencies}")
                    print(f"   Category: {article.category}")
                    print()
                    if i >= 10:  # Limit output
                        print(f"... and {len(articles) - 10} more articles")
                        break
            else:
                print("No articles extracted")
                
        except Exception as e:
            print(f"Error testing news scraper: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_news_scraper())