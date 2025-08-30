#!/usr/bin/env python3
"""
Examine the actual scraped data to see if it contains useful information
"""
import asyncio
from crawl4ai import AsyncWebCrawler

async def examine_rbi_data():
    """Look at actual RBI data content to see if it's useful"""
    print("=== Examining RBI Data Content ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # RBI Press Releases - should have MPC decisions and policy updates
        print("\n--- RBI Press Releases Content ---")
        result = await crawler.arun(
            url="https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
            word_count_threshold=10
        )
        
        print(f"Full content length: {len(result.markdown)}")
        
        # Look for key economic terms
        content = result.markdown.lower()
        key_terms = [
            "monetary policy", "mpc", "repo rate", "inflation", "gdp", 
            "interest rate", "policy", "decision", "meeting", "announcement"
        ]
        
        print("\nKey economic terms found:")
        for term in key_terms:
            count = content.count(term)
            if count > 0:
                print(f"  '{term}': {count} mentions")
        
        # Extract first few relevant paragraphs
        lines = result.markdown.split('\n')
        relevant_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(term in line_lower for term in key_terms) and len(line.strip()) > 20:
                relevant_lines.append(line.strip())
                if len(relevant_lines) >= 10:
                    break
        
        print(f"\nRelevant content samples:")
        for i, line in enumerate(relevant_lines[:5], 1):
            print(f"{i}. {line}")

async def examine_news_data():
    """Look at actual financial news content"""
    print("\n=== Examining Financial News Content ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Test FXStreet and Investing.com (the working ones)
        sites = [
            ("FXStreet", "https://www.fxstreet.com/news"),
            ("Investing.com", "https://www.investing.com/news/forex-news")
        ]
        
        for name, url in sites:
            print(f"\n--- {name} Content ---")
            result = await crawler.arun(url=url, word_count_threshold=10)
            
            content = result.markdown.lower()
            
            # Look for forex/currency relevant terms
            forex_terms = [
                "usd", "eur", "gbp", "jpy", "forex", "currency", "exchange rate",
                "fed", "ecb", "central bank", "interest rate", "inflation"
            ]
            
            print(f"Content length: {len(result.markdown)}")
            print("Currency/Forex terms found:")
            for term in forex_terms:
                count = content.count(term)
                if count > 0:
                    print(f"  '{term}': {count} mentions")
            
            # Extract forex headlines
            lines = result.markdown.split('\n')
            headlines = []
            for line in lines:
                line_clean = line.strip()
                line_lower = line_clean.lower()
                # Look for lines that seem like headlines about forex
                if (any(term in line_lower for term in ['usd', 'eur', 'forex', 'dollar', 'euro']) 
                    and 20 <= len(line_clean) <= 150 
                    and not line_clean.startswith('[')
                    and not line_clean.startswith('*')):
                    headlines.append(line_clean)
                    if len(headlines) >= 5:
                        break
            
            print(f"\nSample forex headlines:")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. {headline}")

async def main():
    await examine_rbi_data()
    await examine_news_data()

if __name__ == "__main__":
    asyncio.run(main())