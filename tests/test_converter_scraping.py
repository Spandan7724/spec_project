#!/usr/bin/env python3
"""
Test scraping converter pages to extract real rates and fees
"""
import asyncio
import re
from crawl4ai import AsyncWebCrawler

async def test_wise_converter():
    """Test scraping Wise converter for real rates and fees"""
    print("=== Testing Wise Converter Scraping ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Test with specific USD to EUR conversion
        url = "https://wise.com/gb/currency-converter/usd-to-eur"
        
        try:
            print(f"Scraping: {url}")
            result = await crawler.arun(url=url, word_count_threshold=10)
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            
            if result.status_code == 200:
                content = result.markdown
                
                # Look for rate information
                rate_patterns = [
                    r"1 USD = ([\d.]+) EUR",
                    r"1\.00 USD = ([\d.]+) EUR", 
                    r"exchange rate.*?([\d.]+)",
                    r"rate.*?([\d.]+)",
                ]
                
                print("\nSearching for exchange rates:")
                for pattern in rate_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
                
                # Look for fee information
                fee_patterns = [
                    r"fee.*?\$?([\d.]+)",
                    r"cost.*?\$?([\d.]+)",
                    r"charge.*?\$?([\d.]+)",
                    r"(\d+\.\d+)%.*fee",
                ]
                
                print("\nSearching for fees:")
                for pattern in fee_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
                
                # Look for total amount received
                amount_patterns = [
                    r"you'll receive.*?([\d,]+\.?\d*)",
                    r"recipient gets.*?([\d,]+\.?\d*)",
                    r"total.*?([\d,]+\.?\d*)",
                ]
                
                print("\nSearching for amounts received:")
                for pattern in amount_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
                        
                # Look for specific USD/EUR data
                usd_eur_matches = re.findall(r"USD.*?EUR.*?([\d.]+)", content)
                if usd_eur_matches:
                    print(f"\nUSD/EUR specific data: {usd_eur_matches[:5]}")
            
        except Exception as e:
            print(f"Error scraping Wise: {e}")

async def test_xe_converter():
    """Test scraping XE.com converter for rates"""
    print("\n=== Testing XE.com Converter Scraping ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        url = "https://www.xe.com/currencyconverter/convert/?Amount=1000&From=USD&To=EUR"
        
        try:
            print(f"Scraping: {url}")
            result = await crawler.arun(url=url, word_count_threshold=10)
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            
            if result.status_code == 200:
                content = result.markdown
                
                # Look for conversion results
                conversion_patterns = [
                    r"1,000\.00 USD.*?([\d,]+\.?\d*) EUR",
                    r"1000 USD = ([\d,]+\.?\d*) EUR",
                    r"= ([\d,]+\.?\d*) EUR",
                    r"([\d,]+\.?\d*) EUR",
                ]
                
                print("\nSearching for conversion amounts:")
                for pattern in conversion_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
                
                # Look for exchange rate
                rate_patterns = [
                    r"1 USD = ([\d.]+) EUR",
                    r"exchange rate.*?([\d.]+)",
                ]
                
                print("\nSearching for exchange rates:")
                for pattern in rate_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
                        
                # Check if we find specific numbers
                numbers = re.findall(r"([\d,]+\.?\d+)", content)
                print(f"\nAll numbers found: {numbers[:10]}")
            
        except Exception as e:
            print(f"Error scraping XE.com: {e}")

async def test_remitly_converter():
    """Test scraping Remitly for rates"""
    print("\n=== Testing Remitly Converter Scraping ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        url = "https://www.remitly.com/"
        
        try:
            print(f"Scraping: {url}")
            result = await crawler.arun(url=url, word_count_threshold=10)
            print(f"Status: {result.status_code}")
            print(f"Content length: {len(result.markdown)}")
            
            if result.status_code == 200:
                content = result.markdown
                
                # Look for any rate or fee information
                rate_indicators = ["exchange rate", "rate", "fee", "cost", "USD", "EUR", "$"]
                
                print("\nChecking for rate/fee content:")
                for term in rate_indicators:
                    count = content.lower().count(term.lower())
                    if count > 0:
                        print(f"  '{term}': {count} mentions")
                
                # Look for converter interface
                converter_terms = ["send", "receive", "amount", "country", "convert"]
                print("\nChecking for converter interface:")
                for term in converter_terms:
                    count = content.lower().count(term.lower())
                    if count > 0:
                        print(f"  '{term}': {count} mentions")
            
        except Exception as e:
            print(f"Error scraping Remitly: {e}")

async def main():
    """Test scraping converter pages for cost data"""
    await test_wise_converter()
    await test_xe_converter()
    await test_remitly_converter()
    
    print("\n=== Converter Scraping Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())