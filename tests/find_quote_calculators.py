#!/usr/bin/env python3
"""
Find actual quote calculator URLs for transfer services
"""
import asyncio
import re
from crawl4ai import AsyncWebCrawler

async def find_wise_calculator():
    """Find Wise's actual quote calculator interface"""
    print("=== Finding Wise Quote Calculator ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Start from main page and look for calculator links
        result = await crawler.arun(url="https://wise.com/gb/send-money/", word_count_threshold=10)
        
        if result.status_code == 200:
            content = result.markdown
            
            # Look for calculator or quote-related URLs
            calculator_links = re.findall(r'https://wise\.com/[^)]*(?:send|quote|calculator|transfer)[^)]*', content)
            
            print(f"Found {len(calculator_links)} potential calculator links:")
            for i, link in enumerate(set(calculator_links[:10]), 1):  # Remove duplicates
                print(f"{i}. {link}")
            
            # Look for form elements or interactive components
            form_indicators = ["amount", "from", "to", "calculate", "get quote", "send"]
            print("\nForm indicators found:")
            for indicator in form_indicators:
                count = content.lower().count(indicator.lower())
                if count > 0:
                    print(f"  '{indicator}': {count} mentions")
            
            # Try to find specific quote URLs in the content
            quote_urls = re.findall(r'(https://[^"\s)]*(?:quote|send|transfer)[^"\s)]*)', content)
            if quote_urls:
                print("\nQuote URLs found:")
                for url in set(quote_urls[:5]):
                    print(f"  - {url}")

async def find_remitly_calculator():
    """Find Remitly's quote calculator"""
    print("\n=== Finding Remitly Quote Calculator ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Try main page to find calculator
        result = await crawler.arun(url="https://www.remitly.com/", word_count_threshold=10)
        
        if result.status_code == 200:
            content = result.markdown
            
            # Look for send money or calculator links
            calculator_patterns = [
                r'https://www\.remitly\.com/[^)]*(?:send|quote|calculator)[^)]*',
                r'href="([^"]*(?:send|transfer|quote)[^"]*)"',
            ]
            
            all_links = []
            for pattern in calculator_patterns:
                links = re.findall(pattern, content, re.IGNORECASE)
                all_links.extend(links)
            
            print(f"Found {len(all_links)} potential calculator links:")
            for i, link in enumerate(set(all_links[:10]), 1):
                print(f"{i}. {link}")
            
            # Look for specific amount/quote functionality
            amount_indicators = ["enter amount", "how much", "send amount", "calculate"]
            print("\nAmount input indicators:")
            for indicator in amount_indicators:
                count = content.lower().count(indicator.lower())
                if count > 0:
                    print(f"  '{indicator}': {count} mentions")

async def find_xe_calculator():
    """Find XE's actual transfer calculator"""
    print("\n=== Finding XE Transfer Calculator ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url="https://www.xe.com/send-money/", word_count_threshold=10)
        
        if result.status_code == 200:
            content = result.markdown
            
            # Look for XE Money URLs or calculator interfaces
            xe_links = re.findall(r'https://[^"\s)]*xe[^"\s)]*(?:money|transfer|send|quote)[^"\s)]*', content)
            
            print(f"Found {len(xe_links)} XE transfer-related links:")
            for i, link in enumerate(set(xe_links[:10]), 1):
                print(f"{i}. {link}")
            
            # Look for calculator interface elements
            calc_terms = ["calculate", "quote", "estimate", "amount", "fee breakdown"]
            print("\nCalculator interface terms:")
            for term in calc_terms:
                count = content.lower().count(term.lower())
                if count > 0:
                    print(f"  '{term}': {count} mentions")

async def test_direct_calculator_urls():
    """Test common calculator URL patterns directly"""
    print("\n=== Testing Direct Calculator URL Patterns ===")
    
    # Common calculator URL patterns
    test_urls = [
        # Wise patterns
        ("Wise Quote", "https://wise.com/gb/send/quote/"),
        ("Wise Calculator", "https://wise.com/calculator/"),
        ("Wise Send Flow", "https://wise.com/gb/send/flow/"),
        
        # Remitly patterns  
        ("Remitly Quote", "https://www.remitly.com/quote/"),
        ("Remitly Calculator", "https://www.remitly.com/calculator/"),
        ("Remitly Send", "https://www.remitly.com/send/"),
        
        # XE patterns
        ("XE Quote", "https://www.xe.com/quote/"),
        ("XE Calculator", "https://www.xe.com/calculator/"),
        ("XE Transfer", "https://transfer.xe.com/quote/"),
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for name, url in test_urls:
            try:
                print(f"\nTesting {name}: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    print(f"✅ {name} - FOUND!")
                    
                    # Quick check for quote functionality
                    content = result.markdown.lower()
                    quote_terms = ["amount", "send", "receive", "fee", "rate", "total"]
                    found_terms = [term for term in quote_terms if term in content]
                    print(f"  Quote terms: {found_terms}")
                else:
                    print(f"❌ {name} - Status {result.status_code}")
                    
            except Exception as e:
                print(f"❌ {name} - Error: {e}")

async def main():
    """Find working quote calculator URLs"""
    await find_wise_calculator()
    await find_remitly_calculator() 
    await find_xe_calculator()
    await test_direct_calculator_urls()

if __name__ == "__main__":
    asyncio.run(main())