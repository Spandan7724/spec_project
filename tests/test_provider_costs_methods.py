#!/usr/bin/env python3
"""
Test different methods for getting provider cost data before implementing
"""
import asyncio
import httpx
from crawl4ai import AsyncWebCrawler

async def test_wise_api():
    """Test if Wise API is accessible without business account"""
    print("=== Testing Wise API Access ===")
    
    # Test public endpoints that might not require full business API access
    test_urls = [
        "https://api.wise.com/v1/rates",  # Public rates endpoint
        "https://wise.com/gb/currency-converter/usd-to-eur",  # Web converter page
        "https://wise.com/gb/currency-converter/",  # Main converter
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in test_urls:
            try:
                print(f"\nTesting: {url}")
                response = await client.get(url)
                print(f"Status: {response.status_code}")
                print(f"Content length: {len(response.text)}")
                
                if response.status_code == 200:
                    print(f"✅ Accessible")
                    print(f"Sample content: {response.text[:200]}...")
                else:
                    print(f"❌ Status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

async def test_revolut_api():
    """Test if Revolut API is accessible"""
    print("\n=== Testing Revolut API Access ===")
    
    test_urls = [
        "https://b2b.revolut.com/api/1.0/rate?from=USD&to=EUR&amount=1000",  # Business API
        "https://www.revolut.com/api/exchange",  # Possible public endpoint
        "https://www.revolut.com/currency-converter/",  # Web converter
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in test_urls:
            try:
                print(f"\nTesting: {url}")
                response = await client.get(url)
                print(f"Status: {response.status_code}")
                print(f"Content length: {len(response.text)}")
                
                if response.status_code == 200:
                    print(f"✅ Accessible")
                    print(f"Sample content: {response.text[:200]}...")
                else:
                    print(f"❌ Status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

async def test_xe_api():
    """Test XE.com API access"""
    print("\n=== Testing XE.com API Access ===")
    
    test_urls = [
        "https://www.xe.com/api/protected/midmarket-converter/",  # Possible API
        "https://www.xe.com/currencyconverter/convert/?Amount=1000&From=USD&To=EUR",  # Web converter
        "https://xecdapi.xe.com/v1/convert_from?from=USD&to=EUR&amount=1000",  # Official API
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in test_urls:
            try:
                print(f"\nTesting: {url}")
                response = await client.get(url)
                print(f"Status: {response.status_code}")
                print(f"Content length: {len(response.text)}")
                
                if response.status_code == 200:
                    print(f"✅ Accessible")
                    print(f"Sample content: {response.text[:200]}...")
                else:
                    print(f"❌ Status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

async def test_bank_scraping():
    """Test scraping major bank conversion pages"""
    print("\n=== Testing Bank Website Scraping ===")
    
    bank_urls = [
        ("Chase", "https://www.chase.com/personal/international/wire-transfers"),
        ("Bank of America", "https://www.bankofamerica.com/foreign-exchange/"),
        ("Wells Fargo", "https://www.wellsfargo.com/international/foreign-exchange/"),
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for bank_name, url in bank_urls:
            try:
                print(f"\nTesting {bank_name}: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    print(f"✅ {bank_name} - Accessible")
                    
                    # Look for rate/fee information
                    content_lower = result.markdown.lower()
                    fee_indicators = ["fee", "cost", "rate", "exchange", "transfer", "$"]
                    found_terms = [term for term in fee_indicators if term in content_lower]
                    print(f"Fee-related terms found: {found_terms}")
                else:
                    print(f"❌ {bank_name} - Status {result.status_code}")
                    
            except Exception as e:
                print(f"❌ {bank_name} - Error: {e}")

async def test_alternative_scraping():
    """Test scraping alternative money transfer services"""
    print("\n=== Testing Alternative Provider Scraping ===")
    
    provider_urls = [
        ("Remitly", "https://www.remitly.com/"),
        ("Western Union", "https://www.westernunion.com/"),
        ("MoneyGram", "https://www.moneygram.com/"),
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for provider_name, url in provider_urls:
            try:
                print(f"\nTesting {provider_name}: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    print(f"✅ {provider_name} - Accessible")
                else:
                    print(f"❌ {provider_name} - Status {result.status_code}")
                    
            except Exception as e:
                print(f"❌ {provider_name} - Error: {e}")

async def main():
    """Run all provider cost method tests"""
    print("Testing different approaches for provider cost data...\n")
    
    await test_wise_api()
    await test_revolut_api() 
    await test_xe_api()
    await test_bank_scraping()
    await test_alternative_scraping()
    
    print("\n=== Testing Complete ===")
    print("Check results to see which methods work for implementation")

if __name__ == "__main__":
    asyncio.run(main())