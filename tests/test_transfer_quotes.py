#!/usr/bin/env python3
"""
Test actual transfer quote systems to get real costs with fees
"""
import asyncio
import re
from crawl4ai import AsyncWebCrawler

async def test_wise_transfer_quote():
    """Test Wise actual transfer quote system"""
    print("=== Testing Wise Transfer Quote System ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Try Wise send money page to get actual transfer costs
        test_urls = [
            "https://wise.com/send-money/",
            "https://wise.com/send/",
            "https://wise.com/gb/send-money/",
            "https://wise.com/us/send/",
        ]
        
        for url in test_urls:
            try:
                print(f"\nTesting Wise URL: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    content = result.markdown
                    
                    # Look for transfer cost information
                    transfer_terms = ["you send", "they receive", "fee", "total cost", "our fee"]
                    print("Transfer cost terms found:")
                    for term in transfer_terms:
                        count = content.lower().count(term.lower())
                        if count > 0:
                            print(f"  '{term}': {count} mentions")
                    
                    # Look for amount fields
                    amount_patterns = [
                        r"send.*?\$?([\d,]+\.?\d*)",
                        r"receive.*?€?([\d,]+\.?\d*)",
                        r"fee.*?\$?([\d,]+\.?\d*)",
                    ]
                    
                    print("Amount patterns found:")
                    for pattern in amount_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            print(f"  '{pattern}': {matches[:5]}")
                else:
                    print(f"❌ Status: {result.status_code}")
                    
            except Exception as e:
                print(f"Error testing {url}: {e}")

async def test_remitly_transfer_quote():
    """Test Remitly transfer quote system"""
    print("\n=== Testing Remitly Transfer Quote System ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Try Remitly send pages to get actual costs
        test_urls = [
            "https://www.remitly.com/us/en/send/usa/germany",  # Specific corridor
            "https://www.remitly.com/us/en/send-money/",
            "https://www.remitly.com/us/en/united-states/send-money/",
        ]
        
        for url in test_urls:
            try:
                print(f"\nTesting Remitly URL: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    content = result.markdown
                    
                    # Look for transfer quote information
                    quote_terms = ["you send", "they get", "exchange rate", "transfer fee", "total"]
                    print("Transfer quote terms found:")
                    for term in quote_terms:
                        count = content.lower().count(term.lower())
                        if count > 0:
                            print(f"  '{term}': {count} mentions")
                    
                    # Look for specific USD/EUR transfer costs
                    usd_eur_patterns = [
                        r"USD.*?EUR.*?([\d,]+\.?\d*)",
                        r"\$(\d+).*?€(\d+)",
                        r"send \$(\d+).*?receive.*?€([\d,]+\.?\d*)",
                    ]
                    
                    print("USD/EUR transfer patterns:")
                    for pattern in usd_eur_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            print(f"  '{pattern}': {matches[:3]}")
                else:
                    print(f"❌ Status: {result.status_code}")
                    
            except Exception as e:
                print(f"Error testing {url}: {e}")

async def test_xe_money_transfer():
    """Test XE.com money transfer service (not just converter)"""
    print("\n=== Testing XE Money Transfer System ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Try XE Money transfer pages
        test_urls = [
            "https://www.xe.com/money-transfer/",
            "https://www.xe.com/send-money/",
            "https://transfer.xe.com/",
        ]
        
        for url in test_urls:
            try:
                print(f"\nTesting XE URL: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    content = result.markdown
                    
                    # Look for transfer service information
                    transfer_terms = ["send money", "transfer fee", "you send", "they receive", "our rate"]
                    print("Transfer service terms found:")
                    for term in transfer_terms:
                        count = content.lower().count(term.lower())
                        if count > 0:
                            print(f"  '{term}': {count} mentions")
                    
                    # Look for cost structure
                    cost_patterns = [
                        r"fee.*?\$?([\d.]+)",
                        r"rate.*?([\d.]+)",
                        r"total.*?\$?([\d,]+\.?\d*)",
                    ]
                    
                    print("Cost structure patterns:")
                    for pattern in cost_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            print(f"  '{pattern}': {matches[:5]}")
                else:
                    print(f"❌ Status: {result.status_code}")
                    
            except Exception as e:
                print(f"Error testing {url}: {e}")

async def test_bank_wire_calculators():
    """Test if banks have online wire transfer calculators"""
    print("\n=== Testing Bank Wire Transfer Calculators ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Try bank wire calculator pages
        test_urls = [
            "https://www.chase.com/personal/international",
            "https://www.bankofamerica.com/deposits/wire-transfers/",
            "https://www.wellsfargo.com/help/international/",
        ]
        
        bank_names = ["Chase", "Bank of America", "Wells Fargo"]
        
        for i, url in enumerate(test_urls):
            bank_name = bank_names[i]
            try:
                print(f"\nTesting {bank_name}: {url}")
                result = await crawler.arun(url=url, word_count_threshold=10)
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(result.markdown)}")
                
                if result.status_code == 200:
                    content = result.markdown
                    
                    # Look for fee information
                    fee_terms = ["wire transfer fee", "international fee", "exchange rate", "$"]
                    print("Fee terms found:")
                    for term in fee_terms:
                        count = content.lower().count(term.lower())
                        if count > 0:
                            print(f"  '{term}': {count} mentions")
                    
                    # Look for specific fee amounts
                    fee_amounts = re.findall(r"\$(\d+\.?\d*)", content)
                    if fee_amounts:
                        print(f"Fee amounts found: {fee_amounts[:10]}")
                        
                else:
                    print(f"❌ {bank_name} - Status: {result.status_code}")
                    
            except Exception as e:
                print(f"Error testing {bank_name}: {e}")

async def main():
    """Test all transfer quote systems"""
    await test_wise_transfer_quote()
    await test_remitly_transfer_quote()
    await test_xe_money_transfer()
    await test_bank_wire_calculators()
    
    print("\n=== Transfer Quote Testing Complete ===")
    print("Look for actual 'send X receive Y' data with real fees included")

if __name__ == "__main__":
    asyncio.run(main())