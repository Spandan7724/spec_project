#!/usr/bin/env python3
"""
Test detailed cost extraction from working converter sources
"""
import asyncio
import re
from crawl4ai import AsyncWebCrawler

async def detailed_xe_extraction():
    """Extract detailed cost breakdown from XE.com"""
    print("=== Detailed XE.com Cost Extraction ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        # Test different amounts to see cost structure
        test_amounts = [100, 1000, 5000]
        
        for amount in test_amounts:
            url = f"https://www.xe.com/currencyconverter/convert/?Amount={amount}&From=USD&To=EUR"
            
            try:
                print(f"\nTesting ${amount} USD to EUR:")
                result = await crawler.arun(url=url, word_count_threshold=10)
                
                if result.status_code == 200:
                    content = result.markdown
                    
                    # Extract the conversion result
                    conversion_match = re.search(rf"{amount:,}\.00 USD.*?([\d,]+\.?\d*) EUR", content)
                    if conversion_match:
                        eur_received = conversion_match.group(1)
                        print(f"  Amount received: {eur_received} EUR")
                        
                        # Calculate effective rate
                        try:
                            eur_float = float(eur_received.replace(',', ''))
                            effective_rate = eur_float / amount
                            print(f"  Effective rate: {effective_rate:.6f}")
                        except:
                            pass
                    
                    # Look for exchange rate
                    rate_match = re.search(r"1 USD = ([\d.]+) EUR", content)
                    if rate_match:
                        mid_rate = rate_match.group(1)
                        print(f"  Mid-market rate: {mid_rate}")
                        
                        # Calculate spread
                        try:
                            eur_float = float(eur_received.replace(',', ''))
                            effective_rate = eur_float / amount
                            mid_rate_float = float(mid_rate)
                            spread = ((mid_rate_float - effective_rate) / mid_rate_float) * 100
                            print(f"  Spread/markup: {spread:.2f}%")
                        except:
                            pass
                    
                    # Look for any fee mentions
                    fee_mentions = re.findall(r"fee.*?(\d+\.?\d*)", content, re.IGNORECASE)
                    if fee_mentions:
                        print(f"  Fee mentions: {fee_mentions}")
                
            except Exception as e:
                print(f"Error testing ${amount}: {e}")

async def detailed_remitly_extraction():
    """Extract detailed cost info from Remitly"""
    print("\n=== Detailed Remitly Cost Extraction ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        url = "https://www.remitly.com/"
        
        try:
            print(f"Scraping: {url}")
            result = await crawler.arun(url=url, word_count_threshold=10)
            
            if result.status_code == 200:
                content = result.markdown
                
                print(f"Content length: {len(content)}")
                
                # Look for fee structure
                fee_lines = []
                lines = content.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    if any(term in line_lower for term in ['fee', 'cost', 'rate', 'exchange']) and len(line.strip()) > 10:
                        fee_lines.append(line.strip())
                        if len(fee_lines) >= 10:
                            break
                
                print("\nFee/rate related content:")
                for i, line in enumerate(fee_lines, 1):
                    print(f"  {i}. {line}")
                
                # Look for specific rates
                rate_mentions = re.findall(r"rate.*?([\d.]+)", content, re.IGNORECASE)
                if rate_mentions:
                    print(f"\nRate values found: {rate_mentions[:10]}")
                
                # Look for USD/EUR specific
                usd_eur_content = re.findall(r"USD.*?EUR.*?([\d.]+)", content, re.IGNORECASE)
                if usd_eur_content:
                    print(f"USD/EUR data: {usd_eur_content}")
        
        except Exception as e:
            print(f"Error scraping Remitly: {e}")

async def test_wise_main_converter():
    """Test Wise main converter page"""
    print("\n=== Testing Wise Main Converter ===")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        url = "https://wise.com/gb/currency-converter/"
        
        try:
            print(f"Scraping: {url}")
            result = await crawler.arun(url=url, word_count_threshold=10)
            
            if result.status_code == 200:
                content = result.markdown
                print(f"Status: {result.status_code}")
                print(f"Content length: {len(content)}")
                
                # Look for any interactive elements or rate data
                interactive_terms = ["convert", "amount", "from", "to", "send", "receive"]
                print("\nInteractive elements found:")
                for term in interactive_terms:
                    count = content.lower().count(term.lower())
                    if count > 0:
                        print(f"  '{term}': {count} mentions")
                
                # Look for any rate information
                rate_patterns = [
                    r"USD.*?EUR.*?([\d.]+)",
                    r"([\d.]+).*rate",
                    r"1.*=([\d.]+)",
                ]
                
                print("\nSearching for rates:")
                for pattern in rate_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern '{pattern}': {matches[:3]}")
        
        except Exception as e:
            print(f"Error scraping Wise main: {e}")

async def main():
    """Test detailed cost extraction from all working sources"""
    await detailed_xe_extraction()
    await detailed_remitly_extraction() 
    await test_wise_main_converter()

if __name__ == "__main__":
    asyncio.run(main())