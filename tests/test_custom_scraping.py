#!/usr/bin/env python3
"""
Command-line test script for custom scraping
Usage: python test_custom_scraping.py "your query here" [optional_url]
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.web_scraper import GenericScrapingInterface

async def test_custom_query(query: str, custom_url: str = None):
    """Test scraping with custom query and optional URL"""
    print("=== Testing Custom Query ===")
    print(f"Query: '{query}'")
    if custom_url:
        print(f"Custom URL: {custom_url}")
    print()
    
    try:
        # First, test decision-making  
        print("1. Testing decision logic...")
        from tools.decision_engine import DecisionEngine
        decision_engine = DecisionEngine()
        decision = decision_engine.analyze_query(query)
        
        print(f"   Should scrape: {decision.should_scrape}")
        print(f"   Reason: {decision.reason}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Content type: {decision.content_type}")
        
        if decision.suggested_urls:
            print(f"   Suggested URLs: {len(decision.suggested_urls)}")
            for i, url in enumerate(decision.suggested_urls[:3], 1):
                print(f"     {i}. {url}")
        
        print()
        
        # Now test actual scraping
        print("2. Testing actual scraping...")
        
        if custom_url:
            # Test single URL scraping
            result = await GenericScrapingInterface.scrape(custom_url, query)
            print(f"   Success: {result['success']}")
            print(f"   Cached: {result['cached']}")
            print(f"   Content length: {len(result['content'])}")
            
            if result['success']:
                preview = result['content'][:300] + '...' if len(result['content']) > 300 else result['content']
                print(f"   Content preview:\n{preview}")
            else:
                print(f"   Error: {result['error']}")
        else:
            # Use decision engine URLs if available
            if decision.should_scrape and decision.suggested_urls:
                urls_to_use = decision.suggested_urls[:2]  # Limit to 2 for testing
                result = await GenericScrapingInterface.scrape_multiple(urls_to_use, query)
                
                print(f"   Success: {result['success']}")
                print(f"   Sources checked: {result['total_sources']}")
                print(f"   Successful sources: {result['successful_sources']}")
                
                if result['sources']:
                    print("   📚 Sources:")
                    for source in result['sources']:
                        print(f"     - {source}")
                
                if result['errors']:
                    print("   ❌ Errors:")
                    for error in result['errors']:
                        print(f"     - {error}")
                
                if result['success']:
                    content_length = len(result['content'])
                    print(f"   Content length: {content_length}")
                    if content_length > 500:
                        preview = result['content'][:500] + '...'
                        print(f"   Content preview:\n{preview}")
            else:
                print("   No scraping recommended or no URLs available")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")

def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python test_custom_scraping.py \"your query here\"")
    print("  python test_custom_scraping.py \"your query\" \"https://example.com\"")
    print()
    print("Examples:")
    print("  python test_custom_scraping.py \"What's the latest USD to EUR rate?\"")
    print("  python test_custom_scraping.py \"Wise fees\" \"https://wise.com/help/\"")
    print("  python test_custom_scraping.py \"ECB announcement today\"")

async def run_example_tests():
    """Run some example tests to demonstrate functionality"""
    print("=== Running Example Tests ===")
    
    examples = [
        ("What's the latest USD to EUR rate?", None),
        ("Did the ECB announce anything today?", None),
        ("Wise current fee structure", None),
        ("Custom XE scraping", "https://www.xe.com/currencyconverter/convert/?Amount=1000&From=USD&To=EUR"),
    ]
    
    for i, (query, url) in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        await test_custom_query(query, url)
        
        if i < len(examples):
            print("\n" + "="*60)

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print_usage()
        print("\nRunning example tests instead...\n")
        await run_example_tests()
        return
    
    query = sys.argv[1]
    custom_url = sys.argv[2] if len(sys.argv) > 2 else None
    
    await test_custom_query(query, custom_url)

if __name__ == "__main__":
    asyncio.run(main())