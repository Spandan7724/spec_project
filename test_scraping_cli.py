#!/usr/bin/env python3
"""
Simple CLI for testing web scraping tool
Usage: python test_scraping_cli.py
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tools.web_scraper import GenericScrapingInterface
from tools.decision_engine import DecisionEngine

async def main():
    """Interactive CLI for testing scraping"""
    print("=== Web Scraping Tool Test CLI ===")
    print("Commands:")
    print("  scrape <url> - Scrape specific URL")
    print("  multi <url1,url2,...> - Scrape multiple URLs") 
    print("  decision <query> - Test decision logic")
    print("  quit - Exit")
    print()
    
    decision_engine = DecisionEngine()
    
    while True:
        try:
            command = input("Enter command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(' ', 1)
            if len(parts) < 2:
                print("Invalid command. Try 'scrape <url>' or 'decision <query>'")
                continue
            
            cmd, arg = parts[0].lower(), parts[1]
            
            if cmd == 'scrape':
                print(f"\nScraping: {arg}")
                result = await GenericScrapingInterface.scrape(arg)
                print(f"Success: {result['success']}")
                print(f"Cached: {result['cached']}")
                print(f"Content length: {len(result['content'])}")
                
                if result['success']:
                    preview = result['content'][:300] + '...' if len(result['content']) > 300 else result['content']
                    print(f"Content preview:\n{preview}")
                else:
                    print(f"Error: {result['error']}")
            
            elif cmd == 'multi':
                urls = [url.strip() for url in arg.split(',')]
                print(f"\nScraping {len(urls)} URLs...")
                result = await GenericScrapingInterface.scrape_multiple(urls)
                print(f"Success: {result['success']}")
                print(f"Successful sources: {result['successful_sources']}/{result['total_sources']}")
                
                if result['sources']:
                    print("Sources:")
                    for source in result['sources']:
                        print(f"  ✅ {source}")
                
                if result['errors']:
                    print("Errors:")
                    for error in result['errors']:
                        print(f"  ❌ {error}")
                
                if result['success']:
                    content_length = len(result['content'])
                    print(f"Total content length: {content_length}")
                    if content_length > 500:
                        preview = result['content'][:500] + '...'
                        print(f"Content preview:\n{preview}")
            
            elif cmd == 'decision':
                print(f"\nAnalyzing query: '{arg}'")
                decision = decision_engine.analyze_query(arg)
                print(f"Should scrape: {decision.should_scrape}")
                print(f"Reason: {decision.reason}")
                print(f"Confidence: {decision.confidence:.2f}")
                print(f"Content type: {decision.content_type}")
                
                if decision.suggested_urls:
                    print("Suggested URLs:")
                    for i, url in enumerate(decision.suggested_urls, 1):
                        print(f"  {i}. {url}")
            
            else:
                print("Unknown command. Use 'scrape', 'multi', or 'decision'")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())