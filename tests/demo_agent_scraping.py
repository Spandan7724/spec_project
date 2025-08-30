#!/usr/bin/env python3
"""
Demo of Agent Web Scraping Tool in realistic conversation scenarios
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.agent_interface import AgentScrapingInterface

async def demo_conversation_scenario():
    """
    Demo realistic conversation where agent uses scraping tool
    """
    print("=== Agent Scraping Tool Demo ===")
    print("Simulating realistic conversation scenarios...")
    
    # Scenario 1: Provider-specific follow-up
    print("\n--- Scenario 1: Provider Follow-up ---")
    print("User: 'Convert $1000 USD to EUR'")
    print("Agent: [Uses APIs to provide plan]")
    print("User: 'But what about Wise's current fees?'")
    
    decision_1 = await AgentScrapingInterface.should_i_scrape(
        "But what about Wise's current fees?",
        "Previous USD to EUR conversion discussion"
    )
    
    print(f"Agent decision: {decision_1['should_scrape']}")
    print(f"Reasoning: {decision_1['reason']}")
    print(f"Confidence: {decision_1['confidence']:.2f}")
    
    if decision_1['should_scrape']:
        print("Agent: 'Let me check Wise's current information...'")
        # In real usage, would scrape here
        print("✅ Tool would scrape Wise pages for current fees")
    
    # Scenario 2: Time-sensitive economic query
    print("\n--- Scenario 2: Recent Economic Events ---")
    print("User: 'Did the ECB announce anything today?'")
    
    decision_2 = await AgentScrapingInterface.should_i_scrape(
        "Did the ECB announce anything today?"
    )
    
    print(f"Agent decision: {decision_2['should_scrape']}")
    print(f"Reasoning: {decision_2['reason']}")
    print(f"Confidence: {decision_2['confidence']:.2f}")
    
    # Scenario 3: Knowledge gap detection
    print("\n--- Scenario 3: Knowledge Gap ---")
    print("User: 'I heard that transfer regulations changed recently'")
    
    decision_3 = await AgentScrapingInterface.should_i_scrape(
        "I heard that transfer regulations changed recently"
    )
    
    print(f"Agent decision: {decision_3['should_scrape']}")
    print(f"Reasoning: {decision_3['reason']}")
    
    # Scenario 4: Educational query (should NOT scrape)
    print("\n--- Scenario 4: Educational Query ---")
    print("User: 'How do exchange rates work?'")
    
    decision_4 = await AgentScrapingInterface.should_i_scrape(
        "How do exchange rates work?"
    )
    
    print(f"Agent decision: {decision_4['should_scrape']}")
    print(f"Reasoning: {decision_4['reason']}")
    print("✅ Agent uses existing knowledge instead of scraping")
    
    print("\n=== Demo Summary ===")
    scraping_scenarios = [decision_1, decision_2, decision_3]
    scrape_count = sum(1 for d in scraping_scenarios if d['should_scrape'])
    print(f"Scraping triggered: {scrape_count}/3 relevant scenarios")
    print(f"Educational query correctly avoided scraping: {not decision_4['should_scrape']}")

async def demo_real_scraping():
    """
    Demo actual scraping with XE.com (safe, reliable source)
    """
    print("\n=== Real Scraping Demo ===")
    print("Testing actual web scraping with XE.com...")
    
    try:
        # Test with XE.com currency converter (known to work)
        info = await AgentScrapingInterface.get_current_info(
            "Current USD to EUR exchange rate",
            specific_urls=["https://www.xe.com/currencyconverter/convert/?Amount=1000&From=USD&To=EUR"]
        )
        
        print(f"Scraping success: {info['success']}")
        print(f"Sources checked: {info['sources_checked']}")
        
        if info['success']:
            print("✅ Successfully scraped currency data")
            if 'currencies_mentioned' in info['data']:
                print(f"Currencies found: {info['data']['currencies_mentioned']}")
            if 'conversions_found' in info['data']:
                print(f"Conversions found: {info['data']['conversions_found'][:3]}")
        else:
            print(f"❌ Scraping failed: {info['errors']}")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(demo_conversation_scenario())
    asyncio.run(demo_real_scraping())