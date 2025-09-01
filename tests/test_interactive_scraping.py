#!/usr/bin/env python3
"""
Interactive test script for Agent Web Scraping Tool
Allows manual testing with custom queries and URLs
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.agent_interface import AgentScrapingInterface

class InteractiveScrapingTester:
    """Interactive testing interface for the scraping tool"""
    
    def __init__(self):
        self.conversation_context = ""
    
    async def run_interactive_session(self):
        """Run interactive testing session"""
        print("=== Interactive Agent Scraping Tool Tester ===")
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        while True:
            try:
                print("Choose test mode:")
                print("1. Test decision-making (should_i_scrape)")
                print("2. Test actual scraping (get_current_info)")
                print("3. Test with custom URLs")
                print("4. Test provider-specific scraping")
                print("5. Test economic news scraping")
                print("6. Set conversation context")
                print("7. Show current context")
                print("8. Quick demo scenarios")
                print("Type 'quit' to exit")
                
                choice = input("\nEnter choice (1-8): ").strip()
                
                if choice.lower() in ['quit', 'exit', 'q']:
                    break
                elif choice == '1':
                    await self.test_decision_making()
                elif choice == '2':
                    await self.test_actual_scraping()
                elif choice == '3':
                    await self.test_custom_urls()
                elif choice == '4':
                    await self.test_provider_scraping()
                elif choice == '5':
                    await self.test_economic_scraping()
                elif choice == '6':
                    self.set_context()
                elif choice == '7':
                    self.show_context()
                elif choice == '8':
                    await self.run_demo_scenarios()
                else:
                    print("Invalid choice. Please enter 1-8.")
                
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def test_decision_making(self):
        """Test the decision-making logic"""
        print("\n--- Decision-Making Test ---")
        query = input("Enter your query: ").strip()
        
        if not query:
            print("Empty query, skipping...")
            return
        
        print(f"\nTesting query: '{query}'")
        print(f"Context: '{self.conversation_context}'")
        
        decision = await AgentScrapingInterface.should_i_scrape(
            query, self.conversation_context
        )
        
        print("\n✅ Decision Results:")
        print(f"Should scrape: {decision['should_scrape']}")
        print(f"Reason: {decision['reason']}")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"Content type: {decision['content_type']}")
        
        if decision['suggested_urls']:
            print(f"\nSuggested URLs ({len(decision['suggested_urls'])}):")
            for i, url in enumerate(decision['suggested_urls'], 1):
                print(f"  {i}. {url}")
        else:
            print("No URLs suggested")
    
    async def test_actual_scraping(self):
        """Test actual scraping functionality"""
        print("\n--- Actual Scraping Test ---")
        query = input("Enter your query: ").strip()
        
        if not query:
            print("Empty query, skipping...")
            return
        
        print(f"\nScraping for: '{query}'")
        print(f"Context: '{self.conversation_context}'")
        print("This may take a few seconds...")
        
        try:
            info = await AgentScrapingInterface.get_current_info(
                query, self.conversation_context
            )
            
            print("\n✅ Scraping Results:")
            print(f"Success: {info['success']}")
            print(f"Sources checked: {info['sources_checked']}")
            print(f"Results found: {info['results_found']}")
            
            if info['success'] and info['data']:
                print("\n📊 Extracted Data:")
                for key, value in info['data'].items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {value[:5]}... (showing first 5)")
                    else:
                        print(f"  {key}: {value}")
            
            if info['citations']:
                print("\n📚 Citations:")
                for citation in info['citations']:
                    print(f"  - {citation}")
            
            if info['errors']:
                print("\n❌ Errors:")
                for error in info['errors']:
                    print(f"  - {error}")
                    
        except Exception as e:
            print(f"❌ Scraping failed: {e}")
    
    async def test_custom_urls(self):
        """Test scraping with custom URLs"""
        print("\n--- Custom URL Test ---")
        
        urls_input = input("Enter URLs (comma-separated): ").strip()
        if not urls_input:
            print("No URLs provided, skipping...")
            return
        
        urls = [url.strip() for url in urls_input.split(',')]
        query = input("Enter query context (what are you looking for?): ").strip()
        
        print(f"\nScraping URLs: {urls}")
        print(f"Query context: '{query}'")
        print("This may take a few seconds...")
        
        try:
            info = await AgentScrapingInterface.get_current_info(
                query or "custom URL scraping", 
                self.conversation_context,
                specific_urls=urls
            )
            
            print("\n✅ Custom URL Results:")
            print(f"Success: {info['success']}")
            print(f"Sources checked: {len(urls)}")
            print(f"Successful sources: {info['results_found']}")
            
            if info['data']:
                print("\n📊 Data Found:")
                for key, value in info['data'].items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"  {key}: {value[:3]}... (showing first 3)")
                    else:
                        print(f"  {key}: {value}")
            
            if info['errors']:
                print(f"\n❌ Errors ({len(info['errors'])}):")
                for error in info['errors']:
                    print(f"  - {error}")
                    
        except Exception as e:
            print(f"❌ Custom URL scraping failed: {e}")
    
    async def test_provider_scraping(self):
        """Test provider-specific scraping"""
        print("\n--- Provider Scraping Test ---")
        
        providers = ['wise', 'remitly', 'xe', 'revolut']
        print(f"Available providers: {', '.join(providers)}")
        
        provider = input("Enter provider name: ").strip().lower()
        if provider not in providers:
            print(f"Unknown provider. Available: {providers}")
            return
        
        print(f"\nChecking {provider.title()} for updates...")
        
        try:
            info = await AgentScrapingInterface.check_provider_updates(provider)
            
            print("\n✅ Provider Results:")
            print(f"Success: {info['success']}")
            
            if info['data']:
                print(f"\n📊 {provider.title()} Data:")
                for key, value in info['data'].items():
                    print(f"  {key}: {value}")
            
            if info['citations']:
                print("\n📚 Sources:")
                for citation in info['citations']:
                    print(f"  - {citation}")
                    
        except Exception as e:
            print(f"❌ Provider scraping failed: {e}")
    
    async def test_economic_scraping(self):
        """Test economic news scraping"""
        print("\n--- Economic News Test ---")
        
        event_types = [
            "Federal Reserve policy",
            "ECB announcements", 
            "inflation data",
            "currency policy",
            "interest rate changes"
        ]
        
        print("Common event types:")
        for i, event in enumerate(event_types, 1):
            print(f"  {i}. {event}")
        
        event = input("\nEnter event type (or custom): ").strip()
        if not event:
            print("No event type provided, skipping...")
            return
        
        print(f"\nChecking for recent: '{event}'")
        
        try:
            info = await AgentScrapingInterface.check_economic_events(event)
            
            print("\n✅ Economic News Results:")
            print(f"Success: {info['success']}")
            
            if info['data']:
                print("\n📰 News Data:")
                for key, value in info['data'].items():
                    if key == 'headlines' and isinstance(value, list):
                        print(f"  Headlines found: {len(value)}")
                        for i, headline in enumerate(value[:5], 1):
                            print(f"    {i}. {headline}")
                    else:
                        print(f"  {key}: {value}")
                        
        except Exception as e:
            print(f"❌ Economic scraping failed: {e}")
    
    def set_context(self):
        """Set conversation context"""
        print("\n--- Set Conversation Context ---")
        print(f"Current context: '{self.conversation_context}'")
        
        new_context = input("Enter new conversation context: ").strip()
        self.conversation_context = new_context
        print(f"Context updated to: '{new_context}'")
    
    def show_context(self):
        """Show current conversation context"""
        print("\n--- Current Context ---")
        print(f"Context: '{self.conversation_context}'" if self.conversation_context else "No context set")
    
    async def run_demo_scenarios(self):
        """Run predefined demo scenarios"""
        print("\n--- Demo Scenarios ---")
        
        scenarios = [
            {
                'name': 'Provider Follow-up',
                'query': "What about Wise's current fees?",
                'context': "Previous USD/EUR conversion discussion"
            },
            {
                'name': 'Time-sensitive News',
                'query': "Did anything happen with ECB today?",
                'context': "Currency timing advice discussion"
            },
            {
                'name': 'Knowledge Gap',
                'query': "I heard Remitly changed their fee structure",
                'context': "Comparing transfer providers"
            },
            {
                'name': 'Educational (Should NOT scrape)',
                'query': "How do exchange rates work?",
                'context': ""
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Query: '{scenario['query']}'")
            print(f"   Context: '{scenario['context']}'")
            
            decision = await AgentScrapingInterface.should_i_scrape(
                scenario['query'], scenario['context']
            )
            
            result = "✅ SCRAPE" if decision['should_scrape'] else "❌ NO SCRAPE"
            print(f"   Result: {result} (confidence: {decision['confidence']:.2f})")
            print(f"   Reason: {decision['reason']}")

async def main():
    """Main interactive testing function"""
    tester = InteractiveScrapingTester()
    await tester.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())