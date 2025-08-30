#!/usr/bin/env python3
"""
Simple Agent Interface for Web Scraping Tool
Provides easy-to-use methods for AI agents in conversational contexts
"""
import asyncio
from typing import Dict, List, Any, Optional

from .web_scraper import SimpleWebScraper, ScrapingResult
from .decision_engine import DecisionEngine

class AgentScrapingInterface:
    """
    Simplified interface for AI agents to use web scraping capabilities
    Designed for conversational gap-filling scenarios
    """
    
    @staticmethod
    async def should_i_scrape(user_query: str, conversation_context: str = "") -> Dict[str, Any]:
        """
        Quick decision check - should the agent scrape for this query?
        
        Usage: Agent calls this first to decide if scraping is warranted
        
        Returns:
            Decision info with recommendation and reasoning
        """
        decision_engine = DecisionEngine()
        decision = decision_engine.analyze_query(user_query, conversation_context)
        
        return {
            'should_scrape': decision.should_scrape,
            'reason': decision.reason,
            'confidence': decision.confidence,
            'content_type': decision.content_type,
            'suggested_urls': decision.suggested_urls
        }
    
    @staticmethod
    async def get_current_info(user_query: str, conversation_context: str = "",
                             specific_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get current information relevant to user query
        
        Main method for agents to get up-to-date info during conversations
        
        Args:
            user_query: What the user is asking about
            conversation_context: Previous conversation for context
            specific_urls: Optional specific URLs to scrape
            
        Returns:
            Structured response with scraped data and sources
        """
        # Determine URLs to scrape
        if specific_urls:
            urls_to_scrape = specific_urls
        else:
            # Use decision engine to get suggested URLs
            decision_engine = DecisionEngine()
            decision = decision_engine.analyze_query(user_query, conversation_context)
            
            if not decision.should_scrape:
                return {
                    'success': False,
                    'results_found': 0,
                    'sources_checked': 0,
                    'data': {},
                    'citations': [],
                    'errors': [f"Scraping not recommended: {decision.reason}"]
                }
            
            urls_to_scrape = decision.suggested_urls
        
        if not urls_to_scrape:
            return {
                'success': False,
                'results_found': 0,
                'sources_checked': 0,
                'data': {},
                'citations': [],
                'errors': ["No URLs available for scraping"]
            }
        
        # Scrape the URLs
        async with SimpleWebScraper() as scraper:
            bypass_cache = scraper.should_bypass_cache(user_query + " " + conversation_context)
            results = await scraper.scrape_multiple_urls(urls_to_scrape, bypass_cache)
            
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            response = {
                'success': len(successful_results) > 0,
                'results_found': len(successful_results),
                'sources_checked': len(results),
                'data': {},
                'citations': [],
                'errors': []
            }
            
            # Combine content from all successful results
            all_content = []
            for result in successful_results:
                all_content.append(f"=== {result.url} ===\n{result.content}")
                
                # Add citation
                cache_info = f" (cached, {result.cache_age_seconds:.0f}s old)" if result.cached else ""
                response['citations'].append(f"{result.url}{cache_info}")
            
            if all_content:
                response['data']['combined_content'] = '\n\n'.join(all_content)
                response['data']['content_length'] = len(response['data']['combined_content'])
            
            # Track errors
            for result in failed_results:
                if result.error:
                    response['errors'].append(f"{result.url}: {result.error}")
            
            return response
    
    @staticmethod
    async def check_provider_updates(provider_name: str) -> Dict[str, Any]:
        """
        Check for recent updates from specific currency provider
        
        Usage: When user asks about specific provider policies/changes
        """
        query = f"latest {provider_name} fees policies updates"
        return await AgentScrapingInterface.get_current_info(query)
    
    @staticmethod
    async def check_economic_events(event_type: str = "currency policy") -> Dict[str, Any]:
        """
        Check for recent economic events affecting currencies
        
        Usage: When user asks about economic factors affecting timing
        """
        query = f"latest {event_type} federal reserve ECB announcements"
        return await AgentScrapingInterface.get_current_info(query)
    
    @staticmethod
    async def verify_claim(user_claim: str) -> Dict[str, Any]:
        """
        Verify user's claim or statement with current information
        
        Usage: When user says "I heard that..." or challenges provided info
        """
        query = f"verify current information about {user_claim}"
        return await AgentScrapingInterface.get_current_info(query)
    
    @staticmethod
    async def research_topic(topic: str, context: str = "") -> Dict[str, Any]:
        """
        General research on any topic related to currency/finance
        
        Usage: When user asks about topics not covered by existing APIs
        """
        query = f"research {topic} currency finance {context}"
        return await AgentScrapingInterface.get_current_info(query, context)

# Example conversation flow
async def example_conversation_usage():
    """
    Example of how this tool would be used in agent conversations
    """
    
    # User: "Should I convert USD to EUR this week?"
    # Agent: Checks APIs, provides analysis
    
    # User: "But what about the new Wise fees I heard about?"
    user_query = "what about the new Wise fees I heard about?"
    conversation_context = "Previous discussion about USD to EUR conversion timing"
    
    # Agent checks if scraping is needed
    decision = await AgentScrapingInterface.should_i_scrape(user_query, conversation_context)
    print(f"Should scrape: {decision['should_scrape']}")
    print(f"Reason: {decision['reason']}")
    
    if decision['should_scrape']:
        # Agent scrapes for current Wise fee information
        info = await AgentScrapingInterface.check_provider_updates("Wise")
        print(f"Current Wise info: {info}")
    
    # User: "Did the Fed announce anything today?"
    fed_info = await AgentScrapingInterface.check_economic_events("federal reserve announcements")
    print(f"Fed updates: {fed_info}")

if __name__ == "__main__":
    asyncio.run(example_conversation_usage())