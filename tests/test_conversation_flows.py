#!/usr/bin/env python3
"""
Integration tests for conversational flows with agent scraping tool
Tests realistic user scenarios and agent decision-making
"""
import asyncio
import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.agent_interface import AgentScrapingInterface

@pytest.mark.asyncio
class TestConversationalFlows:
    """Test realistic conversation scenarios"""
    
    async def test_basic_follow_up_scenario(self):
        """
        Scenario: User gets plan, then asks about specific provider
        """
        # User: "Convert $1000 USD to EUR" (agent uses APIs)
        # User: "But what about Wise's current fees?"
        
        user_query = "But what about Wise's current fees?"
        context = "We discussed USD to EUR conversion timing and costs"
        
        decision = await AgentScrapingInterface.should_i_scrape(user_query, context)
        
        assert decision['should_scrape'] == True
        assert 'wise' in decision['reason'] or 'provider' in decision['reason']
        assert decision['confidence'] > 0.7
    
    async def test_economic_news_follow_up(self):
        """
        Scenario: User asks about recent economic events
        """
        # User: Gets timing advice based on economic calendar
        # User: "Did anything happen with the Fed today?"
        
        user_query = "Did anything happen with the Fed today?"
        context = "Previous discussion about USD conversion timing"
        
        decision = await AgentScrapingInterface.should_i_scrape(user_query, context)
        
        assert decision['should_scrape'] == True
        assert 'today' in decision['reason'] or 'latest' in decision['reason']
    
    async def test_user_challenges_info(self):
        """
        Scenario: User challenges provided information
        """
        user_query = "I heard that Remitly changed their fees recently"
        context = "Agent recommended Remitly as cost-effective option"
        
        decision = await AgentScrapingInterface.should_i_scrape(user_query, context)
        
        assert decision['should_scrape'] == True
        assert 'provider' in decision['reason'] or 'gap' in decision['reason']
    
    async def test_no_scraping_for_general_education(self):
        """
        Scenario: User asks educational questions
        """
        educational_queries = [
            "How do exchange rates work?",
            "What affects currency values?", 
            "Explain inflation impact on currency",
        ]
        
        for query in educational_queries:
            decision = await AgentScrapingInterface.should_i_scrape(query)
            assert decision['should_scrape'] == False
    
    async def test_specific_url_override(self):
        """
        Test manual URL specification overrides decision engine
        """
        # Even for general query, manual URLs should work
        query = "General currency info"
        manual_urls = ["https://www.xe.com/currencyconverter/"]
        
        # This would test the scraping with manual URLs
        # (commenting out actual scraping to avoid network calls in tests)
        # info = await AgentScrapingInterface.get_current_info(query, manual_urls=manual_urls)
        # assert info['sources_checked'] == 1

@pytest.mark.asyncio
class TestErrorHandlingFlows:
    """Test error handling in conversational scenarios"""
    
    async def test_network_failure_graceful_degradation(self):
        """Test graceful handling when scraping fails"""
        # Use invalid URL to simulate network failure
        invalid_urls = ["https://nonexistent-site-12345.com"]
        
        info = await AgentScrapingInterface.get_current_info(
            "test query", 
            specific_urls=invalid_urls
        )
        
        assert info['success'] == False
        assert len(info['errors']) > 0
        assert 'Failed' in str(info['errors']) or 'Error' in str(info['errors'])
    
    async def test_partial_success_handling(self):
        """Test when some URLs work and others fail"""
        # Mix of valid and invalid URLs
        mixed_urls = [
            "https://www.xe.com/currencyconverter/",
            "https://invalid-url-12345.com"
        ]
        
        info = await AgentScrapingInterface.get_current_info(
            "test mixed sources",
            specific_urls=mixed_urls
        )
        
        # Should indicate partial success
        assert 'errors' in info
        assert info['sources_checked'] == 2

# Utility test for quick verification
async def quick_decision_test():
    """Quick test of core decision-making functionality"""
    print("=== Quick Decision Tests ===")
    
    test_cases = [
        ("What's the latest USD rate?", True, "time-sensitive"),
        ("How do rates work?", False, "educational"),
        ("Wise vs Remitly costs", True, "provider-specific"),
        ("I heard ECB announced today", True, "recent-event"),
    ]
    
    for query, expected_scrape, category in test_cases:
        decision = await AgentScrapingInterface.should_i_scrape(query)
        result = "✅" if decision['should_scrape'] == expected_scrape else "❌"
        print(f"{result} {category}: '{query}' -> {decision['should_scrape']}")

if __name__ == "__main__":
    # Run quick decision test
    asyncio.run(quick_decision_test())
    
    # Run pytest
    pytest.main([__file__, "-v"])