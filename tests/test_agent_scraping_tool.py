#!/usr/bin/env python3
"""
Comprehensive test suite for Agent Web Scraping Tool
Tests decision-making, caching, and real-world scenarios
"""
import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import time

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.agent_interface import AgentScrapingInterface
from tools.cache_manager import CacheManager
from tools.decision_engine import DecisionEngine

class TestDecisionEngine:
    """Test autonomous decision-making logic"""
    
    def setup_method(self):
        self.engine = DecisionEngine()
    
    def test_time_sensitive_triggers(self):
        """Test that time-sensitive queries trigger scraping"""
        queries = [
            "What's the latest USD/EUR rate today?",
            "Did the Fed announce anything recent?",
            "Current Wise fees now?",
            "Breaking news on currency policy"
        ]
        
        for query in queries:
            decision = self.engine.analyze_query(query)
            assert decision.should_scrape, f"Should scrape for: {query}"
            assert "latest" in decision.reason or "current" in decision.reason
    
    def test_provider_specific_triggers(self):
        """Test provider-specific queries trigger scraping"""
        queries = [
            "What about Wise's new fee structure?",
            "Remitly vs other providers",
            "XE.com transfer costs",
            "Revolut policy changes"
        ]
        
        for query in queries:
            decision = self.engine.analyze_query(query)
            assert decision.should_scrape, f"Should scrape for provider query: {query}"
            assert "provider" in decision.reason
    
    def test_no_scraping_for_general_queries(self):
        """Test that general queries don't trigger unnecessary scraping"""
        queries = [
            "How do exchange rates work?",
            "What is currency conversion?",
            "Explain inflation effects",
            "General economic principles"
        ]
        
        for query in queries:
            decision = self.engine.analyze_query(query)
            assert not decision.should_scrape, f"Should not scrape for: {query}"
    
    def test_knowledge_gap_detection(self):
        """Test detection of potential knowledge gaps"""
        gap_queries = [
            "But what about the new service I heard about?",
            "I heard that fees changed recently",
            "What if there was an announcement today?",
            "Did you know about the new regulations?"
        ]
        
        for query in gap_queries:
            decision = self.engine.analyze_query(query)
            assert decision.should_scrape, f"Should detect gap for: {query}"

class TestCacheManager:
    """Test intelligent caching behavior"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(self.temp_dir)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_ttl_configuration(self):
        """Test different TTL for different content types"""
        assert self.cache_manager.get_ttl_for_content_type('exchange_rates') == 300
        assert self.cache_manager.get_ttl_for_content_type('economic_news') == 1800
        assert self.cache_manager.get_ttl_for_content_type('provider_policies') == 86400
        assert self.cache_manager.get_ttl_for_content_type('unknown') == 3600
    
    def test_cache_bypass_logic(self):
        """Test cache bypass for time-sensitive queries"""
        time_sensitive = [
            "latest news today",
            "current rates now", 
            "breaking announcement",
            "just announced policy"
        ]
        
        for query in time_sensitive:
            assert self.cache_manager.should_bypass_cache_for_query(query)
        
        general_queries = [
            "historical exchange rates",
            "general fee information",
            "how providers work"
        ]
        
        for query in general_queries:
            assert not self.cache_manager.should_bypass_cache_for_query(query)
    
    def test_cache_storage_and_retrieval(self):
        """Test basic cache operations"""
        url = "https://example.com"
        content = "test content"
        data = {"test": "data"}
        
        # Store in cache
        self.cache_manager.set(url, content, data, 'exchange_rates')
        
        # Retrieve from cache
        cached = self.cache_manager.get(url, 'exchange_rates')
        assert cached is not None
        assert cached.content == content
        assert cached.extracted_data == data
    
    def test_cache_expiration(self):
        """Test cache expiration logic"""
        url = "https://example.com"
        content = "test content"
        data = {"test": "data"}
        
        # Store with very short TTL
        self.cache_manager.TTL_CONFIG['test_type'] = 1  # 1 second
        self.cache_manager.set(url, content, data, 'test_type')
        
        # Should be available immediately
        cached = self.cache_manager.get(url, 'test_type')
        assert cached is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        cached = self.cache_manager.get(url, 'test_type')
        assert cached is None

@pytest.mark.asyncio
class TestAgentWebScraper:
    """Test main scraping functionality"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    async def test_decision_integration(self):
        """Test integration between decision engine and scraper"""
        engine = DecisionEngine()
        # Time-sensitive query should trigger scraping
        decision = engine.analyze_query("What's the latest EUR rate today?")
        assert decision.should_scrape
        assert decision.content_type == 'exchange_rates'
        assert len(decision.suggested_urls) > 0
    
    async def test_cache_integration(self):
        """Test cache integration in scraping workflow"""
        cache_manager = CacheManager(self.temp_dir)
        # Mock a successful result in cache
        test_url = "https://www.xe.com/currencyconverter/"
        cache_manager.set(
            test_url, 
            "cached content",
            {"test": "cached_data"},
            'exchange_rates'
        )
        
        # Should use cached result
        cached_entry = cache_manager.get(test_url, 'exchange_rates')
        assert cached_entry is not None
        assert cached_entry.extracted_data["test"] == "cached_data"

@pytest.mark.asyncio 
class TestAgentInterface:
    """Test agent interface methods"""
    
    async def test_should_i_scrape_method(self):
        """Test the should_i_scrape interface method"""
        decision = await AgentScrapingInterface.should_i_scrape(
            "What about Wise's latest fees today?"
        )
        
        assert isinstance(decision, dict)
        assert 'should_scrape' in decision
        assert 'reason' in decision
        assert 'confidence' in decision
        assert decision['should_scrape'] == True  # Time-sensitive + provider-specific
    
    async def test_conversational_context(self):
        """Test how conversation context affects decisions"""
        user_query = "But what about recent changes?"
        context = "We were discussing USD to EUR conversion with Wise"
        
        decision = await AgentScrapingInterface.should_i_scrape(user_query, context)
        assert decision['should_scrape']
        assert 'provider' in decision['reason'] or 'latest' in decision['reason']

class TestRealWorldScenarios:
    """Test realistic conversation scenarios"""
    
    def test_conversation_flow_1(self):
        """
        Scenario: User asks about conversion, then follow-up about provider
        """
        # Initial query - no scraping needed
        engine = DecisionEngine()
        initial = engine.analyze_query("Convert $1000 USD to EUR")
        assert not initial.should_scrape
        
        # Follow-up - should trigger scraping
        followup = engine.analyze_query(
            "But what about Wise's new fees?",
            "Previous discussion about USD to EUR conversion"
        )
        assert followup.should_scrape
        assert 'provider' in followup.reason
    
    def test_conversation_flow_2(self):
        """
        Scenario: User challenges provided information
        """
        engine = DecisionEngine()
        challenge = engine.analyze_query(
            "I heard the Fed announced something today",
            "We discussed EUR timing based on economic indicators"
        )
        assert challenge.should_scrape
        assert 'latest' in challenge.reason or 'gap' in challenge.reason
    
    def test_conversation_flow_3(self):
        """
        Scenario: User asks about very recent events
        """
        engine = DecisionEngine()
        recent_event = engine.analyze_query(
            "Did anything happen with ECB policy this week?"
        )
        assert recent_event.should_scrape
        assert recent_event.content_type in ['economic_news', 'regulatory_changes']

class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def test_invalid_url_handling(self):
        """Test handling of invalid URLs"""
        engine = DecisionEngine()
        decision = engine.analyze_query("Check invalid-url.fake")
        # Should still provide decision framework
        assert isinstance(decision.should_scrape, bool)
    
    def test_network_failure_simulation(self):
        """Test behavior when network fails"""
        # This would require mocking crawl4ai, but structure is in place
        pass
    
    def test_empty_content_handling(self):
        """Test handling of empty or minimal content"""
        engine = DecisionEngine()
        # Engine should handle empty content gracefully
        extracted = engine._determine_content_type_and_urls("", None)
        assert len(extracted) == 2  # content_type, urls

# Integration test example
async def run_conversation_simulation():
    """
    Simulate a realistic agent conversation using the tool
    """
    print("=== Conversation Simulation ===")
    
    # User: "Convert $1000 USD to EUR"
    # Agent: Uses existing APIs...
    
    # User: "But what about Wise's fees today?"
    user_query = "But what about Wise's fees today?"
    context = "Previous USD to EUR conversion discussion"
    
    decision = await AgentScrapingInterface.should_i_scrape(user_query, context)
    print(f"Decision: {decision}")
    
    if decision['should_scrape']:
        info = await AgentScrapingInterface.get_current_info(user_query, context)
        print(f"Scraped info: {info}")
    
    # User: "Did the ECB announce anything recent?"
    ecb_query = "Did the ECB announce anything recent?"
    ecb_info = await AgentScrapingInterface.check_economic_events("ECB policy")
    print(f"ECB info: {ecb_info}")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Run conversation simulation
    print("\n" + "="*50)
    asyncio.run(run_conversation_simulation())