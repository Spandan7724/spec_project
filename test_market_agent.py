"""
Test script for MarketIntelligenceAgent functionality.

Tests the market intelligence agent with mock data and real LLM analysis
to verify the agent works correctly with the market analysis tools.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from decimal import Decimal

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agents.market_intelligence import MarketIntelligenceAgent
from core.providers import ProviderManager, ProviderType
from core.tools.market_tools import create_market_toolkit, mock_market_data_for_testing
from core.workflows.state_management import CurrencyDecisionState
from core.models import ConversionRequest, UserProfile, RiskTolerance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_market_tools():
    """Test the market analysis tools independently."""
    print("🛠️  Testing Market Analysis Tools...")
    print("=" * 50)
    
    try:
        # Create market toolkit
        toolkit = create_market_toolkit()
        
        # Generate mock data
        mock_data = mock_market_data_for_testing("USD/EUR")
        
        # Test comprehensive analysis
        analysis = await toolkit.comprehensive_market_analysis(
            currency_pair="USD/EUR",
            price_data=mock_data["price_data"],
            news_data=mock_data["news_data"],
            calendar_data=mock_data["calendar_data"]
        )
        
        print(f"✅ Currency Pair: {analysis['currency_pair']}")
        print(f"✅ Overall Confidence: {analysis['overall_confidence']:.2f}")
        print(f"✅ News Items: {analysis['components']['news']['total_items']}")
        print(f"✅ Economic Events: {analysis['components']['economic_events']['total_events']}")
        print(f"✅ Market Regime: {analysis['components']['market_regime']['regime']}")
        print(f"✅ RSI Value: {analysis['components']['rsi']['value']:.1f}")
        print(f"✅ MA Signal: {analysis['components']['moving_averages']['signal']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market tools test failed: {e}")
        return False


async def test_market_intelligence_agent():
    """Test MarketIntelligenceAgent with mock data."""
    print("\n🧠 Testing MarketIntelligenceAgent...")
    print("=" * 50)
    
    try:
        # Setup provider
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(
            ProviderType.COPILOT, 
            "gpt-4o-2024-11-20",
            temperature=0.3,
            max_tokens=1000
        )
        
        # Create agent
        market_agent = MarketIntelligenceAgent(
            agent_name="market_intelligence",
            provider_manager=provider_manager
        )
        
        # Load configuration to avoid warnings
        try:
            from core.config.llm_config import load_llm_config
            config = load_llm_config()
            market_agent._config = config
        except Exception as e:
            logger.info(f"Using default configuration: {e}")
        
        # Test request data
        request_data = {
            "currency_pair": "USD/EUR",
            "amount": 10000.0,
            "timeframe_days": 7,
            "user_risk_tolerance": "moderate"
        }
        
        # Process request
        result = await market_agent.process_request(request_data)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Agent Name: {result.agent_name}")
        print(f"✅ Confidence: {result.confidence:.2f}")
        print(f"✅ Execution Time: {result.execution_time_ms}ms")
        
        if result.success and result.data:
            print(f"✅ Sentiment Score: {result.data.get('sentiment_score', 'N/A')}")
            print(f"✅ News Impact: {result.data.get('news_impact', 'N/A')}")
            print(f"✅ Market Regime: {result.data.get('market_regime', 'N/A')}")
            print(f"✅ Timing Rec: {result.data.get('timing_recommendation', 'N/A')}")
            print(f"✅ Economic Events: {len(result.data.get('economic_events', []))}")
            print(f"✅ Risk Factors: {len(result.data.get('risk_factors', []))}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ MarketIntelligenceAgent test failed: {e}")
        return False


async def test_agent_with_workflow_state():
    """Test agent integration with workflow state management."""
    print("\n🔗 Testing Agent with Workflow State...")
    print("=" * 50)
    
    try:
        # Create conversion request
        user_profile = UserProfile(
            user_id="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.5,
            typical_amount_usd=Decimal('10000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR",
            amount=Decimal('10000'),
            user_profile=user_profile,
            current_rate=Decimal('0.85')
        )
        
        # Create workflow state
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="test_market_workflow"
        )
        
        # Setup agent
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        market_agent = MarketIntelligenceAgent("market_intelligence", provider_manager)
        
        # Process request
        request_data = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 7,
            "user_risk_tolerance": workflow_state.request.user_profile.risk_tolerance.value
        }
        
        result = await market_agent.process_request(request_data)
        
        # Add to workflow state
        workflow_state.add_agent_result("market_intelligence", result)
        
        print(f"✅ Workflow ID: {workflow_state.workflow_id}")
        print(f"✅ Market Analysis Added: {'market_intelligence' in workflow_state.agent_results}")
        print(f"✅ Market Analysis Success: {workflow_state.agent_results['market_intelligence'].success}")
        
        if workflow_state.market_analysis:
            print(f"✅ Market Sentiment: {workflow_state.market_analysis.sentiment_score:.2f}")
            print(f"✅ Market Regime: {workflow_state.market_analysis.market_regime}")
            print(f"✅ Analysis Confidence: {workflow_state.market_analysis.confidence:.2f}")
        
        return workflow_state.agent_results["market_intelligence"].success
        
    except Exception as e:
        print(f"❌ Workflow state integration test failed: {e}")
        return False


async def test_sentiment_analysis():
    """Test specific sentiment analysis functionality."""
    print("\n😊 Testing Sentiment Analysis...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        market_agent = MarketIntelligenceAgent("market_intelligence", provider_manager)
        
        # Test sentiment analysis
        sentiment_result = await market_agent.analyze_currency_sentiment("USD/EUR", 24)
        
        print(f"✅ Sentiment Score: {sentiment_result['sentiment_score']}")
        print(f"✅ Confidence: {sentiment_result['confidence']:.2f}")
        print(f"✅ Reasoning Length: {len(sentiment_result['reasoning'])} characters")
        
        return sentiment_result['confidence'] > 0.0
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False


async def main():
    """Run all MarketIntelligenceAgent tests."""
    print("🧪 MarketIntelligenceAgent Testing Suite")
    print("=" * 60)
    
    # Check API key
    import os
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run LLM tests")
        print("ℹ️  Will run tool tests only...")
        
        # Run tool tests only
        tool_test = await test_market_tools()
        print(f"\n📋 Tool Test Results: {'✅ PASS' if tool_test else '❌ FAIL'}")
        return
    
    test_results = {}
    
    # Test 1: Market Tools
    test_results["market_tools"] = await test_market_tools()
    
    # Test 2: Market Intelligence Agent
    test_results["market_agent"] = await test_market_intelligence_agent()
    
    # Test 3: Workflow State Integration
    test_results["workflow_integration"] = await test_agent_with_workflow_state()
    
    # Test 4: Sentiment Analysis
    test_results["sentiment_analysis"] = await test_sentiment_analysis()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 MarketIntelligenceAgent is working correctly!")
        print("Next step: Implement RiskAnalysisAgent")
    else:
        print("⚠️  Some tests failed - check the implementation")


if __name__ == "__main__":
    asyncio.run(main())