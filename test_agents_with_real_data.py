"""
Test All Agents with Real Market Data Sources.

Tests the complete multi-agent system using real news, economic calendar,
market data, and provider rates instead of mock data.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agents.market_intelligence import MarketIntelligenceAgent
from core.agents.risk_analysis import RiskAnalysisAgent
from core.agents.cost_optimization import CostOptimizationAgent
from core.agents.decision_coordinator import DecisionCoordinator
from core.providers import ProviderManager, ProviderType
from core.workflows.state_management import CurrencyDecisionState, WorkflowStatus
from core.models import ConversionRequest, UserProfile, RiskTolerance
from core.data_sources.data_integration import get_real_market_intelligence_for_agents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_real_data_sources():
    """Test all real data sources independently."""
    print("🌐 Testing Real Data Sources...")
    print("=" * 50)
    
    try:
        # Test comprehensive real data
        real_data = await get_real_market_intelligence_for_agents("USD/EUR", 10000.0, 7)
        
        print(f"✅ News Articles: {len(real_data['news_data'])}")
        if real_data['news_data']:
            latest_news = real_data['news_data'][0]
            print(f"   Latest: {latest_news.get('title', '')[:50]}...")
            print(f"   Relevance: {latest_news.get('relevance_score', 0):.2f}")
        
        print(f"✅ Economic Events: {len(real_data['calendar_data'])}")
        if real_data['calendar_data']:
            next_event = real_data['calendar_data'][0]
            print(f"   Next: {next_event.get('name', '')} ({next_event.get('importance', '')})")
        
        print(f"✅ Price Data Points: {len(real_data['price_data'])}")
        if real_data['price_data']:
            current_price = real_data['price_data'][-1]
            volatility = real_data['market_context']['technical_indicators'].get('volatility')
            print(f"   Current Rate: {current_price:.4f}")
            print(f"   Volatility: {volatility:.3f}" if volatility else "   Volatility: N/A")
        
        print(f"✅ Provider Options: {len(real_data['provider_data'])}")
        if real_data['provider_data']:
            best_provider = min(real_data['provider_data'], key=lambda p: p.get('fee_percentage', 999))
            print(f"   Best Rate: {best_provider.get('name')} ({best_provider.get('fee_percentage', 0):.2f}%)")
        
        print(f"✅ Data Quality Score: {real_data['market_context']['data_quality']:.2f}")
        
        return real_data['market_context']['data_quality'] > 0.3  # Minimum quality threshold
        
    except Exception as e:
        print(f"❌ Real data sources test failed: {e}")
        return False


async def test_market_agent_with_real_data():
    """Test MarketIntelligenceAgent with real market data."""
    print("\n🧠 Testing Market Agent with Real Data...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        market_agent = MarketIntelligenceAgent("market_intelligence", provider_manager)
        
        # Get real market data
        real_data = await get_real_market_intelligence_for_agents("USD/EUR", 15000.0)
        
        # Enhanced request with real data context
        request_data = {
            "currency_pair": "USD/EUR",
            "amount": 15000.0,
            "timeframe_days": 7,
            "user_risk_tolerance": "moderate",
            "real_news_data": real_data['news_data'][:10],  # Top 10 articles
            "real_calendar_data": real_data['calendar_data'][:5],  # Next 5 events
            "real_price_data": real_data['price_data'][-30:] if real_data['price_data'] else [],  # Last 30 days
            "technical_indicators": real_data['market_context']['technical_indicators']
        }
        
        # Process with real data
        result = await market_agent.process_request(request_data)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Confidence: {result.confidence:.2f}")
        print(f"✅ Data Quality Impact: {'High' if real_data['market_context']['data_quality'] > 0.7 else 'Medium' if real_data['market_context']['data_quality'] > 0.4 else 'Low'}")
        
        if result.success and result.data:
            print(f"✅ Market Sentiment: {result.data.get('sentiment_score', 0):.2f}")
            print(f"✅ News Impact: {result.data.get('news_impact', 0):.2f}")
            print(f"✅ Market Regime: {result.data.get('market_regime', 'unknown')}")
            print(f"✅ Timing Recommendation: {result.data.get('timing_recommendation', 'unclear')}")
            
            # Show real data influence
            if real_data['news_data']:
                print(f"   📰 Based on {len(real_data['news_data'])} real news articles")
            if real_data['calendar_data']:
                print(f"   📅 Considering {len(real_data['calendar_data'])} upcoming economic events")
            if real_data['price_data']:
                print(f"   📈 Using {len(real_data['price_data'])} days of real price history")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Market agent with real data failed: {e}")
        return False


async def test_cost_agent_with_real_providers():
    """Test CostOptimizationAgent with real provider rates."""
    print("\n💰 Testing Cost Agent with Real Provider Data...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager)
        
        # Get real provider data
        real_data = await get_real_market_intelligence_for_agents("USD/EUR", 20000.0)
        
        # Enhanced request with real provider data
        request_data = {
            "currency_pair": "USD/EUR",
            "amount": 20000.0,
            "timeframe_days": 7,
            "user_fee_sensitivity": 0.8,
            "available_providers": real_data['provider_data'],
            "current_rates": {
                "USD/EUR": real_data['price_data'][-1] if real_data['price_data'] else 0.85,
                "market_rate": real_data['price_data'][-1] if real_data['price_data'] else 0.85
            }
        }
        
        # Process with real data
        result = await cost_agent.process_request(request_data)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Confidence: {result.confidence:.2f}")
        
        if result.success and result.data:
            print(f"✅ Best Provider: {result.data.get('best_provider', 'Unknown')}")
            print(f"✅ Estimated Cost: ${result.data.get('estimated_cost', 0):.2f}")
            print(f"✅ Cost Percentage: {result.data.get('cost_percentage', 0):.2f}%")
            print(f"✅ Potential Savings: ${result.data.get('potential_savings', 0):.2f}")
            
            # Show real provider influence
            providers_analyzed = len(result.data.get('provider_comparison', {}))
            print(f"   🏦 Analyzed {providers_analyzed} real provider options")
            
            if real_data['price_data']:
                current_rate = real_data['price_data'][-1]
                print(f"   📊 Using live market rate: {current_rate:.4f}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Cost agent with real data failed: {e}")
        return False


async def test_complete_workflow_with_real_data():
    """Test complete workflow using real data sources."""
    print("\n🌟 Testing Complete Workflow with Real Data...")
    print("=" * 60)
    
    try:
        # Setup all agents
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        market_agent = MarketIntelligenceAgent("market_intelligence", provider_manager)
        risk_agent = RiskAnalysisAgent("risk_analysis", provider_manager)
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager) 
        coordinator = DecisionCoordinator("decision_coordinator", provider_manager)
        
        # Create realistic conversion request
        user_profile = UserProfile(
            user_id="real_data_test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.6,
            typical_amount_usd=Decimal('25000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR",
            amount=Decimal('25000'),
            user_profile=user_profile,
            current_rate=Decimal('0.8520'),
            deadline=datetime.utcnow() + timedelta(days=10)
        )
        
        # Initialize workflow
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="real_data_workflow"
        )
        
        print(f"✅ Workflow: {workflow_state.workflow_id}")
        print(f"✅ Request: {workflow_state.request.currency_pair} ${float(workflow_state.request.amount):,.2f}")
        
        # Get real market intelligence once for all agents
        real_intelligence = await get_real_market_intelligence_for_agents(
            "USD/EUR", 25000.0, 10
        )
        
        print(f"✅ Real Data Quality: {real_intelligence['market_context']['data_quality']:.2f}")
        
        # Step 1: Market Analysis with Real Data
        print("\n📊 Market Analysis (Real Data)...")
        market_request = {
            "currency_pair": "USD/EUR",
            "amount": 25000.0,
            "timeframe_days": 10,
            "user_risk_tolerance": "moderate",
            "real_news_data": real_intelligence['news_data'],
            "real_calendar_data": real_intelligence['calendar_data'],
            "real_price_data": real_intelligence['price_data']
        }
        
        market_result = await market_agent.process_request(market_request)
        workflow_state.add_agent_result("market_intelligence", market_result)
        
        print(f"✅ Market Analysis: {market_result.success} (conf: {market_result.confidence:.2f})")
        
        # Step 2: Risk Analysis with Market Context
        print("\n⚠️  Risk Analysis (Real Context)...")
        risk_request = {
            "currency_pair": "USD/EUR", 
            "amount": 25000.0,
            "timeframe_days": 10,
            "user_risk_tolerance": "moderate",
            "has_deadline": True,
            "deadline_days": 10,
            "market_context": market_result.data if market_result.success else {},
            "price_data": real_intelligence['price_data']
        }
        
        risk_result = await risk_agent.process_request(risk_request)
        workflow_state.add_agent_result("risk_analysis", risk_result)
        
        print(f"✅ Risk Analysis: {risk_result.success} (conf: {risk_result.confidence:.2f})")
        
        # Step 3: Cost Analysis with Real Providers
        print("\n💸 Cost Analysis (Real Providers)...")
        cost_request = {
            "currency_pair": "USD/EUR",
            "amount": 25000.0,
            "timeframe_days": 10,
            "user_fee_sensitivity": 0.6,
            "available_providers": real_intelligence['provider_data'],
            "current_rates": {
                "USD/EUR": real_intelligence['price_data'][-1] if real_intelligence['price_data'] else 0.852,
                "market_rate": real_intelligence['price_data'][-1] if real_intelligence['price_data'] else 0.852
            },
            "market_context": market_result.data if market_result.success else {}
        }
        
        cost_result = await cost_agent.process_request(cost_request)
        workflow_state.add_agent_result("cost_optimization", cost_result)
        
        print(f"✅ Cost Analysis: {cost_result.success} (conf: {cost_result.confidence:.2f})")
        
        # Step 4: Final Coordination
        print("\n🎯 Decision Coordination...")
        coordination_request = {"workflow_state": workflow_state}
        coordination_result = await coordinator.process_request(coordination_request)
        
        print(f"✅ Coordination: {coordination_result.success} (conf: {coordination_result.confidence:.2f})")
        
        # Final Results
        print(f"\n📋 Final Results Using Real Data:")
        print("=" * 40)
        
        successful_agents = sum(1 for result in workflow_state.agent_results.values() if result.success)
        total_agents = len(workflow_state.agent_results)
        
        print(f"✅ Successful Agents: {successful_agents}/{total_agents}")
        
        if market_result.success:
            print(f"📊 Market Sentiment: {market_result.data.get('sentiment_score', 0):.2f} (real news analysis)")
            print(f"📊 Market Regime: {market_result.data.get('market_regime', 'unknown')} (real price data)")
        
        if risk_result.success:
            print(f"⚠️  Overall Risk: {risk_result.data.get('overall_risk', 0):.2f} (real volatility)")
            print(f"⚠️  User Alignment: {risk_result.data.get('user_risk_alignment', 0):.2f}")
        
        if cost_result.success:
            print(f"💰 Best Provider: {cost_result.data.get('best_provider', 'Unknown')} (real rates)")
            print(f"💰 Total Cost: ${cost_result.data.get('estimated_cost', 0):.2f}")
        
        if coordination_result.success:
            print(f"🎯 Final Decision: {coordination_result.data.get('final_decision', 'unknown')}")
            print(f"🎯 Overall Confidence: {coordination_result.confidence:.2f}")
        
        # Data source summary
        print(f"\n📈 Data Source Quality:")
        print(f"   News Coverage: {'✅ Good' if len(real_intelligence['news_data']) >= 3 else '⚠️  Limited'}")
        print(f"   Economic Events: {'✅ Good' if len(real_intelligence['calendar_data']) >= 2 else '⚠️  Limited'}")
        print(f"   Price History: {'✅ Good' if len(real_intelligence['price_data']) >= 20 else '⚠️  Limited'}")
        print(f"   Provider Coverage: {'✅ Good' if len(real_intelligence['provider_data']) >= 3 else '⚠️  Limited'}")
        
        return successful_agents >= 3  # At least 3/4 agents should succeed
        
    except Exception as e:
        print(f"❌ Real data workflow test failed: {e}")
        return False


async def test_data_freshness():
    """Test data freshness and update frequencies."""
    print("\n⏰ Testing Data Freshness...")
    print("=" * 50)
    
    try:
        from core.data_sources.data_integration import RealDataProvider
        
        # Initialize data provider
        provider = RealDataProvider(
            newsapi_key=os.getenv('NEWSAPI_KEY'),
            fred_api_key=os.getenv('FRED_API_KEY')
        )
        
        # First fetch
        start_time = datetime.utcnow()
        intelligence1 = await provider.get_comprehensive_market_intelligence("USD/EUR", 5000.0)
        first_fetch_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"✅ First Fetch Time: {first_fetch_time:.1f}s")
        print(f"✅ Data Quality: {intelligence1['data_quality_score']:.2f}")
        
        # Second fetch (should use cache)
        start_time = datetime.utcnow()
        intelligence2 = await provider.get_comprehensive_market_intelligence("USD/EUR", 5000.0)
        second_fetch_time = (datetime.utcnow() - start_time).total_seconds()
        
        print(f"✅ Cached Fetch Time: {second_fetch_time:.1f}s")
        print(f"✅ Cache Performance: {first_fetch_time / max(second_fetch_time, 0.1):.1f}x faster")
        
        # Check cache status
        cache_status = provider.get_cache_status()
        cached_sources = sum(1 for status in cache_status.values() if status.get('cached', False))
        
        print(f"✅ Cached Data Sources: {cached_sources}/4")
        
        return cached_sources >= 2  # At least 2 sources should be cached
        
    except Exception as e:
        print(f"❌ Data freshness test failed: {e}")
        return False


async def main():
    """Run all real data tests."""
    print("🧪 Real Data Integration Testing Suite")
    print("=" * 70)
    
    # Check API availability
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run agent tests")
        return
    
    test_results = {}
    
    # Test 1: Real Data Sources
    test_results["real_data_sources"] = await test_real_data_sources()
    
    # Test 2: Market Agent with Real Data
    test_results["market_agent_real_data"] = await test_market_agent_with_real_data()
    
    # Test 3: Cost Agent with Real Providers
    test_results["cost_agent_real_providers"] = await test_cost_agent_with_real_providers()
    
    # Test 4: Complete Workflow with Real Data
    test_results["complete_workflow_real_data"] = await test_complete_workflow_with_real_data()
    
    # Test 5: Data Freshness and Caching
    test_results["data_freshness"] = await test_data_freshness()
    
    # Summary
    print("\n📋 Real Data Test Summary")
    print("=" * 40)
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 4:  # Allow 1 failure
        print("🎉 Real Data Integration is working!")
        print("✨ Agents are now using live market intelligence!")
        print("\n📈 What's working with real data:")
        print("   📰 RSS news feeds from Reuters, Bloomberg, MarketWatch")
        print("   📅 Economic calendar events (FRED + mock EU/UK)")
        print("   📊 Yahoo Finance historical price data") 
        print("   🏦 Estimated provider rates based on public fee structures")
        print("   🧠 LLM analysis of real market conditions")
        
        print(f"\n🚀 Next Steps:")
        print("   - Add FRED API key for better US economic data")
        print("   - Add NewsAPI key for enhanced news coverage") 
        print("   - Implement LangGraph workflow orchestration")
        print("   - Create FastAPI endpoints for production use")
    else:
        print("⚠️  Real data integration needs improvement")
        print("   Check network connectivity and data source availability")


if __name__ == "__main__":
    asyncio.run(main())