"""
Test script for CostOptimizationAgent functionality.

Tests the cost optimization agent with mock provider data and real LLM analysis
to verify comprehensive cost analysis and optimization capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agents.cost_optimization import CostOptimizationAgent
from core.providers import ProviderManager, ProviderType
from core.tools.cost_tools import create_cost_optimization_toolkit, mock_cost_data_for_testing
from core.workflows.state_management import CurrencyDecisionState
from core.models import ConversionRequest, UserProfile, RiskTolerance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_cost_optimization_tools():
    """Test the cost optimization tools independently."""
    print("💰 Testing Cost Optimization Tools...")
    print("=" * 50)
    
    try:
        # Create cost toolkit
        toolkit = create_cost_optimization_toolkit()
        
        # Generate mock data
        mock_data = mock_cost_data_for_testing("USD/EUR", 10000.0)
        
        # Test comprehensive analysis
        analysis = await toolkit.comprehensive_cost_analysis(
            currency_pair="USD/EUR",
            amount=10000.0,
            available_providers=mock_data["available_providers"],
            market_rate=mock_data["market_rate"],
            market_context=mock_data["market_context"],
            user_preferences=mock_data["user_preferences"]
        )
        
        print(f"✅ Currency Pair: {analysis['currency_pair']}")
        print(f"✅ Amount: ${analysis['amount']:,.2f}")
        print(f"✅ Overall Confidence: {analysis['overall_confidence']:.2f}")
        print(f"✅ Recommended Provider: {analysis['summary']['recommended_provider']}")
        print(f"✅ Estimated Cost: ${analysis['summary']['estimated_total_cost']:.2f}")
        print(f"✅ Cost Percentage: {analysis['summary']['cost_as_percentage']:.2f}%")
        print(f"✅ Potential Savings: ${analysis['summary']['potential_savings']:.2f}")
        print(f"✅ Timing Rec: {analysis['summary']['timing_recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost optimization tools test failed: {e}")
        return False


async def test_cost_optimization_agent():
    """Test CostOptimizationAgent with mock data."""
    print("\n💸 Testing CostOptimizationAgent...")
    print("=" * 50)
    
    try:
        # Setup provider
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(
            ProviderType.COPILOT, 
            "gpt-4o-2024-11-20",
            temperature=0.2,
            max_tokens=1500
        )
        
        # Create agent
        cost_agent = CostOptimizationAgent(
            agent_name="cost_optimization",
            provider_manager=provider_manager
        )
        
        # Load configuration to avoid warnings
        try:
            from core.config.llm_config import load_llm_config
            config = load_llm_config()
            cost_agent._config = config
        except Exception as e:
            logger.info(f"Using default configuration: {e}")
        
        # Test request data with provider and market context
        request_data = {
            "currency_pair": "USD/EUR",
            "amount": 25000.0,
            "timeframe_days": 7,
            "user_fee_sensitivity": 0.8,  # High fee sensitivity
            "available_providers": [
                {"name": "Wise", "fee_percentage": 0.35, "spread_bps": 15, "transfer_speed": "24h"},
                {"name": "Traditional Bank", "fee_percentage": 2.5, "spread_bps": 50, "transfer_speed": "3-5 days"},
                {"name": "Revolut", "fee_percentage": 0.0, "spread_bps": 20, "transfer_speed": "12h"}
            ],
            "current_rates": {
                "USD/EUR": 0.8520,
                "spread": 0.0015
            },
            "market_context": {
                "market_regime": "ranging",
                "volatility_score": 0.4,
                "timing_recommendation": "wait_1_3_days"
            }
        }
        
        # Process request
        result = await cost_agent.process_request(request_data)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Agent Name: {result.agent_name}")
        print(f"✅ Confidence: {result.confidence:.2f}")
        print(f"✅ Execution Time: {result.execution_time_ms}ms")
        
        if result.success and result.data:
            print(f"✅ Best Provider: {result.data.get('best_provider', 'N/A')}")
            print(f"✅ Estimated Cost: ${result.data.get('estimated_cost', 0):.2f}")
            print(f"✅ Cost Percentage: {result.data.get('cost_percentage', 0):.2f}%")
            print(f"✅ Potential Savings: ${result.data.get('potential_savings', 0):.2f}")
            print(f"✅ Timing Impact: {result.data.get('timing_impact', 0):.2f}")
            print(f"✅ Providers Compared: {len(result.data.get('provider_comparison', {}))}")
            print(f"✅ Optimization Recs: {len(result.data.get('optimization_recommendations', []))}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ CostOptimizationAgent test failed: {e}")
        return False


async def test_agent_with_workflow_state():
    """Test cost agent integration with workflow state management.""" 
    print("\n🔗 Testing Cost Agent with Workflow State...")
    print("=" * 50)
    
    try:
        # Create conversion request
        user_profile = UserProfile(
            user_id="test_user",
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            fee_sensitivity=0.9,  # High fee sensitivity
            typical_amount_usd=Decimal('25000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR",
            amount=Decimal('25000'),
            user_profile=user_profile,
            current_rate=Decimal('0.852')
        )
        
        # Create workflow state
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="test_cost_workflow"
        )
        
        # Setup agent
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager)
        
        # Process request with fee sensitivity
        request_data = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 7,
            "user_fee_sensitivity": workflow_state.request.user_profile.fee_sensitivity,
            "available_providers": ["Wise", "Traditional Bank", "Revolut"],
            "current_rates": {"USD/EUR": 0.852, "spread": 0.0012}
        }
        
        result = await cost_agent.process_request(request_data)
        
        # Add to workflow state
        workflow_state.add_agent_result("cost_optimization", result)
        
        print(f"✅ Workflow ID: {workflow_state.workflow_id}")
        print(f"✅ Cost Analysis Added: {'cost_optimization' in workflow_state.agent_results}")
        print(f"✅ Cost Analysis Success: {workflow_state.agent_results['cost_optimization'].success}")
        print(f"✅ User Fee Sensitivity: {workflow_state.request.user_profile.fee_sensitivity}")
        
        if workflow_state.cost_analysis:
            print(f"✅ Best Provider: {workflow_state.cost_analysis.best_provider}")
            print(f"✅ Estimated Cost: ${workflow_state.cost_analysis.estimated_cost:.2f}")
            print(f"✅ Cost Percentage: {workflow_state.cost_analysis.cost_percentage:.2f}%")
            print(f"✅ Potential Savings: ${workflow_state.cost_analysis.potential_savings:.2f}")
        
        return workflow_state.agent_results["cost_optimization"].success
        
    except Exception as e:
        print(f"❌ Workflow state integration test failed: {e}")
        return False


async def test_provider_comparison():
    """Test provider comparison functionality."""
    print("\n🏦 Testing Provider Comparison...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager)
        
        # Mock provider data
        providers = [
            {"name": "Wise", "fee_percentage": 0.35, "spread_bps": 15},
            {"name": "Traditional Bank", "fee_percentage": 2.5, "spread_bps": 50},
            {"name": "Revolut", "fee_percentage": 0.0, "spread_bps": 20}
        ]
        
        # Test provider comparison
        comparison_result = await cost_agent.compare_providers("USD/EUR", 20000.0, providers)
        
        print(f"✅ Best Provider: {comparison_result['best_provider']}")
        print(f"✅ Providers Analyzed: {len(comparison_result['provider_comparison'])}")
        print(f"✅ Cost Range: ${comparison_result['total_cost_range']['min']:.2f} - ${comparison_result['total_cost_range']['max']:.2f}")
        print(f"✅ Confidence: {comparison_result['confidence']:.2f}")
        
        return comparison_result['confidence'] > 0.0
        
    except Exception as e:
        print(f"❌ Provider comparison test failed: {e}")
        return False


async def test_timing_impact_analysis():
    """Test timing impact analysis."""
    print("\n⏰ Testing Timing Impact Analysis...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager)
        
        # Test timing analysis
        timing_result = await cost_agent.analyze_timing_impact("USD/EUR", 15000.0, 3)
        
        print(f"✅ Timing Impact Score: {timing_result['timing_impact_score']:.2f}")
        print(f"✅ Potential Savings: ${timing_result['potential_savings']:.2f}")
        print(f"✅ Opportunity Cost: ${timing_result['opportunity_cost']:.2f}")
        print(f"✅ Net Benefit: ${timing_result['net_benefit']:.2f}")
        print(f"✅ Recommendation: {timing_result['recommendation']}")
        print(f"✅ Confidence: {timing_result['confidence']:.2f}")
        
        return timing_result['confidence'] > 0.0
        
    except Exception as e:
        print(f"❌ Timing impact analysis test failed: {e}")
        return False


async def main():
    """Run all CostOptimizationAgent tests."""
    print("🧪 CostOptimizationAgent Testing Suite")
    print("=" * 60)
    
    # Check API key
    import os
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run LLM tests")
        print("ℹ️  Will run tool tests only...")
        
        # Run tool tests only
        tool_test = await test_cost_optimization_tools()
        print(f"\n📋 Tool Test Results: {'✅ PASS' if tool_test else '❌ FAIL'}")
        return
    
    test_results = {}
    
    # Test 1: Cost Optimization Tools
    test_results["cost_tools"] = await test_cost_optimization_tools()
    
    # Test 2: Cost Optimization Agent
    test_results["cost_agent"] = await test_cost_optimization_agent()
    
    # Test 3: Workflow State Integration
    test_results["workflow_integration"] = await test_agent_with_workflow_state()
    
    # Test 4: Provider Comparison
    test_results["provider_comparison"] = await test_provider_comparison()
    
    # Test 5: Timing Impact Analysis
    test_results["timing_analysis"] = await test_timing_impact_analysis()
    
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
        print("🎉 CostOptimizationAgent is working correctly!")
        print("Next step: Create agent coordination workflow")
    else:
        print("⚠️  Some tests failed - check the implementation")


if __name__ == "__main__":
    asyncio.run(main())