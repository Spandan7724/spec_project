"""
Test script for RiskAnalysisAgent functionality.

Tests the risk analysis agent with mock data and real LLM analysis
to verify comprehensive risk assessment capabilities.
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

from core.agents.risk_analysis import RiskAnalysisAgent
from core.providers import ProviderManager, ProviderType
from core.tools.risk_tools import create_risk_analysis_toolkit, mock_risk_data_for_testing
from core.workflows.state_management import CurrencyDecisionState
from core.models import ConversionRequest, UserProfile, RiskTolerance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_risk_analysis_tools():
    """Test the risk analysis tools independently."""
    print("⚠️  Testing Risk Analysis Tools...")
    print("=" * 50)
    
    try:
        # Create risk toolkit
        toolkit = create_risk_analysis_toolkit()
        
        # Generate mock data
        mock_data = mock_risk_data_for_testing("USD/EUR", "moderate")
        
        # Test comprehensive analysis
        analysis = await toolkit.comprehensive_risk_analysis(
            currency_pair="USD/EUR",
            amount=10000.0,
            user_risk_tolerance="moderate",
            price_data=mock_data["price_data"],
            market_context=mock_data["market_context"],
            deadline_days=mock_data["deadline_days"]
        )
        
        print(f"✅ Currency Pair: {analysis['currency_pair']}")
        print(f"✅ Overall Risk Score: {analysis['overall_risk_score']:.2f}")
        print(f"✅ Analysis Confidence: {analysis['confidence']:.2f}")
        print(f"✅ Volatility: {analysis['components']['volatility_analysis']['current_volatility']:.3f}")
        print(f"✅ VaR 7d: ${analysis['components']['risk_metrics']['value_at_risk_7d']:.2f}")
        print(f"✅ User Compatibility: {analysis['components']['user_compatibility']['overall_compatibility']:.2f}")
        print(f"✅ Time Pressure: {analysis['components']['time_analysis']['time_pressure']:.2f}")
        print(f"✅ Scenarios Generated: {len(analysis['components']['scenarios'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk tools test failed: {e}")
        return False


async def test_risk_analysis_agent():
    """Test RiskAnalysisAgent with mock data."""
    print("\n🎯 Testing RiskAnalysisAgent...")
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
        risk_agent = RiskAnalysisAgent(
            agent_name="risk_analysis",
            provider_manager=provider_manager
        )
        
        # Load configuration to avoid warnings
        try:
            from core.config.llm_config import load_llm_config
            config = load_llm_config()
            risk_agent._config = config
        except Exception as e:
            logger.info(f"Using default configuration: {e}")
        
        # Test request data with market context
        request_data = {
            "currency_pair": "USD/EUR",
            "amount": 15000.0,
            "timeframe_days": 7,
            "user_risk_tolerance": "moderate",
            "has_deadline": True,
            "deadline_days": 10,
            "market_context": {
                "sentiment_score": 0.2,
                "market_regime": "ranging",
                "news_impact": 0.6,
                "economic_events": [{"event": "GDP Release", "importance": "high"}]
            },
            "ml_predictions": {
                "confidence": 0.75,
                "confidence_interval": "0.84-0.86",
                "uncertainty": 0.25
            }
        }
        
        # Process request
        result = await risk_agent.process_request(request_data)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Agent Name: {result.agent_name}")
        print(f"✅ Confidence: {result.confidence:.2f}")
        print(f"✅ Execution Time: {result.execution_time_ms}ms")
        
        if result.success and result.data:
            print(f"✅ Volatility Score: {result.data.get('volatility_score', 'N/A')}")
            print(f"✅ Prediction Uncertainty: {result.data.get('prediction_uncertainty', 'N/A')}")
            print(f"✅ Time Risk: {result.data.get('time_risk', 'N/A')}")
            print(f"✅ User Alignment: {result.data.get('user_risk_alignment', 'N/A')}")
            print(f"✅ Overall Risk: {result.data.get('overall_risk', 'N/A')}")
            print(f"✅ Scenarios: {len(result.data.get('scenarios', []))}")
            print(f"✅ Risk Factors: {len(result.data.get('risk_factors', []))}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ RiskAnalysisAgent test failed: {e}")
        return False


async def test_agent_with_workflow_state():
    """Test risk agent integration with workflow state management."""
    print("\n🔗 Testing Risk Agent with Workflow State...")
    print("=" * 50)
    
    try:
        # Create conversion request with deadline
        user_profile = UserProfile(
            user_id="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.4,
            typical_amount_usd=Decimal('15000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR",
            amount=Decimal('15000'),
            user_profile=user_profile,
            current_rate=Decimal('0.85'),
            deadline=datetime.utcnow() + timedelta(days=10)
        )
        
        # Create workflow state
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="test_risk_workflow"
        )
        
        # Setup agent
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        risk_agent = RiskAnalysisAgent("risk_analysis", provider_manager)
        
        # Process request with deadline information
        deadline_days = (conversion_request.deadline - datetime.utcnow()).days if conversion_request.deadline else None
        
        request_data = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 7,
            "user_risk_tolerance": workflow_state.request.user_profile.risk_tolerance.value,
            "has_deadline": conversion_request.deadline is not None,
            "deadline_days": deadline_days
        }
        
        result = await risk_agent.process_request(request_data)
        
        # Add to workflow state
        workflow_state.add_agent_result("risk_analysis", result)
        
        print(f"✅ Workflow ID: {workflow_state.workflow_id}")
        print(f"✅ Risk Analysis Added: {'risk_analysis' in workflow_state.agent_results}")
        print(f"✅ Risk Analysis Success: {workflow_state.agent_results['risk_analysis'].success}")
        print(f"✅ Has Deadline: {conversion_request.deadline is not None}")
        print(f"✅ Deadline Days: {deadline_days}")
        
        if workflow_state.risk_analysis:
            print(f"✅ Overall Risk: {workflow_state.risk_analysis.overall_risk:.2f}")
            print(f"✅ Volatility Score: {workflow_state.risk_analysis.volatility_score:.2f}")
            print(f"✅ Time Risk: {workflow_state.risk_analysis.time_risk:.2f}")
            print(f"✅ User Alignment: {workflow_state.risk_analysis.user_risk_alignment:.2f}")
        
        return workflow_state.agent_results["risk_analysis"].success
        
    except Exception as e:
        print(f"❌ Workflow state integration test failed: {e}")
        return False


async def test_volatility_analysis():
    """Test specific volatility analysis functionality."""
    print("\n📈 Testing Volatility Analysis...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        risk_agent = RiskAnalysisAgent("risk_analysis", provider_manager)
        
        # Test volatility analysis
        volatility_result = await risk_agent.analyze_volatility_impact("USD/EUR", 7, 10000.0)
        
        print(f"✅ Volatility Score: {volatility_result['volatility_score']}")
        print(f"✅ Potential Impact: ${volatility_result['potential_impact']:.2f}")
        print(f"✅ Confidence: {volatility_result['confidence']:.2f}")
        print(f"✅ Reasoning Length: {len(volatility_result['reasoning'])} characters")
        
        return volatility_result['confidence'] > 0.0
        
    except Exception as e:
        print(f"❌ Volatility analysis test failed: {e}")
        return False


async def test_user_risk_compatibility():
    """Test user risk tolerance compatibility assessment."""
    print("\n👤 Testing User Risk Compatibility...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        risk_agent = RiskAnalysisAgent("risk_analysis", provider_manager)
        
        # Mock market conditions
        market_conditions = {
            "volatility_score": 0.6,  # Moderate volatility
            "sentiment_score": -0.1,   # Slightly negative
            "news_impact": 0.5
        }
        
        # Test compatibility assessment
        compatibility_result = await risk_agent.assess_user_risk_compatibility(
            market_conditions=market_conditions, 
            user_risk_tolerance="conservative", 
            amount=10000.0
        )
        
        print(f"✅ Alignment Score: {compatibility_result['alignment_score']:.2f}")
        print(f"✅ Risk Tolerance: {compatibility_result['risk_tolerance']}")
        print(f"✅ Recommended Amount: ${compatibility_result['recommended_amount']:.2f}")
        print(f"✅ Confidence: {compatibility_result['confidence']:.2f}")
        
        return compatibility_result['confidence'] > 0.0
        
    except Exception as e:
        print(f"❌ User risk compatibility test failed: {e}")
        return False


async def main():
    """Run all RiskAnalysisAgent tests."""
    print("🧪 RiskAnalysisAgent Testing Suite")
    print("=" * 60)
    
    # Check API key
    import os
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run LLM tests")
        print("ℹ️  Will run tool tests only...")
        
        # Run tool tests only
        tool_test = await test_risk_analysis_tools()
        print(f"\n📋 Tool Test Results: {'✅ PASS' if tool_test else '❌ FAIL'}")
        return
    
    test_results = {}
    
    # Test 1: Risk Analysis Tools
    test_results["risk_tools"] = await test_risk_analysis_tools()
    
    # Test 2: Risk Analysis Agent
    test_results["risk_agent"] = await test_risk_analysis_agent()
    
    # Test 3: Workflow State Integration
    test_results["workflow_integration"] = await test_agent_with_workflow_state()
    
    # Test 4: Volatility Analysis
    test_results["volatility_analysis"] = await test_volatility_analysis()
    
    # Test 5: User Risk Compatibility
    test_results["user_compatibility"] = await test_user_risk_compatibility()
    
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
        print("🎉 RiskAnalysisAgent is working correctly!")
        print("Next step: Implement CostOptimizationAgent")
    else:
        print("⚠️  Some tests failed - check the implementation")


if __name__ == "__main__":
    asyncio.run(main())