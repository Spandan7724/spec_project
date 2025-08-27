"""
Test script for complete multi-agent workflow integration.

Tests the full workflow with all agents working together to produce
a coordinated currency conversion decision.
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

from core.agents.market_intelligence import MarketIntelligenceAgent
from core.agents.risk_analysis import RiskAnalysisAgent
from core.agents.cost_optimization import CostOptimizationAgent
from core.agents.decision_coordinator import DecisionCoordinator
from core.providers import ProviderManager, ProviderType
from core.workflows.state_management import CurrencyDecisionState, WorkflowStatus
from core.models import ConversionRequest, UserProfile, RiskTolerance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_complete_workflow():
    """Test complete multi-agent workflow from start to finish."""
    print("🌟 Testing Complete Multi-Agent Workflow...")
    print("=" * 60)
    
    try:
        # Setup provider manager
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(
            ProviderType.COPILOT,
            "gpt-4o-2024-11-20",
            temperature=0.3,
            max_tokens=1500
        )
        
        # Create all agents
        market_agent = MarketIntelligenceAgent("market_intelligence", provider_manager)
        risk_agent = RiskAnalysisAgent("risk_analysis", provider_manager) 
        cost_agent = CostOptimizationAgent("cost_optimization", provider_manager)
        coordinator_agent = DecisionCoordinator("decision_coordinator", provider_manager)
        
        print("✅ All agents initialized successfully")
        
        # Create test conversion request
        user_profile = UserProfile(
            user_id="test_user_workflow",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.7,
            typical_amount_usd=Decimal('20000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR",
            amount=Decimal('20000'),
            user_profile=user_profile,
            current_rate=Decimal('0.8520'),
            deadline=datetime.utcnow() + timedelta(days=14)  # 2-week deadline
        )
        
        # Initialize workflow state
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="complete_workflow_test"
        )
        
        print(f"✅ Workflow initialized: {workflow_state.workflow_id}")
        print(f"✅ Request: {workflow_state.request.currency_pair} ${float(workflow_state.request.amount):,.2f}")
        
        # Step 1: Market Intelligence Analysis
        print("\n🧠 Step 1: Market Intelligence Analysis...")
        workflow_state.update_status(WorkflowStatus.MARKET_ANALYSIS, "market_intelligence")
        
        market_request = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 14,
            "user_risk_tolerance": workflow_state.request.user_profile.risk_tolerance.value
        }
        
        market_result = await market_agent.process_request(market_request)
        workflow_state.add_agent_result("market_intelligence", market_result)
        
        print(f"✅ Market analysis: {market_result.success} (confidence: {market_result.confidence:.2f})")
        if market_result.success:
            print(f"   - Sentiment: {market_result.data.get('sentiment_score', 0):.2f}")
            print(f"   - Regime: {market_result.data.get('market_regime', 'unknown')}")
            print(f"   - Timing: {market_result.data.get('timing_recommendation', 'unclear')}")
        
        # Step 2: Risk Analysis
        print("\n⚠️  Step 2: Risk Analysis...")
        workflow_state.update_status(WorkflowStatus.RISK_ANALYSIS, "risk_analysis")
        
        deadline_days = (conversion_request.deadline - datetime.utcnow()).days if conversion_request.deadline else None
        
        risk_request = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 14,
            "user_risk_tolerance": workflow_state.request.user_profile.risk_tolerance.value,
            "has_deadline": conversion_request.deadline is not None,
            "deadline_days": deadline_days,
            "market_context": market_result.data if market_result.success else {}
        }
        
        risk_result = await risk_agent.process_request(risk_request)
        workflow_state.add_agent_result("risk_analysis", risk_result)
        
        print(f"✅ Risk analysis: {risk_result.success} (confidence: {risk_result.confidence:.2f})")
        if risk_result.success:
            print(f"   - Overall risk: {risk_result.data.get('overall_risk', 0):.2f}")
            print(f"   - Volatility: {risk_result.data.get('volatility_score', 0):.2f}")
            print(f"   - Time risk: {risk_result.data.get('time_risk', 0):.2f}")
        
        # Step 3: Cost Optimization
        print("\n💰 Step 3: Cost Optimization...")
        workflow_state.update_status(WorkflowStatus.COST_ANALYSIS, "cost_optimization")
        
        cost_request = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "timeframe_days": 14,
            "user_fee_sensitivity": workflow_state.request.user_profile.fee_sensitivity,
            "available_providers": [
                {"name": "Wise", "fee_percentage": 0.35, "spread_bps": 15},
                {"name": "Revolut", "fee_percentage": 0.0, "spread_bps": 20},
                {"name": "Traditional Bank", "fee_percentage": 2.5, "spread_bps": 50}
            ],
            "current_rates": {"USD/EUR": 0.8520, "spread": 0.0012},
            "market_context": market_result.data if market_result.success else {}
        }
        
        cost_result = await cost_agent.process_request(cost_request)
        workflow_state.add_agent_result("cost_optimization", cost_result)
        
        print(f"✅ Cost analysis: {cost_result.success} (confidence: {cost_result.confidence:.2f})")
        if cost_result.success:
            print(f"   - Best provider: {cost_result.data.get('best_provider', 'Unknown')}")
            print(f"   - Est. cost: ${cost_result.data.get('estimated_cost', 0):.2f}")
            print(f"   - Savings: ${cost_result.data.get('potential_savings', 0):.2f}")
        
        # Step 4: Decision Coordination
        print("\n🎯 Step 4: Decision Coordination...")
        workflow_state.update_status(WorkflowStatus.COORDINATION, "decision_coordinator")
        
        coordination_request = {"workflow_state": workflow_state}
        coordination_result = await coordinator_agent.process_request(coordination_request)
        workflow_state.add_agent_result("decision_coordinator", coordination_result)
        
        print(f"✅ Coordination: {coordination_result.success} (confidence: {coordination_result.confidence:.2f})")
        if coordination_result.success:
            print(f"   - Final decision: {coordination_result.data.get('final_decision', 'unknown')}")
            print(f"   - Provider: {coordination_result.data.get('recommended_provider', 'Unknown')}")
            print(f"   - Amount: ${coordination_result.data.get('recommended_amount', 0):,.2f}")
        
        # Create final recommendation
        if coordination_result.success:
            final_recommendation = coordinator_agent._create_decision_recommendation(
                coordination_result.data, workflow_state
            )
            workflow_state.set_final_recommendation(final_recommendation, coordination_result.reasoning)
        
        # Workflow summary
        print("\n📋 Workflow Summary...")
        print("=" * 40)
        summary = workflow_state.get_execution_summary()
        
        print(f"✅ Workflow Status: {summary['status']}")
        print(f"✅ Total Execution Time: {summary['total_execution_time_ms']:.0f}ms")
        print(f"✅ Agents Executed: {len(summary['agents_executed'])}")
        print(f"✅ Successful Agents: {len(summary['successful_agents'])}")
        print(f"✅ Failed Agents: {len(summary['failed_agents'])}")
        print(f"✅ Has Final Recommendation: {summary['has_recommendation']}")
        print(f"✅ Final Confidence: {summary['confidence_score']:.2f}")
        
        if workflow_state.recommendation:
            print(f"\n🎯 Final Recommendation:")
            print(f"   Decision: {workflow_state.recommendation.decision.value}")
            print(f"   Provider: {workflow_state.recommendation.recommended_provider}")
            print(f"   Amount: ${workflow_state.recommendation.optimal_amount:.2f}")
            print(f"   Confidence: {workflow_state.recommendation.confidence:.2f}")
        
        return workflow_state.is_complete() and len(summary['failed_agents']) == 0
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False


async def test_agent_consensus():
    """Test agent consensus analysis."""
    print("\n🤝 Testing Agent Consensus Analysis...")
    print("=" * 50)
    
    try:
        # Create workflow state with mock agent results
        user_profile = UserProfile(
            user_id="consensus_test",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.5,
            typical_amount_usd=Decimal('10000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="GBP", 
            amount=Decimal('10000'),
            user_profile=user_profile,
            current_rate=Decimal('0.7450')
        )
        
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="consensus_test"
        )
        
        # Add mock agent results to test consensus
        from core.agents.base_agent import AgentResult
        
        # Mock market result (suggests waiting)
        market_result = AgentResult(
            agent_name="market_intelligence",
            success=True,
            data={"timing_recommendation": "wait_1_3_days", "sentiment_score": -0.2},
            confidence=0.8
        )
        
        # Mock risk result (moderate risk)
        risk_result = AgentResult(
            agent_name="risk_analysis", 
            success=True,
            data={"overall_risk": 0.4, "time_risk": 0.3},
            confidence=0.7
        )
        
        # Mock cost result (suggests immediate conversion)
        cost_result = AgentResult(
            agent_name="cost_optimization",
            success=True,
            data={"timing_recommendation": "immediate", "best_provider": "Wise"},
            confidence=0.9
        )
        
        workflow_state.add_agent_result("market_intelligence", market_result)
        workflow_state.add_agent_result("risk_analysis", risk_result)
        workflow_state.add_agent_result("cost_optimization", cost_result)
        
        # Test consensus analysis
        consensus = workflow_state.get_agent_consensus()
        
        print(f"✅ Consensus Score: {consensus['consensus_score']:.2f}")
        print(f"✅ Average Confidence: {consensus['average_confidence']:.2f}")
        print(f"✅ Participating Agents: {consensus['participating_agents']}")
        print(f"✅ Agreements: {len(consensus['agreements'])}")
        print(f"✅ Conflicts: {len(consensus['conflicts'])}")
        
        for agreement in consensus['agreements']:
            print(f"   ✅ Agreement: {agreement}")
        for conflict in consensus['conflicts']:
            print(f"   ⚠️  Conflict: {conflict}")
        
        return consensus['participating_agents'] == 3
        
    except Exception as e:
        print(f"❌ Agent consensus test failed: {e}")
        return False


async def main():
    """Run all multi-agent workflow tests."""
    print("🧪 Multi-Agent Workflow Testing Suite")
    print("=" * 70)
    
    # Check API key
    import os
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run workflow tests")
        return
    
    test_results = {}
    
    # Test 1: Agent Consensus Analysis
    test_results["agent_consensus"] = await test_agent_consensus()
    
    # Test 2: Complete Workflow
    test_results["complete_workflow"] = await test_complete_workflow()
    
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
        print("🎉 Multi-Agent Workflow is working correctly!")
        print("✨ All agents are successfully coordinated!")
        print("\n🚀 Next steps:")
        print("   - Implement LangGraph workflow orchestration")
        print("   - Add real data integration")
        print("   - Create FastAPI endpoints")
    else:
        print("⚠️  Some tests failed - check the implementation")


if __name__ == "__main__":
    asyncio.run(main())