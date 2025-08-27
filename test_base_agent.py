"""
Test script for BaseAgent functionality.

Tests the base agent framework with a simple echo agent to verify
that the foundation works correctly before building specialized agents.
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

from core.agents.base_agent import BaseAgent, AgentResult
from core.providers import ProviderManager, ProviderType
from core.workflows.state_management import CurrencyDecisionState
from core.models import ConversionRequest, UserProfile, RiskTolerance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EchoAgent(BaseAgent):
    """Simple echo agent for testing the BaseAgent framework."""
    
    def get_system_prompt(self) -> str:
        """System prompt for the echo agent."""
        return """You are a simple Echo Agent for testing purposes.
        
        Your role is to:
        1. Acknowledge the user's input
        2. Provide a brief analysis of the request
        3. Return structured data showing you processed the request
        
        Always respond professionally and include some analysis to demonstrate LLM reasoning.
        """
    
    async def process_request(self, request_data: Dict[str, Any]) -> AgentResult:
        """Process request by echoing back structured information."""
        try:
            # Create user message from request data
            user_message = f"""
            Please analyze this currency conversion request:
            - Currency Pair: {request_data.get('currency_pair', 'Unknown')}
            - Amount: {request_data.get('amount', 'Unknown')}
            - User Risk Tolerance: {request_data.get('risk_tolerance', 'Unknown')}
            - Has Deadline: {request_data.get('has_deadline', False)}
            
            Provide a brief analysis and respond with structured insights.
            """
            
            # Get LLM response
            response = await self.chat_with_llm(user_message, include_tools=False)
            
            # Structure the result data
            result_data = {
                "input_analysis": {
                    "currency_pair": request_data.get('currency_pair'),
                    "amount": request_data.get('amount'),
                    "risk_tolerance": request_data.get('risk_tolerance'),
                    "has_deadline": request_data.get('has_deadline')
                },
                "llm_response": response.content,
                "processing_successful": True,
                "tokens_used": response.usage.get('total_tokens') if response.usage else 0
            }
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data=result_data,
                reasoning=f"Successfully processed request for {request_data.get('currency_pair')} conversion",
                confidence=0.95,  # High confidence for simple echo task
                execution_time_ms=getattr(response, 'execution_time_ms', None)
            )
            
        except Exception as e:
            logger.error(f"EchoAgent failed to process request: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                reasoning=f"Failed to process request: {str(e)}",
                error_message=str(e),
                confidence=0.0
            )


async def test_base_agent_initialization():
    """Test basic agent initialization and configuration."""
    print("🔧 Testing BaseAgent Initialization...")
    print("=" * 50)
    
    try:
        # Initialize provider manager
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(
            ProviderType.COPILOT, 
            "gpt-4o-2024-11-20",
            temperature=0.3,
            max_tokens=500
        )
        
        # Create echo agent
        echo_agent = EchoAgent(
            agent_name="test_echo",
            provider_manager=provider_manager,
            config_override={"temperature": 0.2, "max_tokens": 300}
        )
        
        print(f"✅ Agent Name: {echo_agent.agent_name}")
        print(f"✅ Configuration: temp={echo_agent.config.temperature}, max_tokens={echo_agent.config.max_tokens}")
        print(f"✅ Available Tools: {echo_agent.get_available_tools()}")
        print(f"✅ System Prompt Length: {len(echo_agent.get_system_prompt())} characters")
        
        return echo_agent
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None


async def test_agent_llm_interaction():
    """Test agent LLM interaction capabilities."""
    print("\n💬 Testing Agent LLM Interaction...")
    print("=" * 50)
    
    try:
        # Get agent from previous test
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(
            ProviderType.COPILOT, 
            "gpt-4o-2024-11-20",
            temperature=0.2
        )
        
        echo_agent = EchoAgent("test_echo", provider_manager)
        
        # Test simple chat
        response = await echo_agent.chat_with_llm(
            "Hello! This is a test message. Please acknowledge receipt.",
            include_tools=False
        )
        
        print(f"✅ LLM Response: {response.content[:100]}...")
        print(f"✅ Model Used: {response.model}")
        print(f"✅ Usage: {response.usage}")
        
        # Test conversation history
        history = echo_agent.get_conversation_history()
        print(f"✅ Conversation History: {len(history)} messages")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM interaction test failed: {e}")
        return False


async def test_agent_request_processing():
    """Test agent request processing with structured data."""
    print("\n🔄 Testing Agent Request Processing...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        echo_agent = EchoAgent("test_echo", provider_manager)
        
        # Create test request data
        test_request = {
            "currency_pair": "USD/EUR",
            "amount": 5000.0,
            "risk_tolerance": "moderate",
            "has_deadline": True,
            "deadline_days": 7
        }
        
        # Process the request
        result = await echo_agent.process_request(test_request)
        
        print(f"✅ Processing Success: {result.success}")
        print(f"✅ Agent Name: {result.agent_name}")
        print(f"✅ Confidence: {result.confidence}")
        print(f"✅ Reasoning: {result.reasoning}")
        print(f"✅ Data Keys: {list(result.data.keys()) if result.data else []}")
        print(f"✅ Execution Time: {result.execution_time_ms}ms")
        
        if result.data:
            input_analysis = result.data.get("input_analysis", {})
            print(f"✅ Processed Currency Pair: {input_analysis.get('currency_pair')}")
            print(f"✅ Processed Amount: {input_analysis.get('amount')}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Request processing test failed: {e}")
        return False


async def test_agent_health_check():
    """Test agent health check functionality."""
    print("\n🏥 Testing Agent Health Check...")
    print("=" * 50)
    
    try:
        # Setup
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        echo_agent = EchoAgent("test_echo", provider_manager)
        
        # Run health check
        health_status = await echo_agent.health_check()
        
        print(f"✅ Agent Name: {health_status.get('agent_name')}")
        print(f"✅ LLM Healthy: {health_status.get('llm_healthy')}")
        print(f"✅ Tools Count: {health_status.get('tools_count')}")
        print(f"✅ Configuration Loaded: {health_status.get('configuration_loaded')}")
        print(f"✅ Total Requests: {health_status.get('total_requests')}")
        print(f"✅ Success Rate: {health_status.get('success_rate', 0):.2%}")
        
        return health_status.get('llm_healthy', False)
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False


async def test_workflow_state_integration():
    """Test integration with workflow state management."""
    print("\n🔗 Testing Workflow State Integration...")
    print("=" * 50)
    
    try:
        # Create a sample conversion request
        user_profile = UserProfile(
            user_id="test_user",
            risk_tolerance=RiskTolerance.MODERATE,
            fee_sensitivity=0.6,
            typical_amount_usd=Decimal('5000')
        )
        
        conversion_request = ConversionRequest(
            from_currency="USD",
            to_currency="EUR", 
            amount=Decimal('5000'),
            user_profile=user_profile,
            current_rate=Decimal('0.85')
        )
        
        # Create workflow state
        workflow_state = CurrencyDecisionState(
            request=conversion_request,
            workflow_id="test_workflow_001"
        )
        
        # Setup agent
        provider_manager = ProviderManager()
        await provider_manager.set_primary_provider(ProviderType.COPILOT, "gpt-4o-2024-11-20")
        
        echo_agent = EchoAgent("test_echo", provider_manager)
        
        # Process request using workflow state data
        request_data = {
            "currency_pair": workflow_state.request.currency_pair,
            "amount": float(workflow_state.request.amount),
            "risk_tolerance": workflow_state.request.user_profile.risk_tolerance.value,
            "has_deadline": workflow_state.request.deadline is not None
        }
        
        result = await echo_agent.process_request(request_data)
        
        # Add result to workflow state
        workflow_state.add_agent_result("test_echo", result)
        
        print(f"✅ Workflow ID: {workflow_state.workflow_id}")
        print(f"✅ Request Currency Pair: {workflow_state.request.currency_pair}")
        print(f"✅ Agent Results: {list(workflow_state.agent_results.keys())}")
        print(f"✅ Workflow Status: {workflow_state.status.value}")
        
        # Test execution summary
        summary = workflow_state.get_execution_summary()
        print(f"✅ Execution Summary: {summary['agents_executed']}")
        print(f"✅ Successful Agents: {summary['successful_agents']}")
        
        return len(workflow_state.agent_results) > 0
        
    except Exception as e:
        print(f"❌ Workflow state integration test failed: {e}")
        return False


async def main():
    """Run all base agent tests."""
    print("🧪 BaseAgent Framework Testing")
    print("=" * 60)
    
    # Check if API key is available
    import os
    if not os.getenv("COPILOT_ACCESS_TOKEN"):
        print("❌ COPILOT_ACCESS_TOKEN not found - cannot run tests")
        return
    
    test_results = {}
    
    # Test 1: Agent Initialization
    agent = await test_base_agent_initialization()
    test_results["initialization"] = agent is not None
    
    if not agent:
        print("❌ Cannot continue without successful initialization")
        return
    
    # Test 2: LLM Interaction
    test_results["llm_interaction"] = await test_agent_llm_interaction()
    
    # Test 3: Request Processing
    test_results["request_processing"] = await test_agent_request_processing()
    
    # Test 4: Health Check
    test_results["health_check"] = await test_agent_health_check()
    
    # Test 5: Workflow Integration
    test_results["workflow_integration"] = await test_workflow_state_integration()
    
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
        print("🎉 BaseAgent framework is working correctly!")
        print("Next step: Implement specialized agents")
    else:
        print("⚠️  Some tests failed - check the implementation")


if __name__ == "__main__":
    asyncio.run(main())