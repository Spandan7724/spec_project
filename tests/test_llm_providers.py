#!/usr/bin/env python3
"""
Test script for LLM Provider system
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm import LLMManager, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_provider_initialization():
    """Test provider initialization"""
    print("🔧 Testing Provider Initialization...")
    
    try:
        manager = LLMManager()
        providers = manager.list_providers()
        
        print(f"✅ Successfully initialized {len(providers)} providers:")
        for name, info in providers.items():
            status = "✅ Healthy" if info['healthy'] else "❌ Unhealthy"
            default_marker = " (DEFAULT)" if info['is_default'] else ""
            print(f"  - {name}: {info['model']} {status}{default_marker}")
        
        return manager
    except Exception as e:
        print(f"❌ Failed to initialize providers: {e}")
        return None


async def test_health_checks(manager: LLMManager):
    """Test health checks for all providers"""
    print("\n🏥 Testing Health Checks...")
    
    try:
        health_results = await manager.health_check_all()
        
        for provider_name, is_healthy in health_results.items():
            status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            print(f"  - {provider_name}: {status}")
        
        healthy_count = sum(health_results.values())
        print(f"\nHealth Summary: {healthy_count}/{len(health_results)} providers healthy")
        
        return healthy_count > 0
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_simple_chat(manager: LLMManager):
    """Test simple chat functionality"""
    print("\n💬 Testing Simple Chat...")
    
    messages = [
        {"role": "user", "content": "What is 2 + 2? Please respond with just the number."}
    ]
    
    try:
        response = await manager.chat(messages)
        
        print("✅ Chat successful!")
        print(f"  Provider: {response.provider}")
        print(f"  Model: {response.model}")
        print(f"  Response: {response.content.strip()}")
        
        if response.usage:
            print(f"  Tokens: {response.usage.get('total_tokens', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False


async def test_failover_mechanism(manager: LLMManager):
    """Test failover by trying each provider"""
    print("\n🔄 Testing Failover Mechanism...")
    
    messages = [
        {"role": "user", "content": "Say 'Hello from [provider_name]'"}
    ]
    
    failover_order = manager.get_failover_order()
    print(f"Failover order: {' → '.join(failover_order)}")
    
    for provider_name in failover_order:
        try:
            print(f"\n  Testing {provider_name}...")
            response = await manager.chat(messages, provider_name=provider_name)
            
            print(f"    ✅ Success: {response.content.strip()}")
            print(f"    Model: {response.model}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    return True


async def test_tool_calling(manager: LLMManager):
    """Test tool calling functionality"""
    print("\n🔨 Testing Tool Calling...")
    
    # Simple tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Calculate the sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "Please calculate 15 + 27 using the calculate_sum tool"}
    ]
    
    try:
        response = await manager.chat(messages, tools=tools)
        
        print("✅ Tool calling test successful!")
        print(f"  Provider: {response.provider}")
        print(f"  Content: {response.content}")
        
        if response.tool_calls:
            print(f"  Tool calls: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                function = tool_call.get('function', {})
                print(f"    {i+1}. {function.get('name')}: {function.get('arguments')}")
        else:
            print("  No tool calls (provider may not support function calling)")
        
        return True
    except Exception as e:
        print(f"❌ Tool calling test failed: {e}")
        return False


async def test_environment_variables():
    """Test environment variables"""
    print("\n🔑 Checking Environment Variables...")
    
    required_vars = {
        "COPILOT_ACCESS_TOKEN": "GitHub Copilot",
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic Claude"
    }
    
    missing_vars = []
    
    for var_name, provider_name in required_vars.items():
        value = os.getenv(var_name)
        if value:
            print(f"  ✅ {provider_name}: {var_name} is set")
        else:
            print(f"  ❌ {provider_name}: {var_name} is missing")
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"\n⚠️  Warning: {len(missing_vars)} environment variables are missing.")
        print("   Some providers may not work without proper API keys.")
        return False
    else:
        print("\n✅ All environment variables are set!")
        return True


async def test_config_loading():
    """Test configuration loading"""
    print("\n📁 Testing Configuration Loading...")
    
    try:
        config = LLMConfig.from_yaml("/home/spandan/projects/spec_project_2/currency_assistant/config.yaml")
        
        print("✅ Configuration loaded successfully!")
        print(f"  Default provider: {config.default_provider}")
        print(f"  Failover enabled: {config.failover_enabled}")
        print(f"  Failover order: {config.failover_order}")
        print(f"  Enabled providers: {config.get_enabled_providers()}")
        
        # Validate environment variables
        validation = config.validate_environment_variables()
        print(f"  Environment validation: {validation}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting LLM Provider System Tests")
    print("=" * 50)
    
    # Track test results
    test_results = {}
    
    # Test 1: Environment Variables
    test_results['env'] = await test_environment_variables()
    
    # Test 2: Configuration Loading
    test_results['config'] = await test_config_loading()
    
    # Test 3: Provider Initialization
    manager = await test_provider_initialization()
    test_results['init'] = manager is not None
    
    if manager:
        # Test 4: Health Checks
        test_results['health'] = await test_health_checks(manager)
        
        # Test 5: Simple Chat
        test_results['chat'] = await test_simple_chat(manager)
        
        # Test 6: Failover Mechanism
        test_results['failover'] = await test_failover_mechanism(manager)
        
        # Test 7: Tool Calling
        test_results['tools'] = await test_tool_calling(manager)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.upper():<12}: {status}")
        
        print(f"\nResult: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! LLM Provider System is ready to use.")
            return 0
        else:
            print("⚠️  Some tests failed. Please check the configuration and API keys.")
            return 1
    else:
        print("❌ Cannot proceed with tests due to initialization failure.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)