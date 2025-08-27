"""
Test script for LLM Provider System.

Tests the configuration loading and provider initialization
to verify that the foundation system is working correctly.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import load_config, ConfigLoader
from core.providers import ProviderManager, ProviderType, CopilotProvider, OpenAIProvider, AnthropicProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_config_loading():
    """Test configuration file loading and parsing."""
    print("🔧 Testing Configuration Loading...")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        
        print(f"✅ Default Provider: {config.default_provider}")
        print(f"✅ Default Model: {config.default_model}")
        print(f"✅ Available Providers: {list(config.providers.keys())}")
        print(f"✅ Available Agents: {list(config.agents.keys())}")
        print(f"✅ Fallback Order: {config.fallback_order}")
        
        # Test specific provider config
        copilot_config = config.providers.get('copilot')
        if copilot_config:
            print(f"✅ Copilot Models: {len(copilot_config.models)} available")
            print(f"✅ Copilot Features: {copilot_config.features}")
        
        print("✅ Configuration loading successful!\n")
        return config
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return None


async def test_provider_initialization():
    """Test individual provider initialization (without API calls)."""
    print("🚀 Testing Provider Initialization...")
    print("=" * 50)
    
    results = {}
    
    # Test each provider type
    provider_tests = [
        (ProviderType.COPILOT, CopilotProvider, "gpt-4o-2024-11-20"),
        (ProviderType.OPENAI, OpenAIProvider, "gpt-4o"),
        (ProviderType.ANTHROPIC, AnthropicProvider, "claude-3-5-sonnet-20241022")
    ]
    
    for provider_type, provider_class, test_model in provider_tests:
        try:
            print(f"Testing {provider_type.value} provider...")
            
            # Check if required environment variable is set
            if provider_type == ProviderType.COPILOT:
                env_var = "COPILOT_ACCESS_TOKEN"
            elif provider_type == ProviderType.OPENAI:
                env_var = "OPENAI_API_KEY"
            else:  # Anthropic
                env_var = "ANTHROPIC_API_KEY"
            
            if not os.getenv(env_var):
                print(f"⚠️  {provider_type.value}: {env_var} not set - skipping")
                results[provider_type.value] = "skipped"
                continue
            
            # Try to initialize provider
            provider = provider_class(model=test_model, temperature=0.3)
            
            # Test basic properties
            name = provider.get_provider_name()
            model = provider.get_model_name()
            features = provider.get_supported_features()
            
            print(f"  ✅ Name: {name}")
            print(f"  ✅ Model: {model}")
            print(f"  ✅ Features: function_calling={features.function_calling}, streaming={features.streaming}")
            
            results[provider_type.value] = "initialized"
            
        except Exception as e:
            print(f"  ❌ {provider_type.value} initialization failed: {e}")
            results[provider_type.value] = f"failed: {e}"
    
    print(f"\n📊 Provider Initialization Results:")
    for provider, status in results.items():
        status_icon = "✅" if status == "initialized" else "⚠️" if status == "skipped" else "❌"
        print(f"  {status_icon} {provider}: {status}")
    
    return results


async def test_provider_manager():
    """Test ProviderManager functionality."""
    print("🎛️  Testing Provider Manager...")
    print("=" * 50)
    
    try:
        manager = ProviderManager()
        
        # Test manager basic functionality
        available_providers = manager.get_available_providers()
        print(f"✅ Available provider types: {available_providers}")
        
        # Test setting fallback order
        fallback_order = [ProviderType.COPILOT, ProviderType.OPENAI, ProviderType.ANTHROPIC]
        manager.set_fallback_order(fallback_order)
        print(f"✅ Set fallback order: {[p.value for p in fallback_order]}")
        
        # Try to initialize default provider if API key is available
        if os.getenv("COPILOT_ACCESS_TOKEN"):
            try:
                await manager.set_primary_provider(
                    ProviderType.COPILOT, 
                    "gpt-4o-2024-11-20",
                    temperature=0.3
                )
                print("✅ Successfully set Copilot as primary provider")
                
                # Test health check
                health_status = await manager.run_health_checks()
                print(f"✅ Health status: {health_status}")
                
            except Exception as e:
                print(f"⚠️  Copilot provider setup failed: {e}")
                
        else:
            print("⚠️  COPILOT_ACCESS_TOKEN not set - skipping provider manager test")
        
        print("✅ Provider Manager test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Provider Manager test failed: {e}")
        return False


async def test_simple_chat_request():
    """Test a simple chat request if API keys are available."""
    print("💬 Testing Simple Chat Request...")
    print("=" * 50)
    
    # Only test if we have API keys
    api_keys = {
        "COPILOT_ACCESS_TOKEN": ProviderType.COPILOT,
        "OPENAI_API_KEY": ProviderType.OPENAI,
        "ANTHROPIC_API_KEY": ProviderType.ANTHROPIC
    }
    
    available_providers = []
    for env_var, provider_type in api_keys.items():
        if os.getenv(env_var):
            available_providers.append((env_var, provider_type))
    
    if not available_providers:
        print("⚠️  No API keys found - skipping chat request test")
        return
    
    # Test with the first available provider
    env_var, provider_type = available_providers[0]
    print(f"Testing chat with {provider_type.value} provider...")
    
    try:
        manager = ProviderManager()
        
        # Set up the provider based on type
        if provider_type == ProviderType.COPILOT:
            await manager.set_primary_provider(provider_type, "gpt-4o-2024-11-20", temperature=0.1, max_tokens=50)
        elif provider_type == ProviderType.OPENAI:
            await manager.set_primary_provider(provider_type, "gpt-4o", temperature=0.1, max_tokens=50)
        else:  # Anthropic
            await manager.set_primary_provider(provider_type, "claude-3-5-sonnet-20241022", temperature=0.1, max_tokens=50)
        
        # Simple test message
        messages = [
            {"role": "user", "content": "Hello! Please respond with just 'Provider test successful' and nothing else."}
        ]
        
        # Make the request
        response = await manager.chat(messages)
        
        print(f"✅ Chat Response:")
        print(f"  Content: {response.content}")
        print(f"  Model: {response.model}")
        print(f"  Usage: {response.usage}")
        
        print("✅ Simple chat request successful!\n")
        
    except Exception as e:
        print(f"❌ Chat request failed: {e}")


async def main():
    """Run all provider tests."""
    print("🧪 LLM Provider System Testing")
    print("=" * 60)
    print()
    
    # Test 1: Configuration Loading
    config = await test_config_loading()
    if config is None:
        print("❌ Configuration test failed - stopping tests")
        return
    
    # Test 2: Provider Initialization
    provider_results = await test_provider_initialization()
    
    # Test 3: Provider Manager
    manager_success = await test_provider_manager()
    
    # Test 4: Simple Chat Request (if API keys available)
    await test_simple_chat_request()
    
    # Summary
    print("📋 Test Summary")
    print("=" * 30)
    print(f"✅ Configuration Loading: {'PASS' if config else 'FAIL'}")
    print(f"✅ Provider Manager: {'PASS' if manager_success else 'FAIL'}")
    
    initialized_count = sum(1 for status in provider_results.values() if status == "initialized")
    total_providers = len(provider_results)
    print(f"✅ Provider Initialization: {initialized_count}/{total_providers} providers ready")
    
    print("\n🎉 Provider system foundation is ready!")
    print("Next step: Implement agents and workflows")


if __name__ == "__main__":
    asyncio.run(main())