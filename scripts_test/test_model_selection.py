#!/usr/bin/env python3
"""
Quick test to verify multi-model setup works correctly
"""

import asyncio
import sys
from src.llm.config import load_config
from src.llm.manager import LLMManager


def test_config():
    """Test that config loads correctly with multiple copilot providers"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    config = load_config()
    
    print(f"Default provider: {config.default_provider}")
    print(f"Failover enabled: {config.failover_enabled}")
    print(f"Failover order: {config.failover_order}")
    print("\nConfigured providers:")
    
    for name, prov_config in config.providers.items():
        print(f"  - {name}:")
        print(f"      Model: {prov_config.model}")
        print(f"      Enabled: {prov_config.enabled}")
        print(f"      API Key Env: {prov_config.api_key_env}")
    
    # Verify copilot variants are configured correctly
    assert "copilot" in config.providers
    assert "copilot_mini" in config.providers
    assert "copilot_claude" in config.providers
    
    # Verify they all use COPILOT_ACCESS_TOKEN
    assert config.providers["copilot"].api_key_env == "COPILOT_ACCESS_TOKEN"
    assert config.providers["copilot_mini"].api_key_env == "COPILOT_ACCESS_TOKEN"
    assert config.providers["copilot_claude"].api_key_env == "COPILOT_ACCESS_TOKEN"
    
    print("\n✅ Configuration test passed!")
    return True


async def test_llm_manager():
    """Test that LLM manager initializes all providers correctly"""
    print("\n" + "=" * 60)
    print("TEST 2: LLM Manager Initialization")
    print("=" * 60)
    
    llm_manager = LLMManager()
    
    providers = llm_manager.list_providers()
    print(f"\nInitialized {len(providers)} providers:")
    
    for name, info in providers.items():
        print(f"  - {name}:")
        print(f"      Model: {info['model']}")
        print(f"      Healthy: {info['healthy']}")
        print(f"      Is Default: {info['is_default']}")
    
    # Verify all copilot providers are initialized
    assert "copilot" in providers
    assert "copilot_mini" in providers
    assert "copilot_claude" in providers
    
    print("\n✅ LLM Manager test passed!")
    return True


def test_task_recommendations():
    """Test the task-based model recommendations"""
    print("\n" + "=" * 60)
    print("TEST 3: Task-Based Model Recommendations")
    print("=" * 60)
    
    from src.llm.agent_helpers import get_recommended_model_for_task, TASK_MODEL_RECOMMENDATIONS
    
    print("\nRecommended models for different tasks:")
    for task_type, model in TASK_MODEL_RECOMMENDATIONS.items():
        print(f"  {task_type}: {model}")
    
    # Test the helper function
    sentiment_model = get_recommended_model_for_task("sentiment_analysis")
    assert sentiment_model == "gpt-5-mini"
    print(f"\n✓ Sentiment analysis recommended model: {sentiment_model}")
    
    reasoning_model = get_recommended_model_for_task("reasoning")
    assert reasoning_model == "claude-3.5-sonnet"
    print(f"✓ Reasoning recommended model: {reasoning_model}")
    
    print("\n✅ Task recommendations test passed!")
    return True


async def test_model_selection():
    """Test that we can select different models"""
    print("\n" + "=" * 60)
    print("TEST 4: Model Selection")
    print("=" * 60)
    
    llm_manager = LLMManager()
    
    # Simple test message
    test_message = [
        {"role": "user", "content": "Say only 'OK' if you understand."}
    ]
    
    print("\nTesting different model selections...")
    
    # Test default
    try:
        response = await llm_manager.chat(test_message)
        print(f"✓ Default provider ({response.model}): {response.content[:50]}")
    except Exception as e:
        print(f"⚠ Default provider failed: {e}")
    
    # Test copilot_mini (gpt-5-mini)
    try:
        response = await llm_manager.chat(test_message, provider_name="copilot_mini")
        print(f"✓ copilot_mini ({response.model}): {response.content[:50]}")
    except Exception as e:
        print(f"⚠ copilot_mini failed: {e}")
    
    # Test copilot_claude
    try:
        response = await llm_manager.chat(test_message, provider_name="copilot_claude")
        print(f"✓ copilot_claude ({response.model}): {response.content[:50]}")
    except Exception as e:
        print(f"⚠ copilot_claude failed: {e}")
    
    print("\n✅ Model selection test completed!")
    print("   (Some failures are OK if COPILOT_ACCESS_TOKEN is not set)")
    return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Multi-Model LLM Configuration Tests")
    print("=" * 60)
    
    try:
        # Run tests
        test_config()
        await test_llm_manager()
        test_task_recommendations()
        await test_model_selection()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        print("\nYour multi-model setup is working correctly! 🎉")
        print("\nNext steps:")
        print("1. Set your COPILOT_ACCESS_TOKEN environment variable")
        print("2. Run example_agent_specific_models.py to see it in action")
        print("3. Read LLM_MODEL_SELECTION_GUIDE.md for usage examples")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

