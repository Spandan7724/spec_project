#!/usr/bin/env python3
"""
Example usage of the LLM Provider System
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm import LLMManager


async def basic_chat_example():
    """Basic chat example"""
    print("🤖 Basic Chat Example")
    print("-" * 30)
    
    # Initialize the manager (loads config.yaml automatically)
    manager = LLMManager()
    
    # Simple chat
    messages = [
        {"role": "user", "content": "Explain what a currency exchange rate is in one sentence."}
    ]
    
    try:
        response = await manager.chat(messages)
        
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        
        if response.usage:
            tokens = response.usage.get('total_tokens', 'unknown')
            print(f"Tokens used: {tokens}")
        
    except Exception as e:
        print(f"Error: {e}")


async def tool_calling_example():
    """Tool calling example"""
    print("\n🔨 Tool Calling Example")
    print("-" * 30)
    
    manager = LLMManager()
    
    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount from one currency to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "Amount to convert"},
                        "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
                        "to_currency": {"type": "string", "description": "Target currency code (e.g., EUR)"}
                    },
                    "required": ["amount", "from_currency", "to_currency"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "I want to convert 1000 USD to EUR. Please use the convert_currency tool."}
    ]
    
    try:
        response = await manager.chat(messages, tools=tools)
        
        print(f"Provider: {response.provider}")
        print(f"Response: {response.content}")
        
        if response.tool_calls:
            print("\nTool calls requested:")
            for i, tool_call in enumerate(response.tool_calls):
                function = tool_call.get('function', {})
                print(f"  {i+1}. Function: {function.get('name')}")
                print(f"     Arguments: {function.get('arguments')}")
        
    except Exception as e:
        print(f"Error: {e}")


async def provider_info_example():
    """Provider information example"""
    print("\n📊 Provider Information")
    print("-" * 30)
    
    manager = LLMManager()
    
    # List all providers
    providers = manager.list_providers()
    
    for name, info in providers.items():
        status = "✅ Healthy" if info['healthy'] else "❌ Unhealthy"
        default = " (DEFAULT)" if info['is_default'] else ""
        
        print(f"\nProvider: {name}{default}")
        print(f"  Model: {info['model']}")
        print(f"  Status: {status}")
        print(f"  Features: {list(info['features'].keys())}")
        
        if info['last_error']:
            print(f"  Last Error: {info['last_error']}")


async def failover_example():
    """Failover example"""
    print("\n🔄 Failover Example")
    print("-" * 30)
    
    manager = LLMManager()
    
    print("Failover order:", manager.get_failover_order())
    print("Healthy providers:", manager.get_healthy_providers())
    
    # This will automatically use the first healthy provider
    messages = [
        {"role": "user", "content": "What is the current time? (This is a test message)"}
    ]
    
    try:
        response = await manager.chat(messages)
        print(f"\nUsed provider: {response.provider}")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples"""
    print("🚀 LLM Provider System Examples")
    print("=" * 50)
    
    try:
        await basic_chat_example()
        await tool_calling_example()
        await provider_info_example()
        await failover_example()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())