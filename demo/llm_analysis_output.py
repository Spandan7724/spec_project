#!/usr/bin/env python3
"""
Demo: Live LLM Analysis Output for Multi-Agent System
Shows actual LLM responses from the language model integration system
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.manager import LLMManager
from src.llm.config import LLMConfig

async def get_live_llm_analysis_output():
    """
    Get live LLM analysis output from the actual language model system.
    This demonstrates what the LLMManager actually returns.
    """
    
    print("Initializing LLM analysis system...")
    
    try:
        # Load LLM configuration
        llm_config = LLMConfig.from_yaml("config.yaml")
        
        # Initialize LLM manager with multi-provider support
        llm_manager = LLMManager(llm_config)
        
        # Test different types of analysis requests
        analysis_requests = [
            {
                "type": "market_analysis",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a currency market analyst. Analyze the current USD/EUR market conditions based on provided data."
                    },
                    {
                        "role": "user", 
                        "content": "Current USD/EUR rate is 0.9245. Recent economic events include ECB rate decision and US NFP data. Provide market analysis and direction bias."
                    }
                ]
            },
            {
                "type": "risk_assessment", 
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial risk analyst. Assess currency conversion timing risks."
                    },
                    {
                        "role": "user",
                        "content": "A client wants to convert $100,000 to EUR. Current volatility is high with Fed meeting next week. Assess timing risks and provide recommendation."
                    }
                ]
            },
            {
                "type": "news_interpretation",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a financial news analyst. Interpret news impact on currency markets."
                    },
                    {
                        "role": "user",
                        "content": "Latest headlines: 'ECB signals dovish stance', 'US inflation data beats expectations', 'Geopolitical tensions ease'. Analyze USD/EUR impact."
                    }
                ]
            }
        ]
        
        responses = []
        
        for request in analysis_requests:
            print(f"Processing {request['type']} request...")
            
            try:
                response = await llm_manager.chat(
                    messages=request["messages"],
                    tools=None  # No tools for these simple analysis requests
                )
                
                responses.append({
                    "request_type": request["type"],
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error with {request['type']}: {e}")
                responses.append({
                    "request_type": request["type"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return responses
        
    except Exception as e:
        print(f"Error initializing LLM system: {e}")
        print("This is expected if API keys are not configured.")
        return []

def format_for_agents(llm_responses: list) -> dict:
    """
    Format the LLM responses for consumption by the multi-agent system.
    This is the standardized format agents will receive.
    """
    
    if not llm_responses:
        return {
            "data_type": "llm_analysis",
            "timestamp": datetime.now().isoformat(),
            "responses_generated": 0,
            "error": "No LLM responses generated"
        }
    
    # Extract successful responses
    successful_responses = [r for r in llm_responses if "response" in r]
    failed_responses = [r for r in llm_responses if "error" in r]
    
    # Analyze provider performance
    provider_stats = {}
    for response in successful_responses:
        if response.get("response") and hasattr(response["response"], 'provider'):
            provider = response["response"].provider
            if provider not in provider_stats:
                provider_stats[provider] = {"successful": 0, "failed": 0}
            provider_stats[provider]["successful"] += 1
    
    for response in failed_responses:
        # Would need to track which provider failed, for now assume default
        provider = "unknown"
        if provider not in provider_stats:
            provider_stats[provider] = {"successful": 0, "failed": 0}
        provider_stats[provider]["failed"] += 1
    
    agent_data = {
        "data_type": "llm_analysis",
        "timestamp": datetime.now().isoformat(),
        "system_status": {
            "total_requests": len(llm_responses),
            "successful_responses": len(successful_responses), 
            "failed_responses": len(failed_responses),
            "success_rate": len(successful_responses) / len(llm_responses) if llm_responses else 0,
            "provider_stats": provider_stats
        },
        
        # Analysis results by category
        "analysis_results": {
            response["request_type"]: {
                "content": response["response"].content if "response" in response else None,
                "model": response["response"].model if "response" in response else None,
                "provider": response["response"].provider if "response" in response else None,
                "usage": response["response"].usage if "response" in response else None,
                "timestamp": response["timestamp"],
                "success": "response" in response,
                "error": response.get("error")
            }
            for response in llm_responses
        },
        
        # Aggregated insights from all analyses
        "market_insights": {
            "consensus_themes": [],  # Would extract common themes across responses
            "directional_bias": None,  # Would analyze sentiment across responses
            "confidence_level": "medium",  # Based on response consistency
            "key_factors": []  # Would extract key market factors mentioned
        },
        
        # Agent decision support
        "agent_context": {
            "llm_reliability": "high" if len(successful_responses) == len(llm_responses) else "medium" if len(successful_responses) > len(failed_responses) else "low",
            
            "analysis_coverage": list(set([r["request_type"] for r in successful_responses])),
            
            "recommendation_consistency": "consistent" if len(successful_responses) > 1 else "single_source",
            
            "provider_redundancy": len(provider_stats) > 1,
            
            "use_for_decisions": len(successful_responses) > 0,
            
            "fallback_needed": len(failed_responses) > 0,
            
            "next_analysis_due": datetime.now().isoformat(),  # Would calculate based on data freshness
            
            "context_limitations": [
                limitation for limitation, condition in [
                    ("limited_market_data", len(successful_responses) < 2),
                    ("provider_failures", len(failed_responses) > 0),
                    ("single_provider", len(provider_stats) <= 1),
                    ("analysis_incomplete", len(successful_responses) < len(llm_responses))
                ] if condition
            ]
        }
    }
    
    return agent_data

async def main():
    """Generate and display live LLM analysis output for agents"""
    
    print("=== Live LLM Analysis Output Demo for Multi-Agent System ===\n")
    
    # Get live responses from actual system
    llm_responses = await get_live_llm_analysis_output()
    
    if not llm_responses:
        print("No LLM responses generated. Check provider configuration.")
        return
    
    print(f"Generated {len(llm_responses)} LLM analysis responses")
    successful = len([r for r in llm_responses if "response" in r])
    failed = len([r for r in llm_responses if "error" in r])
    print(f"Success: {successful}, Failed: {failed}")
    print()
    
    # Show raw responses
    print("=== Raw LLM Responses ===")
    for i, response in enumerate(llm_responses, 1):
        print(f"\n{i}. {response['request_type'].upper()}")
        if "response" in response:
            print(f"   Provider: {response['response'].provider}")
            print(f"   Model: {response['response'].model}")
            print(f"   Content: {response['response'].content[:200]}...")
            if response['response'].usage:
                print(f"   Usage: {response['response'].usage}")
        else:
            print(f"   Error: {response.get('error', 'Unknown error')}")
    print()
    
    # Format for agents
    agent_data = format_for_agents(llm_responses)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Insights for Agent Decision Making ===")
    print(f"• LLM Reliability: {agent_data['agent_context']['llm_reliability']}")
    print(f"• Success Rate: {agent_data['system_status']['success_rate']:.1%}")
    print(f"• Analysis Coverage: {', '.join(agent_data['agent_context']['analysis_coverage'])}")
    print(f"• Provider Redundancy: {agent_data['agent_context']['provider_redundancy']}")
    print(f"• Use for Decisions: {agent_data['agent_context']['use_for_decisions']}")
    
    if agent_data['agent_context']['context_limitations']:
        print(f"• Limitations: {', '.join(agent_data['agent_context']['context_limitations'])}")

if __name__ == "__main__":
    asyncio.run(main())