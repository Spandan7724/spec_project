#!/usr/bin/env python3
"""
Demo: Live Exchange Rate Output for Multi-Agent System
Shows actual data from the exchange rate collection system
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.rate_collector import MultiProviderRateCollector
from src.data_collection.models import RateCollectionResult

async def get_live_exchange_rate_output():
    """
    Get live exchange rate output from the actual system components.
    This demonstrates what the MultiProviderRateCollector actually returns.
    """
    
    print("Collecting live exchange rate data...")
    
    try:
        # Initialize the multi-provider rate collector
        collector = MultiProviderRateCollector()
        
        # Collect rates for USD/EUR
        result = await collector.get_rate("USD", "EUR")
        
        return result
        
    except Exception as e:
        print(f"Error collecting live rates: {e}")
        print("This is expected if API keys are not configured.")
        return None

def format_for_agents(rate_result: RateCollectionResult) -> dict:
    """
    Format the exchange rate result for consumption by the multi-agent system.
    This is the standardized format agents will receive.
    """
    
    # Convert to agent-friendly format
    agent_data = {
        "data_type": "exchange_rates",
        "timestamp": rate_result.timestamp.isoformat(),
        "currency_pair": rate_result.currency_pair,
        "collection_success": rate_result.has_data,
        "success_rate_percent": rate_result.success_rate,
        
        # Best rate (primary recommendation)
        "recommended_rate": {
            "rate": rate_result.best_rate.rate if rate_result.best_rate else None,
            "source": rate_result.best_rate.source.value if rate_result.best_rate else None,
            "timestamp": rate_result.best_rate.timestamp.isoformat() if rate_result.best_rate else None,
            "bid": rate_result.best_rate.bid if rate_result.best_rate else None,
            "ask": rate_result.best_rate.ask if rate_result.best_rate else None,
            "spread_bps": rate_result.best_rate.spread_bps if rate_result.best_rate else None,
            "mid_rate": rate_result.best_rate.mid_rate if rate_result.best_rate else None
        },
        
        # All available rates for comparison
        "all_rates": [
            {
                "source": rate.source.value,
                "rate": rate.rate,
                "timestamp": rate.timestamp.isoformat(),
                "bid": rate.bid,
                "ask": rate.ask,
                "spread_bps": rate.spread_bps,
                "mid_rate": rate.mid_rate,
                "age_seconds": (rate_result.timestamp - rate.timestamp).total_seconds()
            }
            for rate in rate_result.rates
        ],
        
        # Market insights for agents
        "market_analysis": {
            "rate_spread": max([r.rate for r in rate_result.rates]) - min([r.rate for r in rate_result.rates]) if rate_result.rates else 0,
            "avg_rate": sum([r.rate for r in rate_result.rates]) / len(rate_result.rates) if rate_result.rates else 0,
            "provider_consensus": len(rate_result.rates) >= 2,
            "data_freshness_seconds": min([(rate_result.timestamp - r.timestamp).total_seconds() for r in rate_result.rates]) if rate_result.rates else None
        },
        
        # Reliability indicators
        "reliability": {
            "providers_responding": len(rate_result.rates),
            "total_providers": 3,  # We have 3 providers configured
            "has_bid_ask": any(r.bid is not None and r.ask is not None for r in rate_result.rates),
            "errors": rate_result.errors
        },
        
        # Agent decision support
        "agent_context": {
            "confidence_level": "high" if rate_result.success_rate >= 66.7 else "medium" if rate_result.success_rate >= 33.3 else "low",
            "use_for_decisions": rate_result.has_data and rate_result.success_rate >= 33.3,
            "requires_fallback": not rate_result.has_data or rate_result.success_rate < 33.3,
            "next_update_recommended": 300  # seconds
        }
    }
    
    return agent_data

async def main():
    """Generate and display live exchange rate output for agents"""
    
    print("=== Live Exchange Rate Output Demo for Multi-Agent System ===\n")
    
    # Get live data from actual system
    rate_result = await get_live_exchange_rate_output()
    
    if rate_result is None:
        print("Could not collect live data. Please check API configuration.")
        return
    
    print("Raw Data Structure (RateCollectionResult):")
    print(f"Currency Pair: {rate_result.currency_pair}")
    print(f"Best Rate: {rate_result.best_rate}")
    print(f"Success Rate: {rate_result.success_rate:.1f}%")
    print(f"Number of Rates: {len(rate_result.rates)}")
    print(f"Timestamp: {rate_result.timestamp}")
    print()
    
    # Format for agents
    agent_data = format_for_agents(rate_result)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Fields for Agent Decision Making ===")
    print(f"• Recommended Rate: {agent_data['recommended_rate']['rate']}")
    print(f"• Confidence Level: {agent_data['agent_context']['confidence_level']}")
    print(f"• Use for Decisions: {agent_data['agent_context']['use_for_decisions']}")
    print(f"• Rate Spread: {agent_data['market_analysis']['rate_spread']:.6f}")
    print(f"• Provider Consensus: {agent_data['market_analysis']['provider_consensus']}")
    print(f"• Errors: {agent_data['reliability']['errors']}")

if __name__ == "__main__":
    asyncio.run(main())