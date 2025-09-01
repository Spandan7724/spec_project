#!/usr/bin/env python3
"""
Demo: Live Web Scraping Output for Multi-Agent System
Shows actual scraping results from the web scraping tools system
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.tools.web_scraper import SimpleWebScraper, GenericScrapingInterface
from src.tools.agent_interface import AgentScrapingInterface

async def get_live_web_scraping_output():
    """
    Get live web scraping output from the actual scraping system.
    This demonstrates what the web scraping tools actually return.
    """
    
    print("Testing live web scraping capabilities...")
    
    # URLs to test different types of financial content
    test_urls = [
        "https://www.investing.com/currencies/eur-usd",
        "https://www.fxstreet.com/currencies/eurusd", 
        "https://finance.yahoo.com/quote/EURUSD=X",
        "https://www.ecb.europa.eu/home/html/index.en.html"
    ]
    
    scraping_results = []
    
    try:
        # Test with SimpleWebScraper
        print("Testing SimpleWebScraper...")
        async with SimpleWebScraper() as scraper:
            for url in test_urls[:2]:  # Test first 2 URLs
                print(f"Scraping {url}...")
                try:
                    result = await scraper.scrape_url(
                        url=url,
                        bypass_cache=False,
                        content_type='financial'
                    )
                    scraping_results.append({
                        "tool": "SimpleWebScraper",
                        "url": url,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
                    scraping_results.append({
                        "tool": "SimpleWebScraper",
                        "url": url,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Test with GenericScrapingInterface  
        print("Testing GenericScrapingInterface...")
        generic_scraper = GenericScrapingInterface()
        
        for url in test_urls[2:]:  # Test remaining URLs
            print(f"Scraping {url} with GenericScrapingInterface...")
            try:
                result = await generic_scraper.scrape_content(
                    url=url,
                    content_type='general'
                )
                scraping_results.append({
                    "tool": "GenericScrapingInterface",
                    "url": url,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                scraping_results.append({
                    "tool": "GenericScrapingInterface", 
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Test with AgentScrapingInterface
        print("Testing AgentScrapingInterface...")
        agent_scraper = AgentScrapingInterface()
        
        # Test a financial query that agents might make
        query_result = await agent_scraper.search_and_scrape(
            query="USD EUR exchange rate news today",
            max_results=2
        )
        
        scraping_results.append({
            "tool": "AgentScrapingInterface",
            "query": "USD EUR exchange rate news today", 
            "result": query_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return scraping_results
        
    except Exception as e:
        print(f"Error in web scraping test: {e}")
        return scraping_results

def format_for_agents(scraping_results: list) -> dict:
    """
    Format the web scraping results for consumption by the multi-agent system.
    This is the standardized format agents will receive.
    """
    
    if not scraping_results:
        return {
            "data_type": "web_scraping",
            "timestamp": datetime.now().isoformat(),
            "scraping_attempts": 0,
            "error": "No scraping results generated"
        }
    
    # Analyze results by tool and success rate
    tool_stats = {}
    successful_scrapes = []
    failed_scrapes = []
    
    for result in scraping_results:
        tool = result.get("tool", "unknown")
        
        if tool not in tool_stats:
            tool_stats[tool] = {"attempts": 0, "successful": 0, "failed": 0}
        
        tool_stats[tool]["attempts"] += 1
        
        if "result" in result and result["result"]:
            tool_stats[tool]["successful"] += 1
            successful_scrapes.append(result)
        else:
            tool_stats[tool]["failed"] += 1
            failed_scrapes.append(result)
    
    # Extract content insights from successful scrapes
    content_analysis = {
        "total_content_size": sum([
            len(result["result"].content) if hasattr(result.get("result"), "content") else 0
            for result in successful_scrapes
        ]),
        "cached_results": len([
            result for result in successful_scrapes 
            if hasattr(result.get("result"), "cached") and result["result"].cached
        ]),
        "content_types": list(set([
            getattr(result.get("result"), "content", "unknown")[:20] + "..." if hasattr(result.get("result"), "content") else "no_content"
            for result in successful_scrapes
        ]))
    }
    
    agent_data = {
        "data_type": "web_scraping",
        "timestamp": datetime.now().isoformat(),
        
        # System performance metrics
        "performance": {
            "total_attempts": len(scraping_results),
            "successful_scrapes": len(successful_scrapes),
            "failed_scrapes": len(failed_scrapes),
            "success_rate": len(successful_scrapes) / len(scraping_results) if scraping_results else 0,
            "tool_performance": {
                tool: {
                    "success_rate": stats["successful"] / stats["attempts"] if stats["attempts"] > 0 else 0,
                    **stats
                }
                for tool, stats in tool_stats.items()
            }
        },
        
        # Content analysis
        "content_analysis": content_analysis,
        
        # Scraped data organized by source
        "scraped_content": [
            {
                "tool_used": result["tool"],
                "source_url": result.get("url", result.get("query", "unknown")),
                "success": "result" in result,
                "content_preview": (
                    result["result"].content[:500] + "..." if 
                    hasattr(result.get("result"), "content") and result["result"].content 
                    else None
                ),
                "status_code": (
                    result["result"].status_code if 
                    hasattr(result.get("result"), "status_code") 
                    else None
                ),
                "cached": (
                    result["result"].cached if 
                    hasattr(result.get("result"), "cached") 
                    else False
                ),
                "timestamp": result["timestamp"],
                "error": result.get("error")
            }
            for result in scraping_results
        ],
        
        # Agent decision support
        "agent_context": {
            "scraping_reliability": (
                "high" if len(successful_scrapes) / len(scraping_results) > 0.8 else
                "medium" if len(successful_scrapes) / len(scraping_results) > 0.5 else
                "low"
            ) if scraping_results else "unknown",
            
            "available_tools": list(tool_stats.keys()),
            
            "recommended_tool": max(tool_stats.items(), key=lambda x: x[1]["successful"] / max(x[1]["attempts"], 1))[0] if tool_stats else None,
            
            "content_availability": len(successful_scrapes) > 0,
            
            "fresh_data": len([
                result for result in successful_scrapes 
                if not (hasattr(result.get("result"), "cached") and result["result"].cached)
            ]) > 0,
            
            "rate_limiting_detected": len([
                result for result in failed_scrapes
                if "rate" in result.get("error", "").lower()
            ]) > 0,
            
            "network_issues": len([
                result for result in failed_scrapes
                if any(term in result.get("error", "").lower() for term in ["timeout", "connection", "network"])
            ]) > 0,
            
            "next_scraping_window": datetime.now().isoformat(),  # Could be calculated based on rate limits
            
            "usage_recommendations": [
                rec for rec, condition in [
                    ("reduce_scraping_frequency", len(failed_scrapes) > len(successful_scrapes)),
                    ("use_cached_data", content_analysis["cached_results"] > 0),
                    ("try_alternative_sources", len(successful_scrapes) < 2),
                    ("implement_delays", any("rate" in result.get("error", "").lower() for result in failed_scrapes))
                ] if condition
            ]
        }
    }
    
    return agent_data

async def main():
    """Generate and display live web scraping output for agents"""
    
    print("=== Live Web Scraping Output Demo for Multi-Agent System ===\n")
    
    # Get live results from actual system
    scraping_results = await get_live_web_scraping_output()
    
    if not scraping_results:
        print("No scraping results generated. Check network connectivity.")
        return
    
    print(f"Completed {len(scraping_results)} scraping attempts")
    successful = len([r for r in scraping_results if "result" in r])
    failed = len([r for r in scraping_results if "error" in r])
    print(f"Success: {successful}, Failed: {failed}")
    print()
    
    # Show brief summary of results
    print("=== Scraping Results Summary ===")
    for i, result in enumerate(scraping_results, 1):
        source = result.get("url", result.get("query", "unknown"))
        status = "✓ Success" if "result" in result else "✗ Failed"
        print(f"{i}. {result['tool']} - {source[:50]}... - {status}")
        if "error" in result:
            print(f"   Error: {result['error'][:100]}...")
    print()
    
    # Format for agents
    agent_data = format_for_agents(scraping_results)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Insights for Agent Decision Making ===")
    print(f"• Scraping Reliability: {agent_data['agent_context']['scraping_reliability']}")
    print(f"• Success Rate: {agent_data['performance']['success_rate']:.1%}")
    print(f"• Available Tools: {', '.join(agent_data['agent_context']['available_tools'])}")
    print(f"• Recommended Tool: {agent_data['agent_context']['recommended_tool']}")
    print(f"• Content Available: {agent_data['agent_context']['content_availability']}")
    print(f"• Fresh Data: {agent_data['agent_context']['fresh_data']}")
    
    if agent_data['agent_context']['usage_recommendations']:
        print(f"• Recommendations: {', '.join(agent_data['agent_context']['usage_recommendations'])}")

if __name__ == "__main__":
    asyncio.run(main())