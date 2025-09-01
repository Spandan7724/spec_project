#!/usr/bin/env python3
"""
Demo: Live Economic Calendar Output for Multi-Agent System
Shows actual economic calendar data from the calendar collector system
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / '.env')

from src.data_collection.economic.calendar_collector import EconomicCalendarCollector, EconomicCalendar

async def get_live_economic_calendar_output():
    """
    Get live economic calendar output from the actual economic calendar system.
    This demonstrates what the EconomicCalendarCollector actually returns.
    """
    
    print("Collecting live economic calendar data...")
    
    try:
        # Initialize the economic calendar collector
        collector = EconomicCalendarCollector()
        
        # Get upcoming events for the next week
        calendar = await collector.get_economic_calendar(
            days_ahead=7,
            force_refresh=True
        )
        
        return calendar
        
    except Exception as e:
        print(f"Error collecting economic calendar: {e}")
        print("This is expected if economic data sources are unavailable.")
        return None

def format_for_agents(calendar: EconomicCalendar) -> dict:
    """
    Format the economic calendar for consumption by the multi-agent system.
    This is the standardized format agents will receive.
    """
    
    if not calendar or not calendar.events:
        return {
            "data_type": "economic_calendar",
            "timestamp": datetime.now().isoformat(),
            "events_found": 0,
            "error": "No calendar events collected"
        }
    
    now = datetime.now()
    
    # Categorize events by timing and impact
    upcoming_events = [e for e in calendar.events if e.is_upcoming]
    recent_events = [e for e in calendar.events if not e.is_upcoming and (now - e.release_date).total_seconds() < 86400]
    high_impact_upcoming = [e for e in upcoming_events if e.is_high_impact]
    
    agent_data = {
        "data_type": "economic_calendar",
        "timestamp": calendar.last_updated.isoformat(),
        "period": {
            "start_date": calendar.start_date.isoformat(),
            "end_date": calendar.end_date.isoformat()
        },
        "data_sources": calendar.sources,
        
        # High-priority events for immediate agent attention
        "critical_upcoming": [
            {
                "event_id": event.event_id,
                "title": event.title,
                "country": event.country,
                "currency": event.currency,
                "release_time": event.release_date.isoformat(),
                "hours_until_release": (event.release_date - now).total_seconds() / 3600,
                "impact": event.impact.value,
                "category": event.category,
                "previous_value": event.previous_value,
                "forecast_value": event.forecast_value,
                "market_relevance": {
                    "affects_usd": event.currency == "USD",
                    "affects_eur": event.currency == "EUR", 
                    "affects_gbp": event.currency == "GBP",
                    "high_impact": event.is_high_impact
                }
            }
            for event in high_impact_upcoming[:5]  # Top 5 critical events
        ],
        
        # Recently released events with market impact
        "recent_releases": [
            {
                "event_id": event.event_id,
                "title": event.title,
                "currency": event.currency,
                "release_time": event.release_date.isoformat(),
                "hours_since_release": (now - event.release_date).total_seconds() / 3600,
                "impact": event.impact.value,
                "result": {
                    "actual": event.actual_value,
                    "forecast": event.forecast_value,
                    "previous": event.previous_value,
                    "surprise_factor": event.surprise_factor
                }
            }
            for event in recent_events
        ],
        
        # Summary for agent decision making
        "market_outlook": {
            "total_events": len(calendar.events),
            "high_impact_this_week": len(high_impact_upcoming),
            "key_currencies_affected": list(set([e.currency for e in high_impact_upcoming])),
            "next_major_event": {
                "title": high_impact_upcoming[0].title if high_impact_upcoming else None,
                "hours_away": (high_impact_upcoming[0].release_date - now).total_seconds() / 3600 if high_impact_upcoming else None,
                "currency": high_impact_upcoming[0].currency if high_impact_upcoming else None
            } if high_impact_upcoming else None
        },
        
        # Agent decision support
        "agent_context": {
            "volatility_warning": len(high_impact_upcoming) > 2,
            "timing_considerations": [
                f"{len([e for e in high_impact_upcoming if (e.release_date - now).total_seconds() < 86400])} high-impact events in next 24h",
                f"Total upcoming events: {len(upcoming_events)}"
            ],
            "risk_level": "high" if len(high_impact_upcoming) > 3 else "medium" if len(high_impact_upcoming) > 1 else "low",
            "recommended_strategy": "wait" if len([e for e in high_impact_upcoming if (e.release_date - now).total_seconds() < 3600]) > 0 else "monitor",
            "data_freshness": "current" if (now - calendar.last_updated).total_seconds() < 1800 else "stale"
        }
    }
    
    return agent_data

async def main():
    """Generate and display live economic calendar output for agents"""
    
    print("=== Live Economic Calendar Output Demo for Multi-Agent System ===\n")
    
    # Get live data from actual system
    calendar = await get_live_economic_calendar_output()
    
    if calendar is None:
        print("Could not collect economic calendar data. Check data source availability.")
        return
    
    print("Raw Economic Calendar:")
    print(f"Events Period: {calendar.start_date.date()} to {calendar.end_date.date()}")
    print(f"Total Events: {len(calendar.events)}")
    print(f"Data Sources: {', '.join(calendar.sources)}")
    print()
    
    # Format for agents
    agent_data = format_for_agents(calendar)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Insights for Agent Decision Making ===")
    print(f"• Risk Level: {agent_data['agent_context']['risk_level']}")
    print(f"• Recommended Strategy: {agent_data['agent_context']['recommended_strategy']}")
    print(f"• High Impact Events This Week: {agent_data['market_outlook']['high_impact_this_week']}")
    if agent_data['market_outlook']['next_major_event']:
        print(f"• Next Major Event: {agent_data['market_outlook']['next_major_event']['title']} in {agent_data['market_outlook']['next_major_event']['hours_away']:.1f} hours")
    print(f"• Key Currencies: {', '.join(agent_data['market_outlook']['key_currencies_affected'])}")

if __name__ == "__main__":
    asyncio.run(main())