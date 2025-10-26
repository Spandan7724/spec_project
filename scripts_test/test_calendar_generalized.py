#!/usr/bin/env python3
"""
Test: Generalized search approach for economic calendar.

Instead of scraping specific sites, just search for:
- "USD economic calendar November 2025"
- "EUR ECB meeting December 2025"
- "Federal Reserve FOMC December 2025"

Then use gpt-5-mini to extract structured data from search results.

This is MUCH simpler and more robust!
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
import httpx
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


async def search_economic_calendar(query: str):
    """Search for economic calendar data with Serper"""
    print(f"\n🔍 Searching: {query}")
    print("-" * 80)
    
    if not SERPER_API_KEY:
        print("❌ SERPER_API_KEY not set")
        return None
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": 10
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Get organic results
                results = data.get("organic", [])
                
                print(f"✅ Found {len(results)} results\n")
                
                # Show what we got
                for i, result in enumerate(results[:5], 1):
                    print(f"{i}. {result.get('title', 'No title')}")
                    print(f"   URL: {result.get('link', '')}")
                    print(f"   Snippet: {result.get('snippet', '')[:150]}...")
                    print()
                
                # Extract text from results for LLM
                combined_text = "\n\n".join([
                    f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}\nURL: {r.get('link', '')}"
                    for r in results[:5]
                ])
                
                return combined_text
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


async def extract_calendar_data_mock(search_results: str, currency: str, start_date: str, end_date: str):
    """Mock LLM extraction - shows what the actual LLM would return"""
    
    print("\n🤖 LLM Extraction (Mock)")
    print("-" * 80)
    print(f"Currency: {currency}")
    print(f"Date Range: {start_date} to {end_date}")
    
    # This is what gpt-5-mini would extract from the search results
    mock_events = {
        "USD": [
            {
                "date": "2025-11-01",
                "time": "14:00",
                "timezone": "America/New_York",
                "event": "FOMC Interest Rate Decision",
                "importance": "high",
                "source": "Federal Reserve"
            },
            {
                "date": "2025-11-08",
                "time": "08:30",
                "timezone": "America/New_York",
                "event": "US Nonfarm Payrolls (NFP)",
                "importance": "high",
                "source": "Bureau of Labor Statistics"
            },
            {
                "date": "2025-11-13",
                "time": "08:30",
                "timezone": "America/New_York",
                "event": "Consumer Price Index (CPI)",
                "importance": "high",
                "source": "Bureau of Labor Statistics"
            },
            {
                "date": "2025-11-27",
                "time": "08:30",
                "timezone": "America/New_York",
                "event": "GDP Growth Rate (Q3 2025)",
                "importance": "high",
                "source": "Bureau of Economic Analysis"
            }
        ],
        "EUR": [
            {
                "date": "2025-11-07",
                "time": "13:15",
                "timezone": "Europe/Frankfurt",
                "event": "ECB Monetary Policy Decision",
                "importance": "high",
                "source": "European Central Bank"
            },
            {
                "date": "2025-11-07",
                "time": "13:45",
                "timezone": "Europe/Frankfurt",
                "event": "ECB Press Conference",
                "importance": "high",
                "source": "European Central Bank"
            },
            {
                "date": "2025-11-29",
                "time": "10:00",
                "timezone": "Europe/Brussels",
                "event": "Eurozone CPI Flash Estimate",
                "importance": "high",
                "source": "Eurostat"
            }
        ]
    }
    
    events = mock_events.get(currency, [])
    
    print(f"\n✅ Extracted {len(events)} events:\n")
    print(json.dumps(events, indent=2))
    
    return events


async def test_generalized_calendar_search():
    """Test the generalized search approach"""
    
    print("=" * 80)
    print("GENERALIZED ECONOMIC CALENDAR SEARCH TEST")
    print("=" * 80)
    print("\nApproach: Search for calendar data, extract with LLM")
    print("Much simpler than scraping specific sites!")
    print("=" * 80)
    
    # Get date ranges
    today = datetime.now()
    next_month = today + timedelta(days=30)
    
    start_date = today.strftime("%Y-%m-%d")
    end_date = next_month.strftime("%Y-%m-%d")
    month_name = today.strftime("%B %Y")
    next_month_name = next_month.strftime("%B %Y")
    
    # Test queries for USD
    print("\n\n📅 SEARCHING FOR USD ECONOMIC EVENTS")
    print("=" * 80)
    
    usd_queries = [
        f"USD economic calendar {month_name}",
        f"Federal Reserve FOMC meeting {month_name} {next_month_name}",
        f"US CPI NFP GDP release dates {month_name}",
    ]
    
    usd_results = []
    for query in usd_queries:
        results = await search_economic_calendar(query)
        if results:
            usd_results.append(results)
        await asyncio.sleep(1)
    
    # Extract events with LLM (mock)
    if usd_results:
        combined_usd = "\n\n---\n\n".join(usd_results)
        usd_events = await extract_calendar_data_mock(combined_usd, "USD", start_date, end_date)
    
    # Test queries for EUR
    print("\n\n📅 SEARCHING FOR EUR ECONOMIC EVENTS")
    print("=" * 80)
    
    eur_queries = [
        f"EUR economic calendar {month_name}",
        f"ECB monetary policy meeting {month_name} {next_month_name}",
        f"Eurozone CPI inflation {month_name}",
    ]
    
    eur_results = []
    for query in eur_queries:
        results = await search_economic_calendar(query)
        if results:
            eur_results.append(results)
        await asyncio.sleep(1)
    
    # Extract events with LLM (mock)
    if eur_results:
        combined_eur = "\n\n---\n\n".join(eur_results)
        eur_events = await extract_calendar_data_mock(combined_eur, "EUR", start_date, end_date)
    
    # Show final output
    print("\n\n📊 FINAL CALENDAR SNAPSHOT FOR USD/EUR")
    print("=" * 80)
    
    # Find next high-impact event
    all_events = []
    if 'usd_events' in locals():
        for event in usd_events:
            event['currency'] = 'USD'
            all_events.append(event)
    if 'eur_events' in locals():
        for event in eur_events:
            event['currency'] = 'EUR'
            all_events.append(event)
    
    # Sort by date
    all_events.sort(key=lambda e: e['date'])
    
    # Find next high-impact event
    next_high = None
    for event in all_events:
        event_date = datetime.fromisoformat(event['date'])
        if event_date >= today and event['importance'] == 'high':
            next_high = event
            break
    
    if next_high:
        event_date = datetime.fromisoformat(next_high['date'])
        days_until = (event_date - today).days
        
        print("\n🔔 NEXT HIGH-IMPACT EVENT:")
        print(f"   Date: {next_high['date']} ({days_until} days)")
        print(f"   Time: {next_high['time']} {next_high['timezone']}")
        print(f"   Event: {next_high['event']}")
        print(f"   Currency: {next_high['currency']}")
        print(f"   Source: {next_high['source']}")
    
    print("\n\n📋 ALL UPCOMING HIGH-IMPACT EVENTS:")
    print("-" * 80)
    
    for event in all_events[:10]:
        if event['importance'] == 'high':
            event_date = datetime.fromisoformat(event['date'])
            days_until = (event_date - today).days
            print(f"[{days_until:2d}d] {event['date']} - {event['currency']} - {event['event']}")
    
    # Show what the agent would receive
    print("\n\n💾 JSON OUTPUT (What agent receives):")
    print("=" * 80)
    
    output = {
        "pair": "USD/EUR",
        "ts_utc": datetime.now().isoformat(),
        "calendar": {
            "next_high_event": next_high if next_high else None,
            "all_upcoming_events": all_events[:10],
            "total_high_impact": sum(1 for e in all_events if e['importance'] == 'high'),
            "data_source": "Serper search + gpt-5-mini extraction"
        }
    }
    
    print(json.dumps(output, indent=2))


async def main():
    print("\n" + "📅" * 40)
    print("GENERALIZED CALENDAR SEARCH APPROACH")
    print("📅" * 40)
    
    if not SERPER_API_KEY:
        print("\n⚠️ WARNING: SERPER_API_KEY not set")
        print("Will show mock data for demonstration\n")
    
    await test_generalized_calendar_search()
    
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
✅ ADVANTAGES OF GENERALIZED SEARCH:

1. No site-specific scraping (no breaking when sites change)
2. Gets data from MULTIPLE sources automatically
3. More resilient - if one source is down, others still work
4. Easier to maintain - just natural language queries
5. Can search for specific events: "Fed rate decision December 2025"

🔧 IMPLEMENTATION:

Search queries:
- "{currency} economic calendar {month}"
- "Fed FOMC meeting {month}"
- "ECB policy decision {month}"
- "{currency} CPI NFP GDP release {month}"

LLM extraction (gpt-5-mini):
- Input: Search results (titles + snippets)
- Output: JSON array with [date, time, event, importance]
- Cost: ~$0.0003 per search (very cheap!)

📊 DATA QUALITY:

- Multiple sources = more complete calendar
- LLM validates and normalizes dates/times
- Can filter by importance level
- Returns source attribution

COST: ~$0.002 per calendar refresh (6 searches × $0.0003)
    """)


if __name__ == "__main__":
    asyncio.run(main())

