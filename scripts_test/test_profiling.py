"""Profile the currency assistant to find bottlenecks.

This script breaks down execution time by component to identify
where the system is spending time.
"""

import asyncio
import time
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state


async def profile_intelligence_service():
    """Profile the intelligence service separately."""
    from src.data_collection.market_intelligence.intelligence_service import MarketIntelligenceService
    
    print("\n🔬 Profiling Market Intelligence Service")
    print("="*60)
    
    service = MarketIntelligenceService()
    base, quote = "USD", "EUR"
    
    timings = {}
    
    # Test news collection
    print("\n1️⃣ Testing News Collection...")
    start = time.time()
    news_snapshot = await service.aggregator.get_pair_snapshot(base, quote)
    timings['news'] = time.time() - start
    print(f"   ⏱️  News aggregation: {timings['news']:.2f}s")
    print(f"   📊 Articles used: {news_snapshot.n_articles_used}")
    
    # Test calendar collection
    print("\n2️⃣ Testing Calendar Collection...")
    start = time.time()
    base_cal = await service.calendar.collect_calendar_urls(base, num_results=4)
    quote_cal = await service.calendar.collect_calendar_urls(quote, num_results=4)
    timings['calendar_collect'] = time.time() - start
    print(f"   ⏱️  Calendar URL collection: {timings['calendar_collect']:.2f}s")
    
    # Test calendar extraction
    print("\n3️⃣ Testing Calendar Extraction...")
    start = time.time()
    base_events = await service.cal_extractor.extract_events_batch(base_cal, base)
    quote_events = await service.cal_extractor.extract_events_batch(quote_cal, quote)
    timings['calendar_extract'] = time.time() - start
    print(f"   ⏱️  Calendar event extraction: {timings['calendar_extract']:.2f}s")
    
    # Test narrative generation
    print("\n4️⃣ Testing Narrative Generation...")
    start = time.time()
    narrative_text = await service.narrative.generate_narrative(news_snapshot.__dict__)
    timings['narrative'] = time.time() - start
    print(f"   ⏱️  Narrative generation: {timings['narrative']:.2f}s")
    
    return timings


async def main():
    print("\n" + "="*60)
    print("🔬 Currency Assistant Performance Profiler")
    print("="*60)
    
    # Profile intelligence service
    intel_timings = await profile_intelligence_service()
    
    # Profile full graph
    print("\n" + "="*60)
    print("5️⃣ Testing Full Graph Execution")
    print("="*60)
    
    g = create_graph()
    s = initialize_state(
        "Convert 5000 USD to EUR",
        base_currency="USD",
        quote_currency="EUR",
    )
    
    start = time.time()
    result = await g.ainvoke(s)  # Async invoke for proper profiling
    total_time = time.time() - start
    
    print(f"   ⏱️  Total execution: {total_time:.2f}s")
    print(f"   📊 Status: {result.get('intelligence_status')}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE BREAKDOWN")
    print("="*60)
    
    print("\n🏭 Market Intelligence Components:")
    for component, duration in intel_timings.items():
        percentage = (duration / sum(intel_timings.values())) * 100
        print(f"   {component:20}: {duration:6.2f}s ({percentage:5.1f}%)")
    
    print(f"\n⚡ Total Intelligence Time: {sum(intel_timings.values()):.2f}s")
    print(f"⚡ Total Graph Time:      {total_time:.2f}s")
    
    other_time = total_time - sum(intel_timings.values())
    if other_time > 0:
        print(f"⚡ Other Components:       {other_time:.2f}s")
    
    # Identify bottlenecks
    print("\n" + "="*60)
    print("🎯 BOTTLENECK ANALYSIS")
    print("="*60)
    
    sorted_timings = sorted(intel_timings.items(), key=lambda x: x[1], reverse=True)
    print("\nSlowest components (sorted):")
    for i, (component, duration) in enumerate(sorted_timings[:3], 1):
        print(f"   {i}. {component:20}: {duration:.2f}s")
    
    # Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if intel_timings.get('news', 0) > 20:
        print("   • News classification is slow - reduce articles or optimize")
    if intel_timings.get('calendar_extract', 0) > 15:
        print("   • Calendar extraction is slow - reduce sources or optimize")
    if intel_timings.get('narrative', 0) > 15:
        print("   • Narrative generation is slow - consider caching or optimization")
    if sorted_timings[0][1] / sum(intel_timings.values()) > 0.5:
        print(f"   • {sorted_timings[0][0]} dominates execution time - optimize first")


if __name__ == "__main__":
    asyncio.run(main())
