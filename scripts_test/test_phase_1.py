"""Phase 1 end-to-end runner using the LangGraph workflow.

Usage:
  uv run python test_graph.py

Notes:
- Reads API keys from your environment/.env (SERPER_API_KEY, COPILOT_ACCESS_TOKEN).
- Prints both statuses and errors if any component returns "partial".
"""

import os
import time
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state


# Optional: ensure demo mode is OFF for real providers/LLMs (honors your env)
OFFLINE = os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}
if OFFLINE:
    print("[warn] OFFLINE_DEMO=true -> nodes may return fallback data. Set OFFLINE_DEMO=false for live run.")

# Start timing
start_time = time.time()

# Create the graph
g = create_graph()

# Initialize state
s = initialize_state(
    "Convert 5000 USD to EUR",
    base_currency="USD",
    quote_currency="EUR",
)

# Invoke the graph
r = g.invoke(s)

# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time

# Market Data output
md_status = r.get("market_data_status", "unknown")
print("market_status:", md_status)
if md_status == "success" and r.get("market_snapshot"):
    print("mid_rate:", r["market_snapshot"].get("mid_rate"))
else:
    print("market_error:", r.get("market_data_error"))

# Market Intelligence output
mi_status = r.get("intelligence_status", "unknown")
print("intel_status:", mi_status)
if mi_status == "success" and r.get("intelligence_report"):
    rep = r["intelligence_report"]
    print("policy_bias:", rep.get("policy_bias"))
    print("news_bias:", rep.get("news", {}).get("pair_bias"), "conf:", rep.get("news", {}).get("confidence"))
    print("next_event:", rep.get("calendar", {}).get("next_high_event"))
    # Show which news articles were used (top evidence)
    te = rep.get("news", {}).get("top_evidence", [])
    if te:
        print("\nNews (top evidence):")
        for i, ev in enumerate(te, 1):
            print(f"  {i}. {ev.get('source')}: {ev.get('title')}")
            print(f"     {ev.get('url')}")
    # Show a few extracted calendar events
    evs = rep.get("calendar", {}).get("events_extracted", [])
    if evs:
        print("\nCalendar (events used):")
        for e in evs:
            print(f"  - {e.get('currency')} {e.get('event')} [{e.get('importance')}] @ {e.get('when_utc')}")
            print(f"    {e.get('source_url')}")
else:
    print("intel_error:", r.get("intelligence_error"))

# Print execution time
print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")
