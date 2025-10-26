#!/usr/bin/env python
"""Phase 3 end-to-end runner including Decision Engine.

Usage:
  uv run python scripts_test/test_phase_3.py

Environment variables (optional):
  PAIR=USD/EUR
  TIMEFRAME=1_week           # immediate | 1_day | 1_week | 1_month
  AMOUNT=5000                # numeric
  RISK=moderate              # conservative | moderate | aggressive
  URGENCY=normal             # urgent | normal | flexible
  OFFLINE_DEMO=false         # set true for offline fallbacks

Notes:
  - If no trained models exist, decision still runs using utility with intelligence/technicals.
  - Heuristic fallback is disabled by default and used only as last resort if enabled.
"""

import os
import time
from typing import Any, Dict

from src.agentic.graph import create_graph
from src.agentic.state import initialize_state
import json


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _print_decision(rec: Dict[str, Any]) -> None:
    print("\n== Decision Recommendation ==")
    print("action:", rec.get("action"))
    print("confidence:", f"{rec.get('confidence'):.2f}" if isinstance(rec.get("confidence"), (int, float)) else rec.get("confidence"))
    print("timeline:", rec.get("timeline"))

    # Staged plan
    sp = rec.get("staged_plan")
    if sp:
        print(f"\nStaged plan: {sp.get('num_tranches')} tranches, spacing ~ {sp.get('spacing_days')} days, extra_cost_bps={sp.get('total_extra_cost_bps')}")
        for t in sp.get("tranches", []):
            print(f"  - Day {t.get('execute_day')}: {t.get('percentage')}%  ({t.get('rationale')})")
        if sp.get("benefit"):
            print("benefit:", sp.get("benefit"))

    # Expected outcome
    eo = rec.get("expected_outcome") or {}
    if eo:
        print("\nExpected outcome:")
        print("  expected_rate:", eo.get("expected_rate"))
        print("  range_low:", eo.get("range_low"), "range_high:", eo.get("range_high"))
        print("  expected_improvement_bps:", eo.get("expected_improvement_bps"))

    # Risk + costs
    rs = rec.get("risk_summary") or {}
    if rs:
        print("\nRisk summary:")
        print("  level:", rs.get("risk_level"), "event:", rs.get("event_risk"), "var_95:", rs.get("var_95"))
    ce = rec.get("cost_estimate") or {}
    if ce:
        print("\nCosts:")
        print("  spread_bps:", ce.get("spread_bps"), "fee_bps:", ce.get("fee_bps"), "total_bps:", ce.get("total_bps"))

    # Rationale + warnings
    rat = rec.get("rationale") or []
    if rat:
        print("\nRationale:")
        for r in rat:
            print(" -", r)
    warns = rec.get("warnings") or []
    if warns:
        print("\nWarnings:")
        for w in warns:
            print(" -", w)

    # Component confidences (for transparency)
    cc = rec.get("component_confidences") or {}
    if cc:
        print("\nComponent confidences:")
        for k, v in cc.items():
            print(f"  {k}: {v:.2f}")


def _days_for_timeframe(tf: str) -> int:
    m = (tf or "1_week").lower()
    if m == "immediate":
        return 0
    if m == "1_day":
        return 1
    if m == "1_week":
        return 7
    if m == "1_month":
        return 30
    return 7


def _print_citations_and_evidence(state: Dict[str, Any], timeframe: str) -> None:
    """Surface source links and model evidence to improve UX in GUIs.

    Pulls from market_snapshot, intelligence_report, and price_forecast to show
    concrete citations (URLs) and the most relevant model evidence for the requested timeframe.
    """
    tf_days = _days_for_timeframe(timeframe)

    print("\n== Evidence & Citations ==")

    # Market data providers and quality notes
    ms = state.get("market_snapshot") or {}
    providers = ms.get("provider_breakdown") or []
    if providers:
        names = ", ".join(sorted(set([p.get("source", "unknown") for p in providers])))
        print("Market data providers:", names)
    q = (ms.get("quality") or {}).get("notes") or []
    if q:
        print("Market data quality notes:")
        for note in q:
            print(" -", note)

    # Market Intelligence: top news evidence + calendar sources
    mi = state.get("intelligence_report") or {}
    news = (mi.get("news") or {}).get("top_evidence") or []
    if news:
        print("\nNews citations (top evidence):")
        for i, ev in enumerate(news[:5], 1):
            print(f"  {i}. {ev.get('source')}: {ev.get('title')}")
            if ev.get("url"):
                print(f"     {ev.get('url')}")
    evs = (mi.get("calendar") or {}).get("events_extracted") or []
    if evs:
        print("\nCalendar sources:")
        for e in evs[:5]:
            if e.get("source_url"):
                print(f"  - {e.get('currency')} {e.get('event')} [{e.get('importance')}] → {e.get('source_url')}")

    # Model evidence for the decision horizon
    pf = state.get("price_forecast") or {}
    expl = (pf.get("explanations") or {}).get("daily") or {}
    if tf_days > 0 and expl.get(str(tf_days)):
        item = expl[str(tf_days)]
        tf = item.get("top_features") or {}
        if tf:
            print("\nModel evidence (top features at decision horizon):")
            top_line = ", ".join(f"{k}:{v:.1f}" for k, v in list(tf.items()))
            print("  ", top_line)
    elif expl:
        # Fall back to any available horizon
        first_key = sorted(expl.keys(), key=lambda x: int(x))[0]
        item = expl.get(first_key) or {}
        tf = item.get("top_features") or {}
        if tf:
            print("\nModel evidence (top features):")
            top_line = ", ".join(f"{k}:{v:.1f}" for k, v in list(tf.items()))
            print("  ", top_line)


def main():
    # Ensure run from repo root for config/model paths
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    try:
        os.chdir(repo_root)
    except Exception:
        pass

    if _env_flag("OFFLINE_DEMO"):
        print("[warn] OFFLINE_DEMO=true -> upstream nodes may use fallback data.")

    pair = _get_env("PAIR", "USD/EUR")
    base, quote = pair.split("/")
    timeframe = _get_env("TIMEFRAME", "1_week")
    amount = float(_get_env("AMOUNT", "5000"))
    risk = _get_env("RISK", "moderate")
    urgency = _get_env("URGENCY", "normal")

    # Create the graph and state
    start_time = time.time()
    g = create_graph()
    s = initialize_state(
        f"Convert {amount:.0f} {base} to {quote}",
        base_currency=base,
        quote_currency=quote,
        amount=amount,
        risk_tolerance=risk,
        urgency=urgency,
        timeframe=timeframe,
    )

    # Invoke full workflow
    r = g.invoke(s)
    end_time = time.time()
    execution_time = end_time - start_time

    # Layer statuses
    md_status = r.get("market_data_status", "unknown")
    mi_status = r.get("intelligence_status", "unknown")
    pr_status = r.get("prediction_status", "unknown")
    dc_status = r.get("decision_status", "unknown")

    print("market_status:", md_status)
    print("intel_status:", mi_status)
    print("prediction_status:", pr_status)
    print("decision_status:", dc_status)

    # Decision output (full recommendation)
    rec = r.get("recommendation") or {}
    _print_decision(rec)

    # Evidence & citations to support GUI UX
    _print_citations_and_evidence(r, timeframe)

    # Structured JSON block for GUI parsing
    ev = rec.get("evidence") or {}
    meta = rec.get("meta") or {}
    print("\n== Evidence JSON ==")
    print(json.dumps({"meta": meta, "evidence": ev}, indent=2))

    print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
