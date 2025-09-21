#!/usr/bin/env python3
"""Command-line wrapper for running the agentic currency workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the src/ directory is importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic import run_agentic_workflow, serialize_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph-based currency assistant for a single request",
    )
    parser.add_argument(
        "currency_pair",
        help="Currency pair in formats like USD/EUR or usdeur",
    )
    parser.add_argument(
        "amount",
        type=float,
        help="Amount to convert in the base currency",
    )
    parser.add_argument(
        "--risk",
        choices=["low", "moderate", "high"],
        default="moderate",
        help="User risk tolerance (default: moderate)",
    )
    parser.add_argument(
        "--timeframe",
        type=int,
        default=7,
        help="Desired timeframe in days for the recommendation (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output instead of human-readable summary",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level when using --json (default: 2)",
    )
    return parser


def format_summary(state_dict: dict[str, Any]) -> str:
    request = state_dict.get("request", {})
    rec = state_dict.get("recommendation", {})
    market = state_dict.get("market_analysis", {})
    economic = state_dict.get("economic_analysis", {})
    risk = state_dict.get("risk_assessment", {})

    lines = []
    lines.append("=== Currency Assistant Recommendation ===")
    lines.append(
        f"Pair: {request.get('currency_pair')}  Amount: {request.get('amount')}  "
        f"Risk tolerance: {request.get('risk_tolerance')}  Timeframe: {request.get('timeframe_days')} days"
    )
    lines.append("")

    lines.append(f"Action: {rec.get('action', 'unknown')} (confidence: {rec.get('confidence', 'n/a')})")
    if rec.get("summary"):
        lines.append(f"Summary: {rec['summary']}")
    if rec.get("timeline"):
        lines.append(f"Timeline: {rec['timeline']}")
    if rec.get("rationale"):
        lines.append("Rationale:")
        for item in rec["rationale"]:
            lines.append(f"  - {item}")

    if rec.get("warnings"):
        lines.append("Warnings:")
        for warn in rec["warnings"]:
            lines.append(f"  - {warn}")

    # Market snapshot
    lines.append("")
    lines.append("-- Market Snapshot --")
    if market.get("summary"):
        lines.append(f"Market: {market['summary']}")
    if market.get("mid_rate"):
        lines.append(f"Mid rate: {market['mid_rate']} at {market.get('rate_timestamp')}")
    if market.get("ml_forecasts") and market["ml_forecasts"].get("model_id"):
        lines.append(
            f"ML Model: {market['ml_forecasts']['model_id']} (conf {market['ml_forecasts'].get('model_confidence', 'n/a')})"
        )

    # Economics
    lines.append("")
    lines.append("-- Economic Outlook --")
    if economic.get("summary"):
        lines.append(economic["summary"])
    high_events = economic.get("high_impact_events", [])
    if high_events:
        lines.append("High impact events:")
        for event in high_events[:5]:
            title = event.get("title", "(unknown)")
            date = event.get("release_date", "?")
            lines.append(f"  - {title} on {date}")
        if len(high_events) > 5:
            lines.append(f"  ...and {len(high_events) - 5} more")

    # Risk
    lines.append("")
    lines.append("-- Risk Assessment --")
    if risk.get("summary"):
        lines.append(risk["summary"])
    scenarios = risk.get("scenarios", {})
    if scenarios:
        lines.append("Scenarios:")
        for key, value in scenarios.items():
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    payload = {
        "currency_pair": args.currency_pair,
        "amount": args.amount,
        "risk_tolerance": args.risk,
        "timeframe_days": max(1, args.timeframe),
    }

    try:
        state = run_agentic_workflow(payload)
    except Exception as exc:  # noqa: BLE001
        parser.error(f"Workflow failed: {exc}")
        return 2

    result = serialize_state(state)

    if args.json:
        print(json.dumps(result, indent=args.indent, default=str))
    else:
        print(format_summary(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
