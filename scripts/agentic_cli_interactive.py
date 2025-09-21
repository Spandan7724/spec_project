#!/usr/bin/env python3
"""Interactive CLI wrapper for the agentic currency assistant."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic import run_agentic_workflow, serialize_state


def prompt(prompt_text: str) -> str:
    try:
        return input(prompt_text)
    except EOFError:
        return ""


def run_interactive() -> None:
    print("Currency Assistant (Interactive Mode)")
    print("Press Enter without typing to exit.\n")

    while True:
        pair = prompt("Currency pair (e.g. USD/EUR): ").strip()
        if not pair:
            print("Goodbye!")
            return

        amount_raw = prompt("Amount in base currency: ").strip()
        if not amount_raw:
            print("No amount entered; exiting.")
            return

        try:
            amount = float(amount_raw)
        except ValueError:
            print("Invalid amount. Please try again.\n")
            continue

        risk = prompt("Risk tolerance [low/moderate/high] (default moderate): ").strip().lower() or "moderate"
        if risk not in {"low", "moderate", "high"}:
            print("Invalid risk tolerance. Please enter low, moderate, or high.\n")
            continue

        timeframe_raw = prompt("Timeframe in days (default 7): ").strip()
        timeframe = 7
        if timeframe_raw:
            try:
                timeframe = max(1, int(timeframe_raw))
            except ValueError:
                print("Invalid timeframe. Please enter a positive integer.\n")
                continue

        payload = {
            "currency_pair": pair,
            "amount": amount,
            "risk_tolerance": risk,
            "timeframe_days": timeframe,
        }

        try:
            state = run_agentic_workflow(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"Error running workflow: {exc}\n")
            continue

        result = serialize_state(state)
        recommendation = result["recommendation"]

        print("\n=== Recommendation ===")
        print(f"Action: {recommendation.get('action', 'unknown')} (confidence: {recommendation.get('confidence', 'n/a')})")
        if recommendation.get("summary"):
            print(f"Summary: {recommendation['summary']}")
        if recommendation.get("timeline"):
            print(f"Timeline: {recommendation['timeline']}")
        if recommendation.get("rationale"):
            print("Rationale:")
            for item in recommendation["rationale"]:
                print(f"  - {item}")
        if recommendation.get("warnings"):
            print("Warnings:")
            for warn in recommendation["warnings"]:
                print(f"  - {warn}")

        print("\n=== Market Snapshot ===")
        market = result["market_analysis"]
        print(market.get("summary", "No market summary available."))

        print("\n=== Economic Outlook ===")
        economic = result["economic_analysis"]
        print(economic.get("summary", "No economic summary available."))
        high_events = economic.get("high_impact_events", [])
        if high_events:
            print("High-impact events:")
            for event in high_events[:5]:
                title = event.get("title", "(unknown)")
                date = event.get("release_date", "?")
                print(f"  - {title} on {date}")
            if len(high_events) > 5:
                print(f"  ...and {len(high_events) - 5} more")

        print("\n=== Risk Assessment ===")
        risk_data = result["risk_assessment"]
        print(risk_data.get("summary", "No risk summary available."))
        scenarios = risk_data.get("scenarios", {})
        if scenarios:
            print("Scenarios:")
            for key, value in scenarios.items():
                print(f"  - {key}: {value}")

        print("\nType another pair to continue, or press Enter to exit.\n")


if __name__ == "__main__":
    run_interactive()
