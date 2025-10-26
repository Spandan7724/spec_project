"""Market Intelligence LangGraph node implementation."""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from src.agentic.state import AgentState
from src.data_collection.market_intelligence.intelligence_service import MarketIntelligenceService
from src.utils.decorators import timeout


@timeout(120.0)
async def market_intelligence_node(state: AgentState) -> Dict[str, Any]:
    start = time.time()
    base = state.get("base_currency") or (state.get("currency_pair", "").split("/")[0] if state.get("currency_pair") else "USD")
    quote = state.get("quote_currency") or (state.get("currency_pair", "").split("/")[-1] if state.get("currency_pair") else "EUR")

    # Check for offline demo mode before making any service calls
    if os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}:
        return {
            "intelligence_status": "success",
            "intelligence_error": None,
            "intelligence_report": {
                "pair": f"{base}/{quote}",
                "ts_utc": "1970-01-01T00:00:00Z",
                "news": {
                    "sent_base": 0.0,
                    "sent_quote": 0.0,
                    "pair_bias": 0.0,
                    "confidence": "low",
                    "n_articles_used": 0,
                    "top_evidence": [],
                    "narrative": "",
                },
                "calendar": {
                    "next_high_event": None,
                    "total_high_impact_events_7d": 0,
                },
                "policy_bias": 0.0,
            },
        }

    try:
        service = MarketIntelligenceService()
        report = await service.get_pair_intelligence(base, quote)
        return {
            "intelligence_status": "success",
            "intelligence_error": None,
            "intelligence_report": report,
        }
    except Exception as e:
        # Market intelligence node failed
        if os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}:
            # Fallback success in demo mode
            return {
                "intelligence_status": "success",
                "intelligence_error": None,
                "intelligence_report": {
                    "pair": f"{base}/{quote}",
                    "ts_utc": "1970-01-01T00:00:00Z",
                    "news": {
                        "sent_base": 0.0,
                        "sent_quote": 0.0,
                        "pair_bias": 0.0,
                        "confidence": "low",
                        "n_articles_used": 0,
                        "top_evidence": [],
                        "narrative": "",
                    },
                    "calendar": {
                        "next_high_event": None,
                        "total_high_impact_events_7d": 0,
                    },
                    "policy_bias": 0.0,
                },
            }
        # Otherwise partial, so production doesn't mask errors
        return {
            "intelligence_status": "partial",
            "intelligence_error": str(e),
            "intelligence_report": None,
        }
