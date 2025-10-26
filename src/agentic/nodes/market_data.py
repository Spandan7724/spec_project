"""Market Data LangGraph node implementation."""
from __future__ import annotations

import time
import os
from typing import Any, Dict, List

from src.agentic.state import AgentState
from src.cache import cache
from src.config import get_config, load_config
from src.data_collection.market_data.snapshot import build_snapshot
from src.data_collection.providers import get_provider
from src.utils.decorators import timeout, log_execution
from src.utils.logging import get_logger


logger = get_logger(__name__)


@timeout(10.0)
@log_execution(log_args=False, log_result=False)
async def market_data_node(state: AgentState) -> Dict[str, Any]:
    """Fetch live market snapshot (rates, indicators, regime) and update state.

    Returns a partial state update with keys:
    - market_data_status: "success" | "partial" | "error"
    - market_data_error: optional error message
    - market_snapshot: dict payload when available
    """
    start = time.time()

    # Extract currencies; default for robustness in early pipeline
    base = state.get("base_currency") or (state.get("currency_pair", "").split("/")[0] if state.get("currency_pair") else "USD")
    quote = state.get("quote_currency") or (state.get("currency_pair", "").split("/")[-1] if state.get("currency_pair") else "EUR")

    try:
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()

        provider_names: List[str] = cfg.get("agents.market_data.providers", ["exchange_rate_host", "yfinance"]) or []
        providers = [get_provider(name) for name in provider_names]

        # Build snapshot (uses cache + yfinance historical internally)
        snapshot = await build_snapshot(base, quote, providers, cache=cache)

        # Serialize snapshot to dict for state storage
        snapshot_dict = {
            "currency_pair": snapshot.currency_pair,
            "rate_timestamp": snapshot.rate_timestamp.isoformat(),
            "mid_rate": snapshot.mid_rate,
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread": snapshot.spread,
            "provider_breakdown": [
                {
                    "source": pr.source,
                    "rate": pr.rate,
                    "bid": pr.bid,
                    "ask": pr.ask,
                    "timestamp": pr.timestamp.isoformat(),
                    "notes": pr.notes,
                }
                for pr in snapshot.provider_breakdown
            ],
            "quality": {
                "sources_success": snapshot.quality.sources_success,
                "sources_total": snapshot.quality.sources_total,
                "dispersion_bps": snapshot.quality.dispersion_bps,
                "fresh": snapshot.quality.fresh,
                "notes": snapshot.quality.notes,
            },
            "indicators": {
                "sma_20": snapshot.indicators.sma_20,
                "sma_50": snapshot.indicators.sma_50,
                "ema_12": snapshot.indicators.ema_12,
                "ema_26": snapshot.indicators.ema_26,
                "rsi_14": snapshot.indicators.rsi_14,
                "macd": snapshot.indicators.macd,
                "macd_signal": snapshot.indicators.macd_signal,
                "macd_histogram": snapshot.indicators.macd_histogram,
                "bb_middle": snapshot.indicators.bb_middle,
                "bb_upper": snapshot.indicators.bb_upper,
                "bb_lower": snapshot.indicators.bb_lower,
                "bb_position": snapshot.indicators.bb_position,
                "atr_14": snapshot.indicators.atr_14,
                "realized_vol_30d": snapshot.indicators.realized_vol_30d,
            },
            "regime": {
                "trend_direction": snapshot.regime.trend_direction,
                "bias": snapshot.regime.bias,
            },
        }

        exec_ms = int((time.time() - start) * 1000)
        logger.info(
            "Market data snapshot built",
            extra={"base": base, "quote": quote, "time_ms": exec_ms},
        )

        return {
            "market_data_status": "success",
            "market_data_error": None,
            "market_snapshot": snapshot_dict,
        }

    except Exception as e:
        # Graceful degradation:
        # - In offline/demo mode, return a minimal success snapshot
        # - Otherwise, return partial with error so production uses real data only
        logger.error(f"Market data node failed: {e}")
        if os.getenv("OFFLINE_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}:
            return {
                "market_data_status": "success",
                "market_data_error": None,
                "market_snapshot": {
                    "currency_pair": f"{base}/{quote}",
                    "rate_timestamp": "1970-01-01T00:00:00Z",
                    "mid_rate": 1.0,
                    "bid": None,
                    "ask": None,
                    "spread": None,
                    "provider_breakdown": [],
                    "quality": {
                        "sources_success": 0,
                        "sources_total": 0,
                        "dispersion_bps": 0.0,
                        "fresh": False,
                        "notes": ["fallback_offline"],
                    },
                    "indicators": {k: None for k in [
                        "sma_20","sma_50","ema_12","ema_26","rsi_14","macd","macd_signal",
                        "macd_histogram","bb_middle","bb_upper","bb_lower","bb_position","atr_14","realized_vol_30d"
                    ]},
                    "regime": {"trend_direction": None, "bias": None},
                },
            }
        return {"market_data_status": "partial", "market_data_error": str(e), "market_snapshot": None}
