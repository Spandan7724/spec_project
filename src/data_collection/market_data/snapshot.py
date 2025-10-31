"""Build a complete LiveSnapshot combining providers, indicators, and regime."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.cache import cache as global_cache
from src.config import get_config, load_config
from src.data_collection.market_data.aggregator import (
    aggregate_rates,
    QualityMetrics,
)
from src.data_collection.market_data.indicators import Indicators, calculate_indicators
from src.data_collection.market_data.regime import Regime, classify_regime
from src.data_collection.providers.base import ProviderRate, BaseProvider
from src.utils.errors import DataProviderError
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class LiveSnapshot:
    currency_pair: str
    rate_timestamp: datetime
    mid_rate: float
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    provider_breakdown: List[ProviderRate]
    quality: QualityMetrics
    indicators: Indicators
    regime: Regime


async def _fetch_providers(base: str, quote: str, providers: List[BaseProvider]) -> List[ProviderRate]:
    async def _safe_call(p: BaseProvider):
        try:
            return await p.get_rate(base, quote)
        except Exception as e:
            logger.error(f"Provider {getattr(p, 'NAME', 'unknown')} failed: {e}")
            return None

    results = await asyncio.gather(*[_safe_call(p) for p in providers], return_exceptions=False)
    return [r for r in results if r is not None]


def _yahoo_symbol(base: str, quote: str) -> str:
    return f"{base}{quote}=X"


async def _load_historical(base: str, quote: str, days: int) -> Optional[pd.DataFrame]:
    symbol = _yahoo_symbol(base, quote)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d")
        if df is None or df.empty:
            return None
        # Ensure expected columns
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            return None
        columns = ["Open", "High", "Low", "Close"]
        if "Volume" in df.columns:
            columns.append("Volume")
        return df[columns]
    except Exception as e:
        logger.error(f"Historical load failed for {symbol}: {e}")
        return None


async def build_snapshot(
    base: str,
    quote: str,
    providers: List[BaseProvider],
    cache=global_cache,
    lookback_days: int = 90,
    historical_df: Optional[pd.DataFrame] = None,
) -> LiveSnapshot:
    # Load configuration if necessary
    try:
        cfg = get_config()
    except Exception:
        cfg = load_config()

    cache_ttl = int(cfg.get("agents.market_data.cache_ttl", 5))

    # Cache key for snapshot
    key = f"market_snapshot:{base}:{quote}"
    cached: Optional[LiveSnapshot] = cache.get(key)
    if cached is not None:
        return cached

    # Gather provider rates
    prov_rates = await _fetch_providers(base, quote, providers)

    # Aggregate; if none, try cache fallback, else raise
    if not prov_rates:
        cached = cache.get(key)
        if cached is not None:
            return cached
        raise DataProviderError("No provider data available for aggregation")

    mid_rate, best_bid, best_ask, spread, ts, quality, kept = aggregate_rates(
        prov_rates, cache_ttl_seconds=cache_ttl
    )

    # Historical for indicators
    if historical_df is None:
        historical_df = await _load_historical(base, quote, days=lookback_days)

    if historical_df is not None:
        inds = calculate_indicators(historical_df)
    else:
        inds = Indicators(*([None] * 14))
        if "insufficient history" not in quality.notes:
            quality.notes.append("Indicators unavailable (no historical data)")

    # Regime classification
    regime = classify_regime(latest_price=mid_rate, ind=inds)

    snapshot = LiveSnapshot(
        currency_pair=f"{base}/{quote}",
        rate_timestamp=ts,
        mid_rate=mid_rate,
        bid=best_bid,
        ask=best_ask,
        spread=spread,
        provider_breakdown=kept,
        quality=quality,
        indicators=inds,
        regime=regime,
    )

    # Cache snapshot
    cache.set(key, snapshot, ttl_seconds=cache_ttl)
    return snapshot


async def get_market_snapshot(base: str, quote: str, providers: List[BaseProvider]) -> LiveSnapshot:
    return await build_snapshot(base, quote, providers)
