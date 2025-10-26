"""Aggregation utilities for market data provider rates."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import List, Optional, Tuple

from src.data_collection.providers.base import ProviderRate


@dataclass
class QualityMetrics:
    sources_success: int
    sources_total: int
    dispersion_bps: float
    fresh: bool
    notes: List[str] = field(default_factory=list)


def _provider_mid(pr: ProviderRate) -> float:
    if pr.bid is not None and pr.ask is not None and pr.bid > 0 and pr.ask > 0:
        return (pr.bid + pr.ask) / 2.0
    return float(pr.rate)


def _bps_diff(a: float, b: float) -> float:
    # basis points difference using b as reference
    if b <= 0:
        return 0.0
    return abs(a - b) / b * 10000.0


def aggregate_rates(
    provider_rates: List[ProviderRate],
    cache_ttl_seconds: int = 5,
    outlier_threshold_bps: float = 100.0,
    dispersion_warn_bps: float = 50.0,
) -> Tuple[float, Optional[float], Optional[float], Optional[float], datetime, QualityMetrics, List[ProviderRate]]:
    """
    Aggregate a list of provider rates into a consensus mid rate and quality metrics.

    Returns:
        (mid_rate, best_bid, best_ask, spread, rate_timestamp, quality_metrics, filtered_rates)
    """
    total = len(provider_rates)
    success = sum(1 for _ in provider_rates)

    if total == 0:
        # Nothing to aggregate
        now = datetime.now(timezone.utc)
        qm = QualityMetrics(0, 0, 0.0, False, notes=["No provider data available"]) 
        return 0.0, None, None, None, now, qm, []

    # Compute per-provider mid values
    mids = [_provider_mid(r) for r in provider_rates]
    med = median(mids)

    # Remove outliers beyond threshold from median
    kept: List[ProviderRate] = []
    removed = 0
    for pr in provider_rates:
        pr_mid = _provider_mid(pr)
        if _bps_diff(pr_mid, med) <= outlier_threshold_bps:
            kept.append(pr)
        else:
            removed += 1

    if kept:
        mids_kept = [_provider_mid(r) for r in kept]
        mid_rate = median(mids_kept)
    else:
        # If all were outliers fall back to original set
        kept = provider_rates
        mids_kept = mids
        mid_rate = med

    # Dispersion in bps
    min_mid = min(mids_kept)
    max_mid = max(mids_kept)
    dispersion_bps = ((max_mid - min_mid) / min_mid) * 10000.0 if min_mid > 0 else 0.0

    # Freshness: based on latest timestamp vs TTL
    latest_ts = max(pr.timestamp for pr in kept)
    age = datetime.now(timezone.utc) - latest_ts
    fresh = age <= timedelta(seconds=cache_ttl_seconds)

    # Best bid = highest; best ask = lowest
    bids = [pr.bid for pr in kept if pr.bid is not None and pr.bid > 0]
    asks = [pr.ask for pr in kept if pr.ask is not None and pr.ask > 0]
    best_bid = max(bids) if bids else None
    best_ask = min(asks) if asks else None
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

    notes: List[str] = []
    if removed:
        notes.append(f"Removed {removed} outlier(s) > {outlier_threshold_bps} bps from median")
    if dispersion_bps > dispersion_warn_bps:
        notes.append("High dispersion between providers")
    if not fresh:
        notes.append("Data not fresh (beyond cache TTL)")

    qm = QualityMetrics(
        sources_success=success,
        sources_total=total,
        dispersion_bps=dispersion_bps,
        fresh=fresh,
        notes=notes,
    )

    return mid_rate, best_bid, best_ask, spread, latest_ts, qm, kept

