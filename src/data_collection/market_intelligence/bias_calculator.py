from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from src.data_collection.market_intelligence.models import EconomicEvent


IMPORTANCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}


def _keyword_bias(event_name: str) -> float:
    """Very simple heuristic: assign sign based on keywords in event name."""
    s = event_name.lower()
    positive = ["hike", "hawkish", "strong", "surprise up", "beat", "hot", "accelerat"]
    negative = ["cut", "dovish", "weak", "miss", "cool", "decel", "slow"]
    if any(k in s for k in positive):
        return 1.0
    if any(k in s for k in negative):
        return -1.0
    return 0.0


def _proximity_weight(minutes_ahead: int) -> float:
    """Weight events closer in time more heavily (within 7 days)."""
    if minutes_ahead < 0:
        return 0.3  # already occurred, small residual weight
    days = minutes_ahead / (60 * 24)
    if days <= 1:
        return 1.0
    if days <= 3:
        return 0.8
    if days <= 7:
        return 0.6
    return 0.4


def calculate_policy_bias(events: List[EconomicEvent]) -> float:
    """Compute a simple policy bias score from events.

    Returns a value roughly in [-1, +1].
    """
    if not events:
        return 0.0

    now = datetime.now(timezone.utc)
    total = 0.0
    weight_sum = 0.0

    for e in events:
        imp_w = IMPORTANCE_WEIGHTS.get((e.importance or "low").lower(), 0.3)
        sign = _keyword_bias(e.event or "")
        minutes_ahead = int((e.when_utc - now).total_seconds() / 60)
        prox_w = _proximity_weight(minutes_ahead)
        w = imp_w * prox_w
        total += sign * w
        weight_sum += w

    return total / weight_sum if weight_sum > 0 else 0.0


def next_high_impact_event(events: List[EconomicEvent]) -> Optional[EconomicEvent]:
    future_high = [e for e in events if (e.importance or "").lower() == "high" and e.proximity_minutes > 0]
    if not future_high:
        return None
    return sorted(future_high, key=lambda e: e.proximity_minutes)[0]

