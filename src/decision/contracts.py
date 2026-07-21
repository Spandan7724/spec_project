"""Canonical adapters for data consumed by the decision engine.

Upstream agents intentionally keep their own rich payloads.  The decision
engine uses the helpers in this module so every component interprets rates,
events, horizons, freshness, and conversion direction in the same way.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def split_pair(pair: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not pair or "/" not in pair:
        return None, None
    base, quote = pair.upper().split("/", 1)
    return (base or None), (quote or None)


def conversion_direction_multiplier(request: Any) -> float:
    """Return +1 when a rising pair rate helps the requested conversion.

    ``currency_pair`` is the market rate (quote units per base unit), while
    ``source_currency`` is sold and ``target_currency`` is bought.  Existing
    requests that only provide a pair retain source=base, target=quote.
    """

    pair_base, pair_quote = split_pair(getattr(request, "currency_pair", None))
    source = getattr(request, "source_currency", None)
    target = getattr(request, "target_currency", None)
    source = str(source).upper() if source else pair_base
    target = str(target).upper() if target else pair_quote

    if pair_base == source and pair_quote == target:
        return 1.0
    if pair_base == target and pair_quote == source:
        return -1.0
    # The request model normally prevents this ambiguity.  Preserve the
    # historical source/target pair interpretation for partial callers.
    return 1.0


def conversion_change_pct(request: Any, pair_change_pct: float) -> float:
    """Translate a market-pair return into the user's conversion-rate return."""

    raw = float(pair_change_pct)
    if conversion_direction_multiplier(request) > 0:
        return raw
    gross_return = 1.0 + raw / 100.0
    if gross_return <= 0:
        # Invalid market prediction; avoid injecting an infinite utility.
        return 0.0
    return (1.0 / gross_return - 1.0) * 100.0


def relative_atr_pct(market: Optional[Dict[str, Any]]) -> Optional[float]:
    """Convert absolute ATR price units into percent of a reference rate."""

    if not market:
        return None
    indicators = market.get("indicators") or {}
    atr = _number(indicators.get("atr_14"))
    if atr is None or atr < 0:
        return None

    reference = None
    for candidate in (
        market.get("mid_rate"),
        market.get("current_rate"),
        indicators.get("bb_middle"),
        indicators.get("sma_20"),
    ):
        value = _number(candidate)
        if value is not None and value > 0:
            reference = value
            break
    if reference is None:
        return None
    return atr / reference * 100.0


def realized_volatility_pct(
    market: Optional[Dict[str, Any]], atr_pct: Optional[float] = None
) -> float:
    """Return the available 30-day volatility proxy in percent units."""

    indicators = (market or {}).get("indicators") or {}
    realized = _number(indicators.get("realized_vol_30d"))
    if realized is not None and realized >= 0:
        # Market-data indicators store return volatility as a decimal.
        return realized * 100.0
    # ATR is a one-session movement proxy.  Scale by sqrt(30), not the old
    # annualization-like factor of 16, when no return-based value is present.
    return max(0.0, float(atr_pct or 0.0)) * math.sqrt(30.0)


def upcoming_events(intelligence: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Normalize current and legacy event payloads to one upcoming-event shape."""

    if not intelligence or not isinstance(intelligence, dict):
        return []

    calendar = intelligence.get("calendar") or {}
    raw_events: Iterable[Any] = list(calendar.get("events_extracted") or []) + list(
        intelligence.get("upcoming_events") or []
    )
    normalized: list[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        days = _number(raw.get("days_until"))
        if days is None:
            proximity = _number(raw.get("proximity_minutes"))
            if proximity is not None:
                days = proximity / (60.0 * 24.0)
        if days is None or days < 0:
            continue

        name = raw.get("event_name") or raw.get("event")
        importance = str(raw.get("importance") or "low").lower()
        currency = raw.get("currency")
        key = (name, currency, round(days, 8), importance)
        if key in seen:
            continue
        seen.add(key)

        event = dict(raw)
        event.update(
            {
                "event_name": name,
                "event": name,
                "importance": importance,
                "days_until": days,
                "proximity_minutes": days * 60.0 * 24.0,
            }
        )
        normalized.append(event)

    return normalized


def nearest_high_event_days(intelligence: Optional[Dict[str, Any]]) -> Optional[float]:
    days = [
        float(event["days_until"])
        for event in upcoming_events(intelligence)
        if event.get("importance") == "high"
    ]
    return min(days) if days else None


def intelligence_improvement_pct(intelligence: Optional[Dict[str, Any]]) -> Optional[float]:
    """Map intelligence signals to an expected pair-rate change in percent.

    The legacy overall bias used a -10..10 scale.  Current pair bias is
    approximately -2..2 and policy bias is -1..1.  Current signals are
    normalized, averaged when both exist, and capped at +/-0.5%.
    """

    if not intelligence or not isinstance(intelligence, dict):
        return None

    legacy = _number(intelligence.get("overall_bias"))
    if legacy is not None:
        return max(-0.5, min(0.5, legacy * 0.05))

    signals: list[float] = []
    pair_bias = _number((intelligence.get("news") or {}).get("pair_bias"))
    if pair_bias is None:
        pair_bias = _number(intelligence.get("pair_bias"))
    if pair_bias is not None:
        signals.append(max(-1.0, min(1.0, pair_bias / 2.0)))

    policy_bias = _number(intelligence.get("policy_bias"))
    if policy_bias is not None:
        signals.append(max(-1.0, min(1.0, policy_bias)))

    if not signals:
        return None
    return 0.5 * (sum(signals) / len(signals))


def parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def prediction_age_hours(prediction: Optional[Dict[str, Any]]) -> Optional[float]:
    timestamp = parse_timestamp((prediction or {}).get("timestamp"))
    if timestamp is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600.0)


def prediction_is_fresh(
    prediction: Optional[Dict[str, Any]], max_age_hours: float
) -> bool:
    if not prediction:
        return False
    age = prediction_age_hours(prediction)
    # Older clients did not include a timestamp.  Treat absence as unknown,
    # while enforcing age whenever the upstream timestamp is present.
    return age is None or age <= float(max_age_hours)


def prediction_timeframe_bucket(
    user_timeframe: Optional[str],
    timeframe_days: Optional[int],
    use_user_timeframe: bool = True,
    default_timeframe: str = "1_day",
) -> str:
    """Resolve a categorical prediction bucket without hiding numeric windows."""

    if not use_user_timeframe:
        return default_timeframe or "1_day"
    if timeframe_days is None:
        return user_timeframe or default_timeframe or "1_day"
    if timeframe_days <= 0:
        return "immediate"
    if timeframe_days <= 1:
        return "1_day"
    if timeframe_days < 30:
        return "1_week"
    return "1_month"


def select_prediction(
    prediction: Optional[Dict[str, Any]], timeframe_days: int
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Select the exact or nearest available daily prediction horizon."""

    predictions = (prediction or {}).get("predictions") or {}
    if not isinstance(predictions, dict) or not predictions:
        return None, None

    exact = predictions.get(timeframe_days) or predictions.get(str(timeframe_days))
    if isinstance(exact, dict):
        return str(timeframe_days), exact

    candidates: list[tuple[int, str, Dict[str, Any]]] = []
    for key, payload in predictions.items():
        if not isinstance(payload, dict):
            continue
        try:
            horizon = int(key)
        except (TypeError, ValueError):
            continue
        candidates.append((horizon, str(key), payload))
    if not candidates:
        return None, None

    target = max(0, int(timeframe_days))
    # Prefer the closest bucket; on an exact-distance tie prefer the horizon
    # that covers the full requested window.
    _horizon, key, payload = min(
        candidates,
        key=lambda item: (abs(item[0] - target), item[0] < target, item[0]),
    )
    return key, payload
