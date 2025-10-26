from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from src.decision.models import DecisionRequest


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


class ConfidenceAggregator:
    """Aggregate confidence from market, intelligence, and prediction with penalties."""

    def aggregate_confidence(
        self, request: DecisionRequest, utility_scores: Dict[str, float] | None = None, is_heuristic: bool = False
    ) -> Dict[str, Any]:
        comp: Dict[str, float] = {}
        penalties: list[str] = []

        # Components
        comp["market"] = self._calculate_market_confidence(request.market)
        comp["intelligence"] = self._calculate_intelligence_confidence(request.intelligence)
        comp["prediction"] = self._calculate_prediction_confidence(request.prediction)

        # Weights
        if comp["prediction"] is not None:
            w_market, w_intel, w_pred = 0.3, 0.2, 0.5
            base = (
                (comp["market"] or 0.0) * w_market
                + (comp["intelligence"] or 0.0) * w_intel
                + (comp["prediction"] or 0.0) * w_pred
            )
        else:
            # redistribute without prediction
            w_market, w_intel = 0.6, 0.4
            base = (comp["market"] or 0.0) * w_market + (comp["intelligence"] or 0.0) * w_intel

        # Global penalties
        missing = [k for k, v in comp.items() if v is None]
        if missing:
            base -= 0.05 * len(missing)
            penalties.append(f"missing_components:{len(missing)}")

        warn_n = len(request.warnings or [])
        if warn_n >= 2:
            dec = 0.1 * (warn_n // 2)
            base -= dec
            penalties.append(f"warnings_penalty:{dec:.2f}")

        # Staleness: if prediction timestamp exists and > 6h
        pred = request.prediction or {}
        ts = _parse_iso(pred.get("timestamp"))
        if ts is not None and datetime.now(timezone.utc) - ts > timedelta(hours=6):
            base -= 0.05
            penalties.append("stale_prediction")

        # Utility spread penalty: if provided
        if utility_scores:
            vals = sorted(utility_scores.values(), reverse=True)
            if len(vals) >= 2 and abs(vals[0] - vals[1]) < 0.05:
                base -= 0.1
                penalties.append("low_utility_spread")

        # Heuristic fallback penalty
        if is_heuristic:
            base -= 0.2
            penalties.append("heuristic_penalty")

        overall = max(0.0, min(1.0, base))
        return {
            "overall_confidence": overall,
            "component_confidences": {k: (0.0 if v is None else v) for k, v in comp.items()},
            "penalties_applied": penalties,
        }

    # ---- per component ----
    @staticmethod
    def _calculate_market_confidence(market: Optional[Dict[str, Any]]) -> Optional[float]:
        if not market:
            return None
        base = 0.5
        q = market.get("quality") or {}
        if q.get("fresh") and (q.get("dispersion_bps") is not None) and q.get("dispersion_bps") < 50.0:
            base += 0.2
        # indicator coherence
        ind = market.get("indicators") or {}
        rsi = ind.get("rsi_14")
        macd = ind.get("macd")
        coherent = (isinstance(rsi, (int, float)) and isinstance(macd, (int, float)) and ((rsi > 60 and macd > 0) or (rsi < 40 and macd < 0)))
        if coherent:
            base += 0.1
        return min(1.0, base)

    @staticmethod
    def _calculate_intelligence_confidence(intelligence: Optional[Dict[str, Any]]) -> Optional[float]:
        if not intelligence:
            return None
        base = 0.4
        news = intelligence.get("news") or {}
        conf = (news.get("confidence") or "low").lower()
        if conf in {"high", "medium"}:
            base += 0.1 if conf == "medium" else 0.2
        cal = intelligence.get("calendar") or {}
        if cal.get("next_high_event") is not None:
            base += 0.2
        return min(1.0, base)

    @staticmethod
    def _calculate_prediction_confidence(prediction: Optional[Dict[str, Any]]) -> Optional[float]:
        if not prediction:
            return None
        val = prediction.get("confidence")
        if isinstance(val, (int, float)):
            return float(val)
        return None

