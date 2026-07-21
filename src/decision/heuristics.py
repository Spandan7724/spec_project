from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.decision.config import DecisionConfig
from src.decision.contracts import (
    conversion_direction_multiplier,
    nearest_high_event_days,
)
from src.decision.models import DecisionRequest


class HeuristicDecisionMaker:
    """Rule-based decision fallback when predictions are unavailable or unreliable."""

    def __init__(self, config: DecisionConfig):
        self.config = config

    def make_heuristic_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        # Event gating first
        gated = self._check_event_gating(request)
        if gated is not None:
            action = gated
            method = "event_gating"
            signals = {"event": True}
            confidence = 0.55
        else:
            # Momentum-based
            action, method, signals = self._momentum_rules(request)
            # Urgency override
            action = self._apply_urgency_override(action, request.urgency, signals)
            confidence = self._calculate_heuristic_confidence(method, signals)

        rationale = self._generate_heuristic_rationale(action, method, signals)
        return {
            "action": action,
            "confidence": confidence,
            "rationale": rationale,
            "source": "heuristic_fallback",
            "method": method,
        }

    # --- Event gating ---
    def _check_event_gating(self, request: DecisionRequest) -> Optional[str]:
        # Determine nearest high-impact event in days
        days_until = self._nearest_high_event_days(request.intelligence)
        if days_until is None:
            return None
        rt = (request.risk_tolerance or "moderate").lower()
        threshold = self.config.risk_profiles.get(rt, self.config.risk_profiles["moderate"]).event_proximity_threshold_days
        if days_until <= threshold:
            if rt == "conservative":
                return "wait"
            if rt == "moderate":
                # Prefer staged if timeframe allows
                return "staged_conversion" if request.timeframe_days >= self.config.thresholds.staged_min_timeframe_days else "wait"
            # aggressive
            return "convert_now"
        return None

    @staticmethod
    def _nearest_high_event_days(intelligence: Optional[Dict[str, Any]]) -> Optional[float]:
        return nearest_high_event_days(intelligence)

    # --- Momentum rules ---
    def _momentum_rules(self, request: DecisionRequest) -> Tuple[str, str, Dict[str, Any]]:
        ind = (request.market or {}).get("indicators", {})
        rsi = ind.get("rsi_14")
        macd_val = ind.get("macd")
        macd_sig = ind.get("macd_signal")
        regime = (request.market or {}).get("regime", {})
        trend = regime.get("trend_direction")
        bias = regime.get("bias")
        direction = conversion_direction_multiplier(request)

        def action_for(raw_pair_signal: float) -> str:
            return "wait" if raw_pair_signal * direction > 0 else "convert_now"

        # RSI heuristic
        if isinstance(rsi, (int, float)):
            if rsi >= 70:
                return action_for(-1.0), "rsi_overbought", {"rsi": rsi}
            if rsi <= 30:
                return action_for(1.0), "rsi_oversold", {"rsi": rsi}

        # MACD heuristic
        if isinstance(macd_val, (int, float)) and isinstance(macd_sig, (int, float)):
            if macd_val > macd_sig:
                return action_for(1.0), "macd_bullish", {"macd": macd_val, "signal": macd_sig}
            if macd_val < macd_sig:
                return action_for(-1.0), "macd_bearish", {"macd": macd_val, "signal": macd_sig}

        # Trend/bias heuristic
        if trend == "up" and bias in {"bullish", "neutral"}:
            return action_for(1.0), "trend_up", {"trend": trend, "bias": bias}
        if trend == "down" and bias in {"bearish", "neutral"}:
            return action_for(-1.0), "trend_down", {"trend": trend, "bias": bias}

        # Sideways or mixed → staged
        return "staged_conversion", "mixed_or_sideways", {"trend": trend, "bias": bias, "rsi": rsi, "macd": macd_val}

    # --- Urgency override ---
    def _apply_urgency_override(self, action: str, urgency: str, signals: Dict[str, Any]) -> str:
        u = (urgency or "normal").lower()
        if u != "urgent":
            return action
        # Urgent → prefer immediate action unless very negative signals present
        very_negative = False
        rsi = signals.get("rsi")
        if isinstance(rsi, (int, float)) and rsi >= 75:
            very_negative = True
        if not very_negative:
            return "convert_now" if action != "convert_now" else action
        # If very negative, compromise to staged
        return "staged_conversion"

    # --- Confidence & rationale ---
    @staticmethod
    def _calculate_heuristic_confidence(method: str, signals: Dict[str, Any]) -> float:
        if method == "event_gating":
            return 0.55
        if method in {"rsi_overbought", "rsi_oversold", "macd_bullish", "macd_bearish", "trend_up", "trend_down"}:
            return 0.5
        if method == "mixed_or_sideways":
            return 0.4
        return 0.3

    @staticmethod
    def _generate_heuristic_rationale(action: str, method: str, signals: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        if method == "event_gating":
            reasons.append("High-impact event approaching; adjusting for event risk")
        elif method == "rsi_overbought":
            reasons.append(f"RSI overbought ({signals.get('rsi'):.1f}) suggests near-term pullback")
        elif method == "rsi_oversold":
            reasons.append(f"RSI oversold ({signals.get('rsi'):.1f}) suggests bounce potential")
        elif method == "macd_bullish":
            reasons.append("MACD bullish crossover indicates upward momentum")
        elif method == "macd_bearish":
            reasons.append("MACD bearish crossover indicates downward momentum")
        elif method == "trend_up":
            reasons.append("Uptrend with supportive bias; waiting captures improvement")
        elif method == "trend_down":
            reasons.append("Downtrend with bearish bias; converting avoids worse rates")
        else:
            reasons.append("Mixed/sideways signals; staging diversifies risk")
        reasons.append(f"Heuristic method: {method}")
        return reasons

