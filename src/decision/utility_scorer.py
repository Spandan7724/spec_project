from __future__ import annotations

import math
from typing import Dict, Optional

from src.decision.config import DecisionConfig, RiskProfile
from src.decision.contracts import (
    conversion_change_pct,
    intelligence_improvement_pct,
    prediction_is_fresh,
    relative_atr_pct,
    select_prediction,
    upcoming_events,
)
from src.decision.models import DecisionRequest


class UtilityScorer:
    """Scores actions using a configurable multi-criteria utility model.

    Utility formula (percent units throughout):
        utility = w_profit * expected_improvement
                - w_risk   * risk_penalty
                - w_cost   * transaction_cost
                + w_urgency* urgency_fit
    """

    def __init__(self, config: DecisionConfig):
        self.config = config

    def score_actions(self, request: DecisionRequest) -> Dict[str, float]:
        profile = self._get_profile(request.risk_tolerance)

        improvement = self._get_expected_improvement(request)
        risk_penalty = self._calculate_risk_penalty(request, profile)
        tx_cost = self._get_transaction_cost(request)

        scores: Dict[str, float] = {}
        scores["convert_now"] = self._score_convert_now(profile, improvement, risk_penalty, tx_cost, request)
        scores["staged_conversion"] = self._score_staged(profile, improvement, risk_penalty, tx_cost, request)
        scores["wait"] = self._score_wait(profile, improvement, risk_penalty, tx_cost, request)
        return scores

    # Internal scoring methods
    def _score_convert_now(
        self,
        profile: RiskProfile,
        improvement: float,
        risk_penalty: float,
        tx_cost: float,
        request: DecisionRequest,
    ) -> float:
        w = profile.weights
        urgency_fit = self._get_urgency_fit("convert_now", request.urgency)
        # Converting now avoids a forecast loss that would be incurred by waiting.
        expected_improvement = max(0.0, -improvement)
        return (
            w.profit * expected_improvement
            - w.risk * risk_penalty
            - w.cost * tx_cost
            + w.urgency * urgency_fit
        )

    def _score_staged(
        self,
        profile: RiskProfile,
        improvement: float,
        risk_penalty: float,
        tx_cost: float,
        request: DecisionRequest,
    ) -> float:
        w = profile.weights
        urgency_fit = self._get_urgency_fit("staged_conversion", request.urgency)
        # Staging captures about half the improvement and reduces risk by ~40%
        staged_improvement = 0.5 * max(0.0, improvement)
        staged_risk = max(0.0, risk_penalty * 0.6)
        staged_multiplier = self.config.costs.staging_cost_multiplier
        staged_cost = tx_cost * staged_multiplier
        return (
            w.profit * staged_improvement
            - w.risk * staged_risk
            - w.cost * staged_cost
            + w.urgency * urgency_fit
        )

    def _score_wait(
        self,
        profile: RiskProfile,
        improvement: float,
        risk_penalty: float,
        tx_cost: float,
        request: DecisionRequest,
    ) -> float:
        w = profile.weights
        urgency_fit = self._get_urgency_fit("wait", request.urgency)
        return (
            w.profit * improvement
            - w.risk * risk_penalty
            - w.cost * tx_cost
            + w.urgency * urgency_fit
        )

    # Components
    def _get_profile(self, risk_tolerance: Optional[str]) -> RiskProfile:
        key = (risk_tolerance or "moderate").lower()
        return self.config.risk_profiles.get(key) or self.config.risk_profiles["moderate"]

    def _get_expected_improvement(self, request: DecisionRequest) -> float:
        """Return expected improvement in percent (e.g., 0.3 for +0.3%).

        Priority: Prediction → Intelligence bias → Technical (RSI/MACD) → 0.0
        """
        # 1) Prediction
        if request.prediction and isinstance(request.prediction, dict):
            is_fresh = prediction_is_fresh(
                request.prediction, self.config.thresholds.max_prediction_age_hours
            )
            _, item = select_prediction(request.prediction, request.timeframe_days)
            if is_fresh and item and isinstance(item, dict):
                val = item.get("mean_change_pct")
                if isinstance(val, (int, float)):
                    return conversion_change_pct(request, float(val))
        # 2) Pair-directional intelligence bias → percent
        if request.intelligence and isinstance(request.intelligence, dict):
            improvement = intelligence_improvement_pct(request.intelligence)
            if improvement is not None:
                return conversion_change_pct(request, improvement)
        # 3) Technical (RSI/MACD)
        rsi = None
        macd = None
        if request.market:
            ind = request.market.get("indicators", {})
            rsi = ind.get("rsi_14")
            macd = ind.get("macd")
        if isinstance(rsi, (int, float)):
            if rsi < 40:
                return conversion_change_pct(request, 0.15)
            if rsi > 60:
                return conversion_change_pct(request, -0.15)
        if isinstance(macd, (int, float)):
            if macd > 0:
                return conversion_change_pct(request, 0.1)
            if macd < 0:
                return conversion_change_pct(request, -0.1)
        return 0.0

    def _calculate_risk_penalty(self, request: DecisionRequest, profile: RiskProfile) -> float:
        """Combine volatility and event-based penalty (percent units)."""
        vol_pen = 0.0
        if request.market:
            atr_pct = relative_atr_pct(request.market)
            if atr_pct is not None:
                vol_pen = atr_pct * profile.volatility_penalty_multiplier

        event_pen = 0.0
        if request.intelligence and isinstance(request.intelligence, dict):
            events = upcoming_events(request.intelligence)
            nearest_high = None
            for ev in events:
                if (ev or {}).get("importance") == "high":
                    d = ev.get("days_until")
                    if isinstance(d, (int, float)):
                        nearest_high = d if nearest_high is None else min(nearest_high, d)
            if nearest_high is not None:
                # Exponential penalty: larger when event is close
                event_pen = 0.5 * math.exp(-max(0.0, float(nearest_high)))

        return max(0.0, vol_pen + event_pen)

    def _get_transaction_cost(self, request: DecisionRequest) -> float:
        spread_bps = request.spread_bps if request.spread_bps is not None else self.config.costs.default_spread_bps
        fee_bps = request.fee_bps if request.fee_bps is not None else self.config.costs.default_fee_bps
        total_bps = float(spread_bps) + float(fee_bps)
        return total_bps / 100.0  # convert bps to percent

    def _get_urgency_fit(self, action: str, urgency: str) -> float:
        u = (urgency or "normal").lower()
        if u == "urgent":
            return {"convert_now": 0.5, "staged_conversion": 0.2, "wait": -0.3}.get(action, 0.0)
        if u == "flexible":
            return {"convert_now": -0.1, "staged_conversion": 0.2, "wait": 0.4}.get(action, 0.0)
        # normal
        return {"convert_now": 0.2, "staged_conversion": 0.3, "wait": 0.2}.get(action, 0.0)

