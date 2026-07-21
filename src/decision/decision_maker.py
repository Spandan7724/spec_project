from __future__ import annotations

from typing import Dict, Optional

from src.decision.config import DecisionConfig
from src.decision.contracts import (
    conversion_change_pct,
    conversion_direction_multiplier,
    nearest_high_event_days,
    prediction_is_fresh,
    select_prediction,
    upcoming_events,
)
from src.decision.cost_calculator import CostCalculator
from src.decision.heuristics import HeuristicDecisionMaker
from src.decision.models import (
    DecisionRequest,
    DecisionResponse,
    ExpectedOutcome,
    RiskSummary,
    StagedPlan,
)
from src.decision.risk_calculator import RiskCalculator
from src.decision.staging_planner import StagingPlanner
from src.decision.utility_scorer import UtilityScorer
from src.decision.confidence_aggregator import ConfidenceAggregator


class DecisionMaker:
    """Orchestrates decision scoring, staging, cost, and confidence aggregation."""

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig.from_yaml()
        self.utility_scorer = UtilityScorer(self.config)
        self.staging_planner = StagingPlanner(self.config.staging, self.config.costs)
        self.cost_calculator = CostCalculator(self.config.costs)
        self.heuristic_maker = HeuristicDecisionMaker(self.config)
        self.confidence_agg = ConfidenceAggregator(
            max_prediction_age_hours=self.config.thresholds.max_prediction_age_hours
        )
        self.risk_calculator = RiskCalculator()

    def make_decision(self, request: DecisionRequest) -> DecisionResponse:
        # 1) Risk summary
        risk_summary = self.risk_calculator.calculate_risk_summary(request.market, request.intelligence)

        # 2) Decide path: utility vs heuristic (heuristic only if no prediction AND no intelligence)
        use_heuristic = self._should_use_heuristic(request)

        if not use_heuristic:
            # Utility-based path
            utility_scores = self.utility_scorer.score_actions(request)
            utility_scores = self._apply_hard_constraints(utility_scores, request)
            action = self._select_best_action(utility_scores)

            staged_plan = self._maybe_build_staging(action, request)
            expected = self._generate_expected_outcome(request, action, is_heuristic=False)
            is_staged = action == "staged_conversion"
            costs = self.cost_calculator.calculate_cost_estimate(request, is_staged=is_staged)
            conf_data = self.confidence_agg.aggregate_confidence(request, utility_scores, is_heuristic=False)
            timeline = self._generate_timeline(action, staged_plan, request.timeframe_days)

            rationale = self._generate_rationale(
                action=action,
                utility_scores=utility_scores,
                request=request,
                risk_summary=risk_summary,
                is_heuristic=False,
            )

            warnings = self._base_warnings(request)
            if risk_summary.event_risk in {"moderate", "high"}:
                warnings.append("High-impact event approaching")
            if risk_summary.risk_level in {"moderate", "high"}:
                warnings.append("Elevated volatility")

            return DecisionResponse(
                action=action,
                confidence=conf_data["overall_confidence"],
                timeline=timeline,
                staged_plan=staged_plan,
                expected_outcome=expected,
                risk_summary=risk_summary,
                cost_estimate=costs,
                rationale=rationale,
                warnings=warnings,
                utility_scores=utility_scores,
                component_confidences=conf_data["component_confidences"],
            )

        # Heuristic fallback path: rare and conservative by design
        h = self.heuristic_maker.make_heuristic_decision(request)
        action = h["action"]
        staged_plan = self._maybe_build_staging(action, request)
        expected = self._generate_expected_outcome(request, action, is_heuristic=True)
        is_staged = action == "staged_conversion"
        costs = self.cost_calculator.calculate_cost_estimate(request, is_staged=is_staged)
        conf_data = self.confidence_agg.aggregate_confidence(request, utility_scores={}, is_heuristic=True)
        timeline = self._generate_timeline(action, staged_plan, request.timeframe_days)
        rationale = ["Using heuristic fallback (prediction unavailable)"] + list(h.get("rationale", []))
        warnings = self._base_warnings(request) + ["Prediction unavailable, using heuristics"]

        return DecisionResponse(
            action=action,
            confidence=conf_data["overall_confidence"],
            timeline=timeline,
            staged_plan=staged_plan,
            expected_outcome=expected,
            risk_summary=risk_summary,
            cost_estimate=costs,
            rationale=rationale,
            warnings=warnings,
            utility_scores={},
            component_confidences=conf_data["component_confidences"],
        )

    # ---- Helpers ----
    def _should_use_heuristic(self, request: DecisionRequest) -> bool:
        # Global gate: allow disabling heuristics entirely
        if not getattr(self.config, "heuristics_enabled", False):
            return False
        # Reliability check for prediction
        pred = request.prediction or {}
        reliable = (
            isinstance(pred, dict)
            and str(pred.get("status", "success")).lower() == "success"
            and isinstance(pred.get("confidence"), (int, float))
            and float(pred["confidence"]) >= float(self.config.thresholds.min_model_confidence)
            and prediction_is_fresh(pred, self.config.thresholds.max_prediction_age_hours)
        )
        if reliable:
            return False
        # Trigger policy: in strict mode require both pred unreliable and intel missing
        policy = getattr(self.config, "heuristics_trigger_policy", "strict")
        # If intelligence is present, prefer utility scorer with fallback extraction over heuristics
        has_intel = bool(request.intelligence)
        if policy == "strict":
            return not has_intel
        # Relaxed mode falls back when intelligence is absent or too weak to
        # supply either a directional signal or meaningful event context.
        return not self._has_usable_intelligence(request.intelligence)

    def _apply_hard_constraints(self, utility_scores: Dict[str, float], request: DecisionRequest) -> Dict[str, float]:
        original = dict(utility_scores)
        constrained = dict(utility_scores)
        rt = (request.risk_tolerance or "moderate").lower()
        profile = self.config.risk_profiles.get(rt, self.config.risk_profiles["moderate"])
        blocked = -999.0

        # Waiting must clear the risk-profile-specific minimum benefit.
        improvement_bps = self.utility_scorer._get_expected_improvement(request) * 100.0
        if improvement_bps < profile.min_improvement_bps:
            constrained["wait"] = blocked

        # Staging is not feasible inside the configured minimum window.
        if request.timeframe_days < self.config.thresholds.staged_min_timeframe_days:
            constrained["staged_conversion"] = blocked

        # Do not recommend immediate conversion unless its absolute utility
        # clears the configured quality gate.
        if constrained.get("convert_now", blocked) < self.config.thresholds.convert_now_min_utility:
            constrained["convert_now"] = blocked

        days_until = nearest_high_event_days(request.intelligence)
        if days_until is not None:
            if rt == "conservative" and days_until <= profile.event_proximity_threshold_days:
                constrained["convert_now"] = blocked
            elif (
                rt == "moderate"
                and days_until <= self.config.thresholds.wait_event_proximity_days
                and constrained.get("convert_now", blocked) > blocked
            ):
                # Moderate users receive the planned soft event penalty.
                constrained["convert_now"] -= 0.25

        # Conflicting gates can otherwise eliminate every action (for example,
        # an immediate neutral request).  Preserve one feasible conservative
        # fallback while keeping every independently applicable gate enforced.
        if max(constrained.values()) <= blocked:
            if request.timeframe_days >= self.config.thresholds.staged_min_timeframe_days:
                constrained["staged_conversion"] = original["staged_conversion"]
            elif (request.urgency or "normal").lower() == "urgent" or request.timeframe_days <= 1:
                constrained["convert_now"] = original["convert_now"]
            else:
                constrained["wait"] = original["wait"]
        return constrained

    @staticmethod
    def _select_best_action(utility_scores: Dict[str, float]) -> str:
        return max(utility_scores, key=utility_scores.get)

    def _maybe_build_staging(self, action: str, request: DecisionRequest) -> Optional[StagedPlan]:
        if action != "staged_conversion":
            return None
        return self.staging_planner.create_staged_plan(request)

    def _generate_expected_outcome(self, request: DecisionRequest, action: str, is_heuristic: bool) -> ExpectedOutcome:
        # Pull the same exact/nearest horizon used by utility scoring.
        pred = request.prediction or {}
        is_fresh = prediction_is_fresh(pred, self.config.thresholds.max_prediction_age_hours)
        _, selected = select_prediction(pred, request.timeframe_days)
        item = selected if is_fresh and isinstance(selected, dict) else {}
        raw_mean_pct = float(item.get("mean_change_pct", 0.0))
        improvement_pct = conversion_change_pct(request, raw_mean_pct)
        latest_close = float(pred.get("latest_close", 0.0)) if isinstance(pred.get("latest_close"), (int, float)) else 0.0

        raw_expected_rate = latest_close * (1.0 + (raw_mean_pct / 100.0)) if latest_close > 0 else 0.0
        q = item.get("quantiles") if isinstance(item, dict) else None
        if isinstance(q, dict):
            low = float(q.get("p10", 0.0))
            high = float(q.get("p90", 0.0))
            raw_range_low = latest_close * (1.0 + (low / 100.0)) if latest_close > 0 else 0.0
            raw_range_high = latest_close * (1.0 + (high / 100.0)) if latest_close > 0 else 0.0
        else:
            raw_range_low = 0.0
            raw_range_high = 0.0

        if conversion_direction_multiplier(request) > 0:
            expected_rate = raw_expected_rate
            range_low = raw_range_low
            range_high = raw_range_high
        else:
            expected_rate = 1.0 / raw_expected_rate if raw_expected_rate > 0 else 0.0
            # Inversion reverses quantile ordering.
            range_low = 1.0 / raw_range_high if raw_range_high > 0 else 0.0
            range_high = 1.0 / raw_range_low if raw_range_low > 0 else 0.0

        return ExpectedOutcome(
            expected_rate=expected_rate,
            range_low=range_low,
            range_high=range_high,
            expected_improvement_bps=improvement_pct * 100.0,  # percent → bps
        )

    def _base_warnings(self, request: DecisionRequest) -> list[str]:
        warnings = list(request.warnings or [])
        if request.prediction and not prediction_is_fresh(
            request.prediction, self.config.thresholds.max_prediction_age_hours
        ):
            warnings.append("Prediction is stale and was ignored")
        return warnings

    @staticmethod
    def _has_usable_intelligence(intelligence: Optional[Dict]) -> bool:
        if not intelligence or not isinstance(intelligence, dict):
            return False
        news = intelligence.get("news") or {}
        pair_bias = news.get("pair_bias", intelligence.get("pair_bias"))
        confidence = str(news.get("confidence") or "low").lower()
        if isinstance(pair_bias, (int, float)) and confidence in {"medium", "high"}:
            return True
        policy_bias = intelligence.get("policy_bias")
        if isinstance(policy_bias, (int, float)) and abs(float(policy_bias)) > 0.0:
            return True
        return bool(upcoming_events(intelligence))

    @staticmethod
    def _generate_timeline(action: str, staged_plan: Optional[StagedPlan], timeframe_days: int) -> str:
        if action == "convert_now":
            return "Immediate execution recommended"
        if action == "staged_conversion":
            if staged_plan is not None:
                return f"Execute in {staged_plan.num_tranches} tranches over {timeframe_days} days"
            return f"Execute in tranches over {timeframe_days} days"
        return f"Wait up to {timeframe_days} days for better rate"

    def _generate_rationale(
        self,
        action: str,
        utility_scores: Dict[str, float],
        request: DecisionRequest,
        risk_summary: RiskSummary,
        is_heuristic: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not is_heuristic:
            reasons.append(f"Best utility action: {action}")
            reasons.append(f"Risk level: {risk_summary.risk_level}; event risk: {risk_summary.event_risk}")
            if request.urgency:
                reasons.append(f"Aligned with urgency: {request.urgency}")
        else:
            reasons.append("Heuristic fallback used due to missing prediction")
        return reasons[:5]
