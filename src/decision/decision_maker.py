from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

from src.decision.config import DecisionConfig
from src.decision.cost_calculator import CostCalculator
from src.decision.heuristics import HeuristicDecisionMaker
from src.decision.models import (
    CostEstimate,
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
        self.confidence_agg = ConfidenceAggregator()
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

            warnings = list(request.warnings or [])
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
        warnings = list(request.warnings or []) + ["Prediction unavailable, using heuristics"]

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
        )
        if reliable:
            return False
        # Trigger policy: in strict mode require both pred unreliable and intel missing
        policy = getattr(self.config, "heuristics_trigger_policy", "strict")
        # If intelligence is present, prefer utility scorer with fallback extraction over heuristics
        has_intel = bool(request.intelligence)
        if policy == "strict":
            return not has_intel
        # relaxed: if prediction unreliable and intel weak/missing
        return not has_intel

    def _apply_hard_constraints(self, utility_scores: Dict[str, float], request: DecisionRequest) -> Dict[str, float]:
        # Conservative near high-impact events: block convert_now
        from src.decision.heuristics import HeuristicDecisionMaker

        rt = (request.risk_tolerance or "moderate").lower()
        if rt != "conservative":
            return utility_scores
        days_until = HeuristicDecisionMaker._nearest_high_event_days(request.intelligence)
        threshold = self.config.risk_profiles["conservative"].event_proximity_threshold_days
        if days_until is not None and days_until <= threshold:
            utility_scores = dict(utility_scores)
            utility_scores["convert_now"] = -999.0
        return utility_scores

    @staticmethod
    def _select_best_action(utility_scores: Dict[str, float]) -> str:
        return max(utility_scores, key=utility_scores.get)

    def _maybe_build_staging(self, action: str, request: DecisionRequest) -> Optional[StagedPlan]:
        if action != "staged_conversion":
            return None
        return self.staging_planner.create_staged_plan(request)

    def _generate_expected_outcome(self, request: DecisionRequest, action: str, is_heuristic: bool) -> ExpectedOutcome:
        # Pull pred for timeframe_days; fallback to 0
        pred = request.prediction or {}
        preds = pred.get("predictions") or {}
        item = preds.get(request.timeframe_days) or preds.get(str(request.timeframe_days)) or {}
        mean_pct = float(item.get("mean_change_pct", 0.0)) if isinstance(item, dict) else 0.0
        latest_close = float(pred.get("latest_close", 0.0)) if isinstance(pred.get("latest_close"), (int, float)) else 0.0

        expected_rate = latest_close * (1.0 + (mean_pct / 100.0)) if latest_close > 0 else 0.0
        q = item.get("quantiles") if isinstance(item, dict) else None
        if isinstance(q, dict):
            low = float(q.get("p10", 0.0))
            high = float(q.get("p90", 0.0))
            range_low = latest_close * (1.0 + (low / 100.0)) if latest_close > 0 else 0.0
            range_high = latest_close * (1.0 + (high / 100.0)) if latest_close > 0 else 0.0
        else:
            range_low = 0.0
            range_high = 0.0

        return ExpectedOutcome(
            expected_rate=expected_rate,
            range_low=range_low,
            range_high=range_high,
            expected_improvement_bps=mean_pct * 100.0,  # percent → bps
        )

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
