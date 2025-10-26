from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.decision.config import StagingConfig, DecisionConfig, CostConfig
from src.decision.models import DecisionRequest, StagedPlan, TrancheSpec


class StagingPlanner:
    """Generate multi-tranche staged conversion plans with event awareness."""

    def __init__(self, staging_config: StagingConfig, cost_config: Optional[CostConfig] = None):
        self.cfg = staging_config
        self.costs = cost_config

    def create_staged_plan(self, request: DecisionRequest) -> StagedPlan:
        tf = max(0, int(request.timeframe_days))
        num_tranches = self._determine_tranche_count(tf)

        # Percentages
        percentages = self._get_sizing_pattern(request.urgency, num_tranches)
        if len(percentages) != num_tranches:
            # normalize to required length
            if num_tranches == 2:
                percentages = self.cfg.urgent_pattern if request.urgency == "urgent" else self.cfg.normal_pattern
            else:
                # derive equal split
                eq = [round(100.0 / num_tranches, 2) for _ in range(num_tranches - 1)]
                eq.append(round(100.0 - sum(eq), 2))
                percentages = [p for p in eq]

        # Execution schedule
        days = self._calculate_execution_schedule(num_tranches, tf)
        # Event avoidance
        events = (request.intelligence or {}).get("upcoming_events") or []
        days_adj = self._adjust_for_events(days, events, tf)

        # If too constrained (duplicates or out of bounds), reduce tranches progressively
        unique_days = sorted(set(days_adj))
        while len(unique_days) < num_tranches and num_tranches > 1:
            num_tranches -= 1
            percentages = self._get_sizing_pattern(request.urgency, num_tranches)
            days = self._calculate_execution_schedule(num_tranches, tf)
            days_adj = self._adjust_for_events(days, events, tf)
            unique_days = sorted(set(days_adj))

        # Normalize percentages to sum 100
        s = sum(percentages)
        if abs(s - 100.0) > 1e-6:
            # Scale and correct rounding on last tranche
            percentages = [round(p * 100.0 / s, 2) for p in percentages]
            if percentages:
                percentages[-1] = round(100.0 - sum(percentages[:-1]), 2)

        # Build tranche specs
        tranches = self._generate_tranche_specs(days_adj[:num_tranches], percentages, events)

        # Extra cost approximation
        spread_bps = request.spread_bps
        fee_bps = request.fee_bps
        extra_cost_bps = 0.0
        if self.costs is not None:
            sp = spread_bps if spread_bps is not None else self.costs.default_spread_bps
            # Approx difference vs single conversion, using staging multiplier
            extra_cost_bps = max(0.0, (num_tranches - 1) * float(sp) * float(self.costs.staging_cost_multiplier))

        spacing_days = 0.0
        if len(tranches) > 1:
            diffs = [tranches[i].execute_day - tranches[i - 1].execute_day for i in range(1, len(tranches))]
            if diffs:
                spacing_days = float(sum(diffs)) / float(len(diffs))

        benefit = self._build_benefit_text(request.urgency, events)
        return StagedPlan(
            num_tranches=len(tranches),
            tranches=tranches,
            spacing_days=spacing_days,
            total_extra_cost_bps=round(extra_cost_bps, 2),
            benefit=benefit,
        )

    # ----- internals -----
    def _determine_tranche_count(self, timeframe_days: int) -> int:
        if timeframe_days <= 5:
            return max(2, self.cfg.short_timeframe_tranches)
        return min(self.cfg.max_tranches, self.cfg.long_timeframe_tranches)

    def _get_sizing_pattern(self, urgency: str, num_tranches: int) -> List[float]:
        u = (urgency or "normal").lower()
        if u == "urgent":
            if num_tranches == 2:
                return [round(x * 100, 2) for x in self.cfg.urgent_pattern[:2]]
            elif num_tranches == 3:
                # front-loaded 50/30/20
                return [50.0, 30.0, 20.0]
        # normal/flexible → equal split via config pattern for 2 or 3
        if num_tranches == 2:
            pattern = self.cfg.normal_pattern
            return [round(x * 100, 2) for x in pattern[:2]]
        # 3-tranche equal
        eq = [round(100.0 / num_tranches, 2) for _ in range(num_tranches - 1)]
        eq.append(round(100.0 - sum(eq), 2))
        return eq

    def _calculate_execution_schedule(self, num_tranches: int, timeframe_days: int) -> List[int]:
        if num_tranches <= 1:
            return [0]
        step = max(self.cfg.min_spacing_days, timeframe_days / float(num_tranches))
        days = []
        for i in range(num_tranches):
            d = int(round(i * step))
            d = min(max(0, d), timeframe_days)
            # ensure strictly non-decreasing sequence
            if days and d <= days[-1]:
                d = min(timeframe_days, days[-1] + self.cfg.min_spacing_days)
            days.append(d)
        # always try to land last tranche near end of window
        if days:
            days[-1] = min(timeframe_days, max(days[-1], timeframe_days))
        return days

    def _adjust_for_events(self, execution_days: List[int], events: List[Dict], timeframe_days: int) -> List[int]:
        adjusted = list(execution_days)
        # find high-impact events within timeframe
        highs = [e for e in events if (e or {}).get("importance") == "high" and isinstance((e or {}).get("days_until"), (int, float)) and 0 <= e.get("days_until") <= timeframe_days]
        if not highs:
            return adjusted
        for idx, d in enumerate(adjusted):
            for ev in highs:
                evd = float(ev.get("days_until"))
                if abs(d - evd) <= 0.5:  # avoid within 12 hours
                    # shift to day after event
                    nd = int(math.ceil(evd + 1.0))
                    nd = min(max(0, nd), timeframe_days)
                    # keep spacing at least min_spacing_days from previous tranche
                    if idx > 0 and nd <= adjusted[idx - 1]:
                        nd = min(timeframe_days, adjusted[idx - 1] + self.cfg.min_spacing_days)
                    adjusted[idx] = nd
        # Ensure monotonic sequence
        for i in range(1, len(adjusted)):
            if adjusted[i] < adjusted[i - 1]:
                adjusted[i] = min(timeframe_days, adjusted[i - 1] + self.cfg.min_spacing_days)
        return adjusted

    def _generate_tranche_specs(
        self, execution_days: List[int], percentages: List[float], events: List[Dict]
    ) -> List[TrancheSpec]:
        tranches: List[TrancheSpec] = []
        for i, (d, p) in enumerate(zip(execution_days, percentages), start=1):
            rationale = "Initial conversion" if i == 1 else ("Final tranche" if i == len(execution_days) else "Intermediate tranche")
            # If event present after a tranche, mention it
            nearest = None
            nearest_name = None
            for ev in events or []:
                if (ev or {}).get("importance") == "high" and isinstance((ev or {}).get("days_until"), (int, float)):
                    diff = ev["days_until"] - d
                    if diff >= 0 and (nearest is None or diff < nearest):
                        nearest = diff
                        nearest_name = ev.get("event_name")
            if nearest_name is not None:
                rationale += f"; planned before {nearest_name}"
            tranches.append(TrancheSpec(tranche_number=i, percentage=round(p, 2), execute_day=int(d), rationale=rationale))
        return tranches

    def _build_benefit_text(self, urgency: str, events: List[Dict]) -> str:
        parts = ["Reduces concentration risk via diversification"]
        if events:
            names = [e.get("event_name") for e in events if (e or {}).get("importance") == "high" and e.get("event_name")]
            if names:
                parts.append(f"avoids concentration around {', '.join(names)}")
        if urgency == "urgent":
            parts.append("front-loaded to meet urgency")
        return ", ".join(parts)

