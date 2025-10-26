from __future__ import annotations

from typing import Optional

from src.decision.config import CostConfig
from src.decision.models import CostEstimate, DecisionRequest


class CostCalculator:
    """Calculate transaction cost estimates in basis points (bps)."""

    def __init__(self, cost_config: CostConfig):
        self.cfg = cost_config

    def calculate_cost_estimate(self, request: DecisionRequest, is_staged: bool) -> CostEstimate:
        spread_bps = request.spread_bps if request.spread_bps is not None else self.cfg.default_spread_bps
        fee_bps = request.fee_bps if request.fee_bps is not None else self.cfg.default_fee_bps
        base_total = float(spread_bps) + float(fee_bps)
        multiplier = float(self.cfg.staging_cost_multiplier) if is_staged else 1.0
        total = base_total * multiplier
        return CostEstimate(
            spread_bps=float(spread_bps),
            fee_bps=float(fee_bps),
            total_bps=round(total, 4),
            staged_multiplier=multiplier,
        )

