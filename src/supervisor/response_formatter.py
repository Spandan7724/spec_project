from __future__ import annotations

from typing import Any, Dict, List


class ResponseFormatter:
    """Format final recommendation for user display."""

    def format_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Format recommendation as user-friendly text."""
        if recommendation.get("status") == "error":
            return self._format_error(recommendation)

        lines: List[str] = []
        lines.append("━" * 60)
        lines.append("RECOMMENDATION FOR CURRENCY CONVERSION")
        lines.append("━" * 60)
        lines.append("")

        # Action
        action_raw = recommendation.get("action", "").replace("_", " ").upper()
        lines.append(f"ACTION: {action_raw}")

        # Confidence
        confidence = recommendation.get("confidence")
        if isinstance(confidence, (int, float)):
            level = self._format_confidence_level(float(confidence))
            lines.append(f"CONFIDENCE: {float(confidence):.2f} ({level})")

        # Timeline
        timeline = recommendation.get("timeline")
        if timeline:
            lines.append(f"TIMELINE: {timeline}")
        lines.append("")

        # Staged plan
        if recommendation.get("staged_plan"):
            lines.append("STAGED CONVERSION PLAN:")
            plan = recommendation["staged_plan"] or {}
            for t in (plan.get("tranches") or []):
                num = t.get("tranche_number") or t.get("number")
                pct = t.get("percentage")
                day = t.get("execute_day")
                try:
                    pct_str = f"{float(pct):.0f}%" if pct is not None else "—"
                except Exception:
                    pct_str = f"{pct}%" if pct is not None else "—"
                lines.append(f"  • Tranche {num}: {pct_str} on Day {day}")
            lines.append("")

        # Expected outcome
        if recommendation.get("expected_outcome"):
            out = recommendation["expected_outcome"]
            lines.append("EXPECTED OUTCOME:")
            if isinstance(out.get("expected_rate"), (int, float)):
                lines.append(f"  • Expected rate: {out['expected_rate']:.4f}")
            if isinstance(out.get("expected_improvement_bps"), (int, float)):
                lines.append(f"  • Expected improvement: {out['expected_improvement_bps']:.1f} bps")
            if out.get("range_low") is not None and out.get("range_high") is not None:
                try:
                    lines.append(
                        f"  • Range: {float(out['range_low']):.4f} – {float(out['range_high']):.4f}"
                    )
                except Exception:
                    pass
            lines.append("")

        # Rationale
        rationale = recommendation.get("rationale") or []
        if rationale:
            lines.append("RATIONALE:")
            for i, reason in enumerate(rationale, 1):
                lines.append(f"  {i}. {reason}")
            lines.append("")

        # Risk
        if recommendation.get("risk_summary"):
            risk = recommendation["risk_summary"] or {}
            lvl = risk.get("risk_level")
            if lvl:
                lines.append(f"RISK ASSESSMENT: {str(lvl).title()}")
            # Optional details
            rv = risk.get("realized_vol_30d")
            var95 = risk.get("var_95")
            if isinstance(rv, (int, float)):
                lines.append(f"  • 30d realized vol: {rv:.2f}")
            if isinstance(var95, (int, float)):
                lines.append(f"  • 95% VaR: {var95:.2f}")
            if risk.get("event_risk"):
                lines.append(f"  • Event risk: {risk['event_risk']}")
            if risk.get("event_details"):
                lines.append(f"  • Details: {risk['event_details']}")
            lines.append("")

        # Costs
        if recommendation.get("cost_estimate"):
            cost = recommendation["cost_estimate"] or {}
            total = cost.get("total_bps")
            if isinstance(total, (int, float)):
                lines.append(f"ESTIMATED COSTS: {total:.1f} bps")
            # Optional breakdown
            if isinstance(cost.get("spread_bps"), (int, float)) or isinstance(cost.get("fee_bps"), (int, float)):
                sp = cost.get("spread_bps")
                fee = cost.get("fee_bps")
                parts = []
                if isinstance(sp, (int, float)):
                    parts.append(f"spread {sp:.1f}")
                if isinstance(fee, (int, float)):
                    parts.append(f"fee {fee:.1f}")
                if parts:
                    lines.append(f"  • Breakdown: {', '.join(parts)} bps")
            lines.append("")

        # Warnings
        warnings = recommendation.get("warnings") or []
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  ⚠️  {w}")
            lines.append("")

        # Next steps
        lines.append("Would you like to:")
        lines.append("  - Execute this recommendation")
        lines.append("  - Get alternative scenarios")
        lines.append("  - Start a new analysis")

        return "\n".join(lines)

    def _format_confidence_level(self, c: float) -> str:
        if c > 0.7:
            return "High"
        if c > 0.4:
            return "Moderate"
        return "Low"

    def _format_error(self, recommendation: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("━" * 60)
        lines.append("ERROR GENERATING RECOMMENDATION")
        lines.append("━" * 60)
        lines.append("")
        lines.append(f"Error: {recommendation.get('error', 'Unknown error')}")
        lines.append("")
        if recommendation.get("warnings"):
            lines.append("Additional details:")
            for w in recommendation["warnings"]:
                lines.append(f"  • {w}")
        lines.append("")
        lines.append("Please try again or contact support if the issue persists.")
        return "\n".join(lines)

