from typing import Dict, Optional

from src.decision.contracts import relative_atr_pct, realized_volatility_pct, upcoming_events
from src.decision.models import RiskSummary


class RiskCalculator:
    """Calculate volatility and event-risk based summary (percent units)."""

    def calculate_risk_summary(
        self, market: Optional[Dict], intelligence: Optional[Dict]
    ) -> RiskSummary:
        vol_pct = relative_atr_pct(market) or 0.0

        realized_vol_30d = realized_volatility_pct(market, vol_pct)
        var_95 = 1.65 * vol_pct

        # Event risk classification
        event_risk = "none"
        event_details = None
        nearest = None
        if intelligence:
            events = upcoming_events(intelligence)
            for ev in events:
                if (ev or {}).get("importance") == "high":
                    d = ev.get("days_until")
                    if isinstance(d, (int, float)):
                        nearest = d if nearest is None else min(nearest, d)
                        event_details = ev.get("event_name") or event_details
        if nearest is not None:
            if nearest <= 1.0:
                event_risk = "high"
            elif nearest <= 3.0:
                event_risk = "moderate"
            else:
                event_risk = "low"

        # Overall risk level driven by event risk and volatility
        risk_level = "low"
        if event_risk in ("high", "moderate") or realized_vol_30d > 15.0:
            risk_level = "high" if event_risk == "high" or realized_vol_30d > 20.0 else "moderate"
            # If both are elevated strongly, escalate to high
            if realized_vol_30d > 25.0 or event_risk == "high":
                risk_level = "high"

        return RiskSummary(
            risk_level=risk_level,
            realized_vol_30d=realized_vol_30d,
            var_95=var_95,
            event_risk=event_risk,
            event_details=event_details,
        )

