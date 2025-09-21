"""Risk assessment agent node for LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from data_collection.analysis.historical_data import HistoricalDataCollector

from ..state import AgentGraphState, RiskAssessment

logger = logging.getLogger(__name__)


class RiskAssessmentAgent:
    """Quantifies downside risk using historical distributions."""

    def __init__(self, historical_collector: HistoricalDataCollector | None = None) -> None:
        self.historical_collector = historical_collector or HistoricalDataCollector()

    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        request = state.request
        risk_assessment = RiskAssessment()

        try:
            dataset = await self.historical_collector.get_historical_data(
                request.currency_pair,
                days=max(90, request.timeframe_days * 3),
            )
            if not dataset:
                risk_assessment.errors.append("Historical dataset unavailable for risk computation")
                return state.with_risk(risk_assessment)

            df = dataset.to_dataframe()
            if df.empty or "close" not in df.columns:
                risk_assessment.errors.append("Historical price series empty")
                return state.with_risk(risk_assessment)

            returns = df["close"].pct_change().dropna()
            if returns.empty:
                risk_assessment.errors.append("Insufficient return history for risk analysis")
                return state.with_risk(risk_assessment)

            daily_vol = float(returns.std())
            annualized_vol = daily_vol * np.sqrt(252)
            risk_assessment.volatility = annualized_vol

            var_quantile = float(returns.quantile(0.05))
            risk_assessment.var_95 = var_quantile

            worst_case = float(returns.quantile(0.01))
            best_case = float(returns.quantile(0.99))
            expected = float(returns.mean())

            amount = request.amount
            scenarios: Dict[str, float] = {
                "best_case_pct": best_case,
                "expected_pct": expected,
                "worst_case_pct": worst_case,
                "var_95_pct": var_quantile,
                "var_95_amount": var_quantile * amount,
                "worst_case_amount": worst_case * amount,
            }
            risk_assessment.scenarios = scenarios

            if annualized_vol < 0.05:
                risk_assessment.risk_level = "low"
            elif annualized_vol < 0.12:
                risk_assessment.risk_level = "medium"
            else:
                risk_assessment.risk_level = "high"

            loss_at_var = -var_quantile * amount
            risk_assessment.summary = (
                f"Estimated annualized volatility {annualized_vol:.2%}; "
                f"95% one-day loss around {loss_at_var:.2f} {request.base_currency}."
            )
            risk_assessment.confidence = 0.55

            if risk_assessment.risk_level == "high":
                risk_assessment.hedging_notes.append("Consider staggering conversions or using limit orders")
            elif risk_assessment.risk_level == "medium":
                risk_assessment.hedging_notes.append("Monitor key economic releases before executing")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Risk agent failed to compute metrics", exc_info=exc)
            risk_assessment.errors.append(f"Risk calculation error: {exc}")

        return state.with_risk(risk_assessment)


async def run_risk_agent(
    state: AgentGraphState,
    agent: RiskAssessmentAgent | None = None,
) -> AgentGraphState:
    """Convenience coroutine for LangGraph nodes."""
    agent = agent or RiskAssessmentAgent()
    return await agent(state)
