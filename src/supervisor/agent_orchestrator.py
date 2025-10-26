from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.agentic.state import AgentState
from src.agentic.nodes.market_data import market_data_node
from src.agentic.nodes.market_intelligence import market_intelligence_node
from src.agentic.nodes.prediction import prediction_node
from src.agentic.nodes.decision import decision_node
from .models import ExtractedParameters


logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrate execution of Layer 1–3 nodes and extract a recommendation."""

    async def run_analysis(
        self, parameters: ExtractedParameters, correlation_id: str
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        state: AgentState = self._initialize_state(parameters, correlation_id)

        # Layer 1: Market Data + Market Intelligence (parallel)
        try:
            logger.info("[%s] Layer 1 start", correlation_id)
            md_task = market_data_node(state)
            mi_task = market_intelligence_node(state)
            md_res, mi_res = await asyncio.gather(md_task, mi_task, return_exceptions=True)

            if isinstance(md_res, Exception):
                logger.error("[%s] Market data failed: %s", correlation_id, md_res)
                warnings.append(f"Market data failed: {md_res}")
            else:
                state.update(md_res)
                if md_res.get("market_data_error"):
                    warnings.append(str(md_res.get("market_data_error")))

            if isinstance(mi_res, Exception):
                logger.error("[%s] Market intelligence failed: %s", correlation_id, mi_res)
                warnings.append(f"Market intelligence failed: {mi_res}")
            else:
                state.update(mi_res)
                if mi_res.get("intelligence_error"):
                    warnings.append(str(mi_res.get("intelligence_error")))

        except Exception as e:
            logger.error("[%s] Layer 1 error: %s", correlation_id, e)
            warnings.append(f"Layer 1 error: {e}")

        # Layer 2: Prediction
        try:
            logger.info("[%s] Layer 2 start", correlation_id)
            pred_res = await prediction_node(state)
            state.update(pred_res)
            if pred_res.get("prediction_error"):
                warnings.append(str(pred_res.get("prediction_error")))
        except Exception as e:
            logger.error("[%s] Prediction failed: %s", correlation_id, e)
            warnings.append(f"Price prediction failed: {e}")

        # Layer 3: Decision (sync)
        try:
            logger.info("[%s] Layer 3 start", correlation_id)
            dec_res = decision_node(state)
            state.update(dec_res)
        except Exception as e:
            logger.error("[%s] Decision failed: %s", correlation_id, e)
            return {"status": "error", "error": f"Decision engine failed: {e}", "warnings": warnings}

        recommendation = state.get("recommendation")
        if not recommendation:
            return {"status": "error", "error": "No decision generated", "warnings": warnings}

        # Merge warnings
        if isinstance(recommendation.get("warnings"), list):
            merged_warnings = list(recommendation["warnings"]) + warnings
        else:
            merged_warnings = warnings

        out: Dict[str, Any] = {
            "status": "success",
            "action": recommendation.get("action"),
            "confidence": recommendation.get("confidence"),
            "timeline": recommendation.get("timeline"),
            "rationale": recommendation.get("rationale", []),
            "warnings": merged_warnings,
            "metadata": {"correlation_id": correlation_id, "timestamp": datetime.utcnow().isoformat()},
        }

        # Optional fields
        for key in (
            "staged_plan",
            "expected_outcome",
            "risk_summary",
            "cost_estimate",
            "evidence",
            "meta",
            "utility_scores",
            "component_confidences",
        ):
            if recommendation.get(key) is not None:
                out[key] = recommendation[key]

        return out

    def _initialize_state(self, p: ExtractedParameters, correlation_id: str) -> AgentState:
        tf_days = p.timeframe_days
        # Build minimal AgentState expected by nodes
        state: AgentState = {
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow(),
            "user_query": "supervised_request",
            "currency_pair": p.currency_pair or "",
            "base_currency": p.base_currency or "",
            "quote_currency": p.quote_currency or "",
            "amount": p.amount,
            "risk_tolerance": (p.risk_tolerance or "moderate"),
            "urgency": (p.urgency or "normal"),
            "timeframe": (p.timeframe or "1_day"),
            "timeframe_days": tf_days or None,  # not in TypedDict but used by decision builder
            "warnings": [],
            "errors": [],
            "processing_stage": "initialized",
            "agent_metrics": {},
            # Init statuses
            "market_data_status": "pending",
            "intelligence_status": "pending",
            "prediction_status": "pending",
            "decision_status": "pending",
        }
        return state
