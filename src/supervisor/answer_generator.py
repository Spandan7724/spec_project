from __future__ import annotations

"""Natural-language Q&A generator for TUI chat using the LLMManager.

Generates concise, natural answers grounded in the current recommendation and
extracted parameters. Falls back gracefully if LLM is unavailable.
"""

import json
import os
from typing import Any, Dict, Optional

from src.llm.manager import LLMManager
from src.llm.agent_helpers import chat_with_model_for_task
from src.supervisor.models import ExtractedParameters


SYSTEM_PROMPT = (
    "You are the Currency Assistant. Answer the user's question naturally and concisely based on the provided context. "
    "Only use facts from the context (parameters, recommendation, and evidence). If something is unknown, say you don't know. "
    "Prefer plain language. Summarize upcoming events, risks, costs, and timing in user-friendly terms where relevant."
)


class AnswerGenerator:
    def __init__(self, llm_manager: Optional[LLMManager] = None, use_llm: Optional[bool] = None) -> None:
        self.llm = llm_manager or LLMManager()
        # OFFLINE_DEMO=true disables LLM usage
        if use_llm is None:
            use_llm = str(os.getenv("OFFLINE_DEMO", "false")).strip().lower() in {"0", "false", "no"}
        self.use_llm = use_llm

    async def agenerate_answer(
        self,
        question: str,
        recommendation: Dict[str, Any],
        parameters: ExtractedParameters,
    ) -> str:
        if not self.use_llm:
            raise RuntimeError("LLM disabled via OFFLINE_DEMO or configuration")

        # Build compact, grounded context for the model
        # Extract select market fields from evidence if present
        ev_market = (recommendation.get("evidence") or {}).get("market") or {}

        ctx: Dict[str, Any] = {
            "parameters": {
                "currency_pair": parameters.currency_pair,
                "amount": parameters.amount,
                "risk_tolerance": parameters.risk_tolerance,
                "urgency": parameters.urgency,
                "timeframe": parameters.timeframe,
                "timeframe_days": parameters.timeframe_days,
            },
            "recommendation": {
                "action": recommendation.get("action"),
                "confidence": recommendation.get("confidence"),
                "timeline": recommendation.get("timeline"),
                "rationale": recommendation.get("rationale"),
                "risk_summary": recommendation.get("risk_summary"),
                "cost_estimate": recommendation.get("cost_estimate"),
                "current_rate": ev_market.get("mid_rate"),
                "bid": ev_market.get("bid"),
                "ask": ev_market.get("ask"),
                "spread": ev_market.get("spread"),
                "rate_timestamp": ev_market.get("rate_timestamp"),
                "expected_outcome": recommendation.get("expected_outcome"),
                "staged_plan": recommendation.get("staged_plan"),
                "utility_scores": recommendation.get("utility_scores"),
                "component_confidences": recommendation.get("component_confidences"),
                "meta": recommendation.get("meta"),
            },
            "evidence": recommendation.get("evidence") or {},
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Context (JSON):\n" + json.dumps(ctx, default=str) + "\n\n"
                    + "User question: " + question.strip()
                ),
            },
        ]

        # Use provider's main model for conversational Q&A (user-facing, requires good reasoning)
        response = await chat_with_model_for_task(messages, "conversation", self.llm)
        return response.content.strip() if response and response.content else ""
