from __future__ import annotations

from src.llm.agent_helpers import chat_with_model_for_task
from src.utils.logging import get_logger


logger = get_logger(__name__)


class NarrativeGenerator:
    """Generate a concise narrative summary for a pair's news snapshot.

    Model: Uses {provider}_main for narrative generation.
    Rationale: Creating coherent, user-facing summaries requires good language
    understanding and generation capabilities, making the main model the better choice.
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def generate_narrative(self, snapshot) -> str:
        # Expect snapshot to have attributes/keys like pair, sent_base, sent_quote, pair_bias, confidence, n_articles_used, top_evidence
        pair = getattr(snapshot, "pair", None) or snapshot.get("pair")
        sent_base = getattr(snapshot, "sent_base", None) or snapshot.get("sent_base", 0.0)
        sent_quote = getattr(snapshot, "sent_quote", None) or snapshot.get("sent_quote", 0.0)
        pair_bias = getattr(snapshot, "pair_bias", None) or snapshot.get("pair_bias", 0.0)
        confidence = getattr(snapshot, "confidence", None) or snapshot.get("confidence", "low")
        n_articles = getattr(snapshot, "n_articles_used", None) or snapshot.get("n_articles_used", 0)
        top = getattr(snapshot, "top_evidence", None) or snapshot.get("top_evidence", [])

        bias_text = "bullish" if pair_bias > 0.2 else "bearish" if pair_bias < -0.2 else "neutral"
        headlines = "\n".join(f"- {e.get('title','')}" for e in top[:3])

        prompt = f"""Generate a concise 1-2 sentence summary of current news sentiment for {pair}.

Data:
- Base sentiment: {sent_base:+.2f}
- Quote sentiment: {sent_quote:+.2f}
- Pair bias: {pair_bias:+.2f} ({bias_text})
- Confidence: {confidence}
- Based on {n_articles} articles

Top headlines:
{headlines}
"""

        messages = [
            {"role": "system", "content": "You are a financial analyst. Be concise and objective."},
            {"role": "user", "content": prompt},
        ]
        # Use provider's main model for narrative generation
        resp = await chat_with_model_for_task(messages, "summarization", self.llm_manager)
        return resp.content.strip()

