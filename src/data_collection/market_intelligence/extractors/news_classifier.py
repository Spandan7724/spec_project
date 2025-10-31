from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import asyncio
from typing import Dict, List

from src.data_collection.market_intelligence.models import NewsClassification
from src.llm.agent_helpers import chat_with_model_for_task
from src.utils.decorators import retry, log_execution, timeout
from src.utils.logging import get_logger


logger = get_logger(__name__)


class NewsClassifier:
    """Classify news article sentiment and relevance using an LLM.

    Model: Uses {provider}_fast for fast, cost-effective classification.
    Rationale: News classification is a simple categorization task that doesn't
    require complex reasoning, making it ideal for the cheaper, faster model.
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    @retry(max_attempts=2, delay=1.0)
    @timeout(15.0)
    @log_execution(log_args=False, log_result=False)
    async def classify_article(
        self, title: str, snippet: str, url: str, source: str, currencies: List[str]
    ) -> NewsClassification:
        prompt = f"""Analyze this financial news article.

Title: {title}
Snippet: {snippet}
Currencies: {', '.join(currencies)}

Return JSON with:
relevance: per-currency 0.0-1.0
sentiment: per-currency -1.0..+1.0
quality_flags: clickbait, rumor_speculative, non_econ
"""

        messages = [
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ]
        # Use provider's fast model for simple classification
        resp = await chat_with_model_for_task(messages, "classification", self.llm_manager)
        content = resp.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            data = json.loads(content)
        except Exception:
            logger.warning("News classification JSON parse failed; returning neutral values")
            data = {
                "relevance": {c: 0.0 for c in currencies},
                "sentiment": {c: 0.0 for c in currencies},
                "quality_flags": {"clickbait": False, "rumor_speculative": False, "non_econ": False},
            }

        article_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        published_dt = None
        try:
            published_dt = datetime.now(timezone.utc)
        except Exception:
            published_dt = None

        return NewsClassification(
            article_id=article_id,
            url=url,
            source=source,
            title=title,
            published_utc=published_dt,
            relevance=data.get("relevance", {}),
            sentiment=data.get("sentiment", {}),
            quality_flags=data.get("quality_flags", {}),
        )

    async def classify_batch(self, articles: List[Dict[str, str]], currencies: List[str]) -> List[NewsClassification]:
        """Classify articles concurrently with a small concurrency cap."""
        sem = asyncio.Semaphore(10)  # Increased from 4

        async def _one(a: Dict[str, str]) -> NewsClassification:
            async with sem:
                return await self.classify_article(
                    title=a.get("title", ""),
                    snippet=a.get("snippet", ""),
                    url=a.get("url", a.get("link", "")),
                    source=a.get("source", ""),
                    currencies=currencies,
                )

        tasks = [asyncio.create_task(_one(a)) for a in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[NewsClassification] = []
        for r in results:
            if isinstance(r, Exception):
                # Skip failed item; continue processing others
                continue
            out.append(r)
        return out
