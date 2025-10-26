from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import os

import numpy as np

from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier
from src.data_collection.market_intelligence.models import NewsClassification
from src.config import get_config, load_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class PairNewsSnapshot:
    pair: str
    ts_utc: datetime
    sent_base: float
    sent_quote: float
    pair_bias: float
    confidence: str  # "high" | "medium" | "low"
    n_articles_used: int
    top_evidence: List[Dict[str, Any]]


class NewsAggregator:
    """Aggregate news sentiment for a currency pair."""

    MIN_RELEVANCE_THRESHOLD = 0.3

    def __init__(self, news_collector: NewsCollector, classifier: NewsClassifier):
        self.collector = news_collector
        self.classifier = classifier

    async def get_pair_snapshot(self, base: str, quote: str, hours_back: int = 24) -> PairNewsSnapshot:
        logger.info(
            "Aggregating news for pair",
            extra={"pair": f"{base}/{quote}", "hours_back": hours_back},
        )

        pair_news = await self.collector.collect_pair_news(base, quote, hours_back)

        # Convert SerperNewsResult objects to dicts for classifier
        def to_dict(item) -> Dict[str, str]:
            return {
                "title": getattr(item, "title", ""),
                "snippet": getattr(item, "snippet", ""),
                "url": getattr(item, "url", getattr(item, "link", "")),
                "source": getattr(item, "source", ""),
            }

        all_articles = [to_dict(a) for a in pair_news.get("base", [])] + [to_dict(a) for a in pair_news.get("quote", [])]

        # Deduplicate by URL
        seen = set()
        unique_articles: List[Dict[str, str]] = []
        for a in all_articles:
            u = a.get("url", "")
            if u and u not in seen:
                seen.add(u)
                unique_articles.append(a)

        # Limit number of articles to classify (configurable)
        try:
            cfg = get_config()
        except Exception:
            cfg = load_config()
        max_articles = int(cfg.get("agents.market_intelligence.max_articles", 10))
        # Env override for ad-hoc runs
        env_override = os.getenv("MI_MAX_ARTICLES")
        if env_override is not None:
            try:
                max_articles = int(env_override)
            except ValueError:
                pass
        if max_articles > 0:
            unique_articles = unique_articles[:max_articles]

        # Classify
        classifications = await self.classifier.classify_batch(unique_articles, currencies=[base, quote])

        # Aggregate
        sent_base, sent_quote, confidence, top_evidence = self._aggregate_sentiment(classifications, base, quote)
        pair_bias = sent_base - sent_quote

        snapshot = PairNewsSnapshot(
            pair=f"{base}/{quote}",
            ts_utc=datetime.now(timezone.utc),
            sent_base=float(sent_base),
            sent_quote=float(sent_quote),
            pair_bias=float(pair_bias),
            confidence=confidence,
            n_articles_used=len(classifications),
            top_evidence=top_evidence,
        )

        return snapshot

    def _aggregate_sentiment(
        self, classifications: List[NewsClassification], base: str, quote: str
    ) -> Tuple[float, float, str, List[Dict[str, Any]]]:
        if not classifications:
            return 0.0, 0.0, "low", []

        base_vals: List[float] = []
        quote_vals: List[float] = []

        for c in classifications:
            if c.relevance.get(base, 0) >= self.MIN_RELEVANCE_THRESHOLD:
                base_vals.append(c.sentiment.get(base, 0.0))
            if c.relevance.get(quote, 0) >= self.MIN_RELEVANCE_THRESHOLD:
                quote_vals.append(c.sentiment.get(quote, 0.0))

        sent_base = float(np.mean(base_vals)) if base_vals else 0.0
        sent_quote = float(np.mean(quote_vals)) if quote_vals else 0.0

        # Confidence heuristic based on article count and variance
        all_vals = base_vals + quote_vals
        variance = float(np.var(all_vals)) if all_vals else 1.0
        n = len(classifications)
        if n >= 10 and variance < 0.3:
            confidence = "high"
        elif n >= 5 and variance < 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        # Top evidence: highest max relevance
        top = sorted(
            classifications,
            key=lambda c: max(c.relevance.values()) if c.relevance else 0.0,
            reverse=True,
        )[:5]

        top_evidence = [
            {
                "title": c.title,
                "url": c.url,
                "source": c.source,
                "relevance": c.relevance,
                "sentiment": c.sentiment,
            }
            for c in top
        ]

        return sent_base, sent_quote, confidence, top_evidence
