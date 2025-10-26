"""News collector for currency-specific news using Serper news endpoint."""
from __future__ import annotations

from typing import Dict, List

from src.data_collection.market_intelligence.serper_client import (
    SerperClient,
    SerperNewsResult,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class NewsCollector:
    """Collect news articles for currencies and pairs."""

    def __init__(self, serper_client: SerperClient):
        self.serper = serper_client

    async def collect_currency_news(self, currency: str, hours_back: int = 24, num_results: int = 20) -> List[SerperNewsResult]:
        # time_range mapping: qdr:d ~ last day, qdr:w ~ last week
        tbs = "qdr:d" if hours_back <= 24 else "qdr:w"
        # simple query targeting currency economy/monetary topics
        query = f"({currency}) (currency OR rates OR inflation OR central bank OR economy)"
        logger.info("Collecting currency news", extra={"currency": currency, "tbs": tbs})
        return await self.serper.search_news(query, time_range=tbs, num_results=num_results)

    async def collect_pair_news(self, base: str, quote: str, hours_back: int = 24) -> Dict[str, List[SerperNewsResult]]:
        base_news = await self.collect_currency_news(base, hours_back=hours_back)
        quote_news = await self.collect_currency_news(quote, hours_back=hours_back)
        return {"base": base_news, "quote": quote_news}

