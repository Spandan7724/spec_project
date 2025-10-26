"""Calendar URL collector using Serper general search."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.data_collection.market_intelligence.serper_client import (
    SerperClient,
    SerperSearchResult,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class CalendarCollector:
    """Collect economic calendar URLs using generalized search."""

    CURRENCY_NAMES = {
        "USD": "US Dollar",
        "EUR": "Euro",
        "GBP": "British Pound",
        "JPY": "Japanese Yen",
        "CHF": "Swiss Franc",
        "CAD": "Canadian Dollar",
        "AUD": "Australian Dollar",
        "NZD": "New Zealand Dollar",
    }

    def __init__(self, serper_client: SerperClient):
        self.serper = serper_client

    async def collect_calendar_urls(
        self, currency: str, month: Optional[str] = None, year: Optional[int] = None, num_results: int = 10
    ) -> List[SerperSearchResult]:
        # Default to current month/year
        now = datetime.now()
        if not month:
            month = now.strftime("%B")
        if not year:
            year = now.year

        currency_name = self.CURRENCY_NAMES.get(currency, currency)
        query = f'("{currency}" OR "{currency_name}") economic calendar events {month} {year}'

        logger.info("Collecting calendar URLs", extra={"currency": currency, "query": query})
        results = await self.serper.search(query, num_results=num_results)
        return results

