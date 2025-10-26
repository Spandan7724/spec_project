import pytest

from src.data_collection.market_intelligence.serper_client import SerperSearchResult, SerperNewsResult
from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
from src.data_collection.market_intelligence.news_collector import NewsCollector


class FakeSerper:
    async def search(self, query: str, num_results: int = 10):
        return [
            SerperSearchResult(title="Cal 1", url="https://example.com/cal", snippet="...", position=1),
            SerperSearchResult(title="Cal 2", url="https://example.com/cal2", snippet="...", position=2),
        ]

    async def search_news(self, query: str, time_range: str = "qdr:d", num_results: int = 20):
        return [
            SerperNewsResult(title="N1", url="https://www.reuters.com/a", source="Reuters", snippet="...", date="2025-10-24", position=1),
            SerperNewsResult(title="N2", url="https://www.bloomberg.com/b", source="Bloomberg", snippet="...", date="2025-10-24", position=2),
        ]


@pytest.mark.asyncio
async def test_calendar_collector_basic():
    cc = CalendarCollector(FakeSerper())
    res = await cc.collect_calendar_urls("USD")
    assert len(res) >= 1
    assert res[0].url.startswith("https://example.com/")


@pytest.mark.asyncio
async def test_news_collector_basic():
    nc = NewsCollector(FakeSerper())
    out = await nc.collect_pair_news("USD", "EUR", hours_back=24)
    assert set(out.keys()) == {"base", "quote"}
    assert len(out["base"]) >= 1
    assert len(out["quote"]) >= 1

