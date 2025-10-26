import pytest
from datetime import datetime, timezone

from src.data_collection.market_intelligence.intelligence_service import MarketIntelligenceService


@pytest.mark.asyncio
async def test_intelligence_service_integration(monkeypatch):
    # Fake Serper client methods used by collectors
    import src.data_collection.market_intelligence.intelligence_service as svc_mod
    import src.data_collection.market_intelligence.extractors.calendar_extractor as cal_mod

    class DummySerper:
        def __init__(self, *a, **k):
            pass

    # Patch collectors to use deterministic SerperClient-like behavior
    async def fake_search(self, query: str, num_results: int = 10):
        return []

    async def fake_collect_calendar_urls(self, currency: str, month=None, year=None):
        from src.data_collection.market_intelligence.serper_client import SerperSearchResult
        return [
            SerperSearchResult(title=f"{currency} Event", url="https://example.com", snippet=f"{currency} policy meeting", position=1)
        ]

    monkeypatch.setattr(svc_mod, "SerperClient", lambda *a, **k: DummySerper())
    # Monkeypatch CalendarCollector.collect_calendar_urls
    from src.data_collection.market_intelligence.calendar_collector import CalendarCollector
    monkeypatch.setattr(CalendarCollector, "collect_calendar_urls", fake_collect_calendar_urls)

    # Patch calendar extractor to produce two events per currency
    async def fake_extract_batch(self, results, currency: str):
        from src.data_collection.market_intelligence.models import EconomicEvent
        now = datetime.now(timezone.utc)
        return [
            EconomicEvent(
                when_utc=now,
                when_local=now,
                timezone="UTC",
                country="US",
                currency=currency,
                event=f"{currency} Rate hike expected",
                importance="high",
                source="test",
                source_url="https://example.com",
            ),
            EconomicEvent(
                when_utc=now,
                when_local=now,
                timezone="UTC",
                country="US",
                currency=currency,
                event=f"{currency} PMI",
                importance="medium",
                source="test",
                source_url="https://example.com",
            ),
        ]

    monkeypatch.setattr(cal_mod.CalendarExtractor, "extract_events_batch", fake_extract_batch)

    # Also patch NewsAggregator dependency to avoid LLM calls: use minimal news snapshot via monkeypatch classify

    async def fake_collect_pair_news(self, base: str, quote: str, hours_back: int = 24):
        class Obj:
            def __init__(self, title, snippet, url, source):
                self.title = title
                self.snippet = snippet
                self.url = url
                self.source = source

        return {
            "base": [Obj("A", "...", "https://www.reuters.com/a", "Reuters")],
            "quote": [Obj("B", "...", "https://www.bloomberg.com/b", "Bloomberg")],
        }

    from src.data_collection.market_intelligence.news_collector import NewsCollector
    monkeypatch.setattr(NewsCollector, "collect_pair_news", fake_collect_pair_news)

    async def fake_classify_batch(self, articles, currencies):
        from src.data_collection.market_intelligence.models import NewsClassification
        out = []
        for a in articles:
            out.append(
                NewsClassification(
                    article_id="id",
                    url=a["url"],
                    source=a["source"],
                    title=a["title"],
                    published_utc=None,
                    relevance={curr: 0.8 for curr in currencies},
                    sentiment={curr: 0.2 if curr == currencies[0] else -0.1 for curr in currencies},
                    quality_flags={"clickbait": False, "rumor_speculative": False, "non_econ": False},
                )
            )
        return out

    from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier
    monkeypatch.setattr(NewsClassifier, "classify_batch", fake_classify_batch)

    service = MarketIntelligenceService(serper_api_key=None, llm_manager=None)
    report = await service.get_pair_intelligence("USD", "EUR")
    assert report["pair"] == "USD/EUR"
    assert "news" in report and "calendar" in report
    assert report["calendar"]["total_high_impact_events_7d"] >= 1
    # policy bias should reflect a positive bias due to "Rate hike expected"
    assert report["policy_bias"] >= 0

