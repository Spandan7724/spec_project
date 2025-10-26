import json
import pytest

from datetime import timezone

from src.data_collection.market_intelligence.models import EconomicEvent
from src.data_collection.market_intelligence.serper_client import SerperSearchResult
from src.data_collection.market_intelligence.extractors import (
    CalendarExtractor,
    NewsClassifier,
    NarrativeGenerator,
)


class DummyResp:
    def __init__(self, content: str):
        self.content = content
        self.model = "dummy"
        self.usage = None
        self.tool_calls = None
        self.finish_reason = None
        self.provider = "test"


@pytest.mark.asyncio
async def test_calendar_extractor(monkeypatch):
    # Mock chat_with_model to return a JSON array for events
    from src.data_collection.market_intelligence.extractors import calendar_extractor as ce

    async def fake_chat(messages, model_name, llm_manager=None, tools=None):
        arr = [
            {"date": "2025-10-30", "time": "12:15", "event": "ECB Decision", "importance": "high"},
            {"date": "2025-11-01", "time": None, "event": "US Jobs Report", "importance": "high"},
        ]
        return DummyResp(json.dumps(arr))

    monkeypatch.setattr(ce, "chat_with_model", fake_chat)

    ext = CalendarExtractor(llm_manager=None)
    result = await ext.extract_events_from_snippet(
        SerperSearchResult(title="ECB Calendar", url="https://ecb.europa.eu", snippet="Upcoming ECB meeting", position=1),
        "EUR",
    )

    assert len(result) == 2
    assert isinstance(result[0], EconomicEvent)
    assert result[0].when_utc.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_news_classifier(monkeypatch):
    from src.data_collection.market_intelligence.extractors import news_classifier as nc

    async def fake_chat(messages, model_name, llm_manager=None, tools=None):
        data = {
            "relevance": {"USD": 0.8, "EUR": 0.4},
            "sentiment": {"USD": 0.3, "EUR": -0.1},
            "quality_flags": {"clickbait": False, "rumor_speculative": False, "non_econ": False},
        }
        return DummyResp(json.dumps(data))

    monkeypatch.setattr(nc, "chat_with_model", fake_chat)

    clf = NewsClassifier(llm_manager=None)
    nc_res = await clf.classify_article(
        title="Fed signals...",
        snippet="...",
        url="https://www.reuters.com/a",
        source="Reuters",
        currencies=["USD", "EUR"],
    )
    assert nc_res.relevance["USD"] == 0.8
    assert -1.0 <= nc_res.sentiment["EUR"] <= 1.0


@pytest.mark.asyncio
async def test_narrative_generator(monkeypatch):
    from src.data_collection.market_intelligence.extractors import narrative_generator as ng

    async def fake_chat(messages, model_name, llm_manager=None, tools=None):
        return DummyResp("USD is modestly favored due to stronger macro prints.")

    monkeypatch.setattr(ng, "chat_with_model", fake_chat)

    gen = NarrativeGenerator(llm_manager=None)
    snapshot = {
        "pair": "USD/EUR",
        "sent_base": 0.2,
        "sent_quote": -0.1,
        "pair_bias": 0.3,
        "confidence": "medium",
        "n_articles_used": 10,
        "top_evidence": [{"title": "Fed hikes"}, {"title": "ECB dovish"}],
    }
    narrative = await gen.generate_narrative(snapshot)
    assert isinstance(narrative, str) and len(narrative) > 0

