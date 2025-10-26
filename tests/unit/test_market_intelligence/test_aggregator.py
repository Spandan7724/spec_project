import pytest

from src.data_collection.market_intelligence.aggregator import NewsAggregator
from src.data_collection.market_intelligence.news_collector import NewsCollector
from src.data_collection.market_intelligence.extractors.news_classifier import NewsClassifier


class FakeCollector(NewsCollector):
    def __init__(self):
        pass

    async def collect_pair_news(self, base: str, quote: str, hours_back: int = 24):
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


class FakeClassifier(NewsClassifier):
    def __init__(self):
        pass

    async def classify_batch(self, articles, currencies):
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


@pytest.mark.asyncio
async def test_news_aggregator_snapshot():
    aggr = NewsAggregator(FakeCollector(), FakeClassifier())
    snap = await aggr.get_pair_snapshot("USD", "EUR")
    assert snap.pair == "USD/EUR"
    assert snap.n_articles_used == 2
    assert snap.pair_bias > 0

