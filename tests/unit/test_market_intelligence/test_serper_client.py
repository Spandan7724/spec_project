import httpx
import pytest

from src.data_collection.market_intelligence.serper_client import (
    SerperClient,
)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class DummyClient:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        return DummyResponse(self.payload, status_code=200)


@pytest.mark.asyncio
async def test_search_news_filters_domains(monkeypatch):
    payload = {
        "news": [
            {"title": "A", "link": "https://www.reuters.com/a", "source": "Reuters", "snippet": "...", "date": "2025-10-24"},
            {"title": "B", "link": "https://unknownsite.xyz/b", "source": "Unknown", "snippet": "...", "date": "2025-10-24"},
        ]
    }

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=None: DummyClient(payload))

    client = SerperClient(api_key="dummy")
    res = await client.search_news("USD economy", time_range="qdr:d", num_results=5)
    assert len(res) == 1
    assert res[0].url.startswith("https://www.reuters.com")


@pytest.mark.asyncio
async def test_search_news_no_whitelist_when_opt_out(monkeypatch):
    payload = {
        "news": [
            {"title": "A", "link": "https://www.reuters.com/a", "source": "Reuters", "snippet": "...", "date": "2025-10-24"},
            {"title": "B", "link": "https://unknownsite.xyz/b", "source": "Unknown", "snippet": "...", "date": "2025-10-24"},
        ]
    }
    monkeypatch.setenv("SERPER_ENABLE_WHITELIST", "false")
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=None: DummyClient(payload))
    client = SerperClient(api_key="dummy")
    res = await client.search_news("USD economy")
    # Should include both articles when whitelist disabled
    assert len(res) == 2


@pytest.mark.asyncio
async def test_search_general_returns_results(monkeypatch):
    payload = {
        "organic": [
            {"title": "Cal", "link": "https://calendar.example/events", "snippet": "..."},
            {"title": "Other", "link": "https://example.com/other", "snippet": "..."},
        ]
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=None: DummyClient(payload))

    client = SerperClient(api_key="dummy")
    res = await client.search("USD calendar", num_results=3)
    assert len(res) == 2
    assert res[0].url.startswith("https://calendar.example/")


@pytest.mark.asyncio
async def test_health_check(monkeypatch):
    payload = {"organic": []}
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=None: DummyClient(payload))

    client = SerperClient(api_key="dummy")
    ok = await client.health_check()
    assert ok is True
