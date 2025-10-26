from datetime import timezone

import httpx
import pytest

from src.data_collection.providers.exchange_rate_host import ExchangeRateHostClient
from src.utils.errors import ValidationError, DataProviderError


class DummyResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class DummyClient:
    def __init__(self, timeout=None, data=None, should_raise: bool = False):
        self._data = data or {"quotes": {"USDEUR": 0.85}, "timestamp": 1729800000}
        self._should_raise = should_raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        if self._should_raise:
            raise httpx.HTTPError("network error")
        return DummyResponse(self._data, status_code=200)


@pytest.mark.asyncio
async def test_exchange_rate_host_success(monkeypatch):
    # Patch AsyncClient to return a controlled response
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda timeout=None: DummyClient(timeout=timeout)
    )

    client = ExchangeRateHostClient()
    rate = await client.get_rate("USD", "EUR")

    assert rate.source == "exchange_rate_host"
    assert rate.rate > 0
    assert rate.bid is None and rate.ask is None
    assert rate.timestamp.tzinfo is not None and rate.timestamp.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_exchange_rate_host_invalid_currency():
    client = ExchangeRateHostClient()
    with pytest.raises(ValidationError):
        await client.get_rate("usd", "EUR")  # lowercase should fail


@pytest.mark.asyncio
async def test_exchange_rate_host_http_error(monkeypatch):
    # Force HTTP error
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda timeout=None: DummyClient(timeout=timeout, should_raise=True)
    )
    client = ExchangeRateHostClient()
    with pytest.raises(DataProviderError):
        await client.get_rate("USD", "EUR")

