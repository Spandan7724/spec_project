import pytest
from datetime import timezone

from src.data_collection.providers.yfinance_client import YFinanceClient
from src.utils.errors import ValidationError, DataProviderError


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self._fast_info = {
            "bid": 0.8606,
            "ask": 0.8611,
            "last_price": 0.86085,
        }
        self._info = {
            "bid": 0.8606,
            "ask": 0.8611,
            "regularMarketPrice": 0.86085,
        }

    @property
    def fast_info(self):
        return self._fast_info

    @property
    def info(self):
        return self._info


@pytest.mark.asyncio
async def test_yfinance_success(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: FakeTicker(symbol))

    client = YFinanceClient()
    rate = await client.get_rate("USD", "EUR")

    assert rate.source == "yfinance"
    assert rate.rate > 0
    assert rate.bid is not None and rate.ask is not None
    assert rate.timestamp.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_yfinance_invalid_pair():
    client = YFinanceClient()
    with pytest.raises(ValidationError):
        await client.get_rate("USD", "USD")


class FakeTickerNoBook(FakeTicker):
    @property
    def fast_info(self):
        return {"last_price": 0.75}

    @property
    def info(self):
        return {"regularMarketPrice": 0.75}


@pytest.mark.asyncio
async def test_yfinance_fallback_to_last_price(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: FakeTickerNoBook(symbol))

    client = YFinanceClient()
    rate = await client.get_rate("GBP", "USD")
    assert rate.rate == pytest.approx(0.75, rel=1e-6)
    assert rate.bid is None and rate.ask is None


class FakeTickerNoData(FakeTicker):
    @property
    def fast_info(self):
        return {}

    @property
    def info(self):
        return {}


@pytest.mark.asyncio
async def test_yfinance_no_data(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: FakeTickerNoData(symbol))
    client = YFinanceClient()
    with pytest.raises(DataProviderError):
        await client.get_rate("EUR", "JPY")

