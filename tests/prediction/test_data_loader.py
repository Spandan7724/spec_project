import pytest

from src.prediction.data_loader import HistoricalDataLoader


@pytest.mark.asyncio
async def test_fetch_historical_data_basic():
    loader = HistoricalDataLoader()
    df = await loader.fetch_historical_data("USD", "EUR", days=90)

    assert df is not None, "Expected a DataFrame from yfinance"
    assert len(df) > 20, "Expected at least 20 rows of OHLC data"
    for col in ["Open", "High", "Low", "Close"]:
        assert col in df.columns


def test_yahoo_symbol_conversion():
    loader = HistoricalDataLoader()
    assert loader.get_yahoo_symbol("USD", "EUR") == "USDEUR=X"

