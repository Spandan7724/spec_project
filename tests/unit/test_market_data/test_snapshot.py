from datetime import datetime, timezone

import pandas as pd

from src.data_collection.market_data.snapshot import build_snapshot
from src.data_collection.providers.base import BaseProvider, ProviderRate


class FakeProvider(BaseProvider):
    NAME = "fake"

    def __init__(self, val: float):
        self.val = val

    async def get_rate(self, base: str, quote: str) -> ProviderRate:
        return ProviderRate(
            source=self.NAME,
            rate=self.val,
            bid=None,
            ask=None,
            timestamp=datetime.now(timezone.utc),
            notes=[],
        )

    async def health_check(self) -> bool:
        return True


def make_hist_df():
    idx = pd.date_range(end=datetime.now(), periods=60, freq='D')
    close = pd.Series(1.0 + (0.001 * pd.RangeIndex(60)), index=idx)
    df = pd.DataFrame({
        'Open': close * 0.999,
        'High': close * 1.001,
        'Low': close * 0.999,
        'Close': close,
    }, index=idx)
    return df


import pytest


@pytest.mark.asyncio
async def test_build_snapshot_basic():
    providers = [FakeProvider(1.0), FakeProvider(1.001)]
    hist = make_hist_df()

    snap = await build_snapshot("USD", "EUR", providers, historical_df=hist)

    assert snap.currency_pair == "USD/EUR"
    assert snap.mid_rate > 0
    assert snap.rate_timestamp.tzinfo is not None
    assert snap.quality.sources_success == 2
    assert snap.indicators is not None
    assert snap.regime is not None
