from datetime import datetime, timedelta, timezone

from src.data_collection.market_data.aggregator import aggregate_rates
from src.data_collection.providers.base import ProviderRate


def make_rate(source: str, rate: float, bid=None, ask=None, age_sec: int = 0):
    return ProviderRate(
        source=source,
        rate=rate,
        bid=bid,
        ask=ask,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age_sec),
        notes=[],
    )


def test_aggregate_basic_median():
    rates = [
        make_rate("a", 1.0000),
        make_rate("b", 1.0010),
        make_rate("c", 0.9990),
    ]

    mid, bid, ask, spread, ts, qm, kept = aggregate_rates(rates, cache_ttl_seconds=5)
    assert round(mid, 6) == 1.0000  # median of [1.0, 1.001, 0.999]
    assert qm.sources_success == 3 and qm.sources_total == 3
    assert qm.fresh is True
    assert len(kept) == 3


def test_aggregate_with_bid_ask_and_spread():
    rates = [
        make_rate("a", rate=1.0, bid=0.9995, ask=1.0005),
        make_rate("b", rate=1.0, bid=0.9990, ask=1.0010),
    ]
    mid, bid, ask, spread, ts, qm, kept = aggregate_rates(rates)
    assert bid == 0.9995  # highest bid
    assert ask == 1.0005  # lowest ask
    assert round(spread, 6) == round(1.0005 - 0.9995, 6)


def test_outlier_removal():
    rates = [
        make_rate("a", 1.0),
        make_rate("b", 1.0010),
        make_rate("c", 1.8),  # extreme outlier
    ]
    mid, _, _, _, _, qm, kept = aggregate_rates(rates, outlier_threshold_bps=100.0)
    assert len(kept) == 2
    assert any("outlier" in n.lower() for n in qm.notes)


def test_staleness_flag():
    rates = [
        make_rate("a", 1.0, age_sec=10),
    ]
    _, _, _, _, _, qm, _ = aggregate_rates(rates, cache_ttl_seconds=5)
    assert qm.fresh is False
    assert any("not fresh" in n.lower() for n in qm.notes)

