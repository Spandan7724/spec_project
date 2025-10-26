import pytest

from src.agentic.nodes.market_data import market_data_node
from src.agentic.state import initialize_state


@pytest.mark.asyncio
async def test_market_data_node_offline_demo(monkeypatch):
    # Force offline demo to avoid network
    monkeypatch.setenv("OFFLINE_DEMO", "true")

    state = initialize_state("Convert 1000 USD to EUR", base_currency="USD", quote_currency="EUR")
    out = await market_data_node(state)

    assert out["market_data_status"] == "success"
    assert out["market_snapshot"]["currency_pair"] == "USD/EUR"

