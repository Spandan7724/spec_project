import pytest

from src.agentic.nodes.market_intelligence import market_intelligence_node
from src.agentic.state import initialize_state


@pytest.mark.asyncio
async def test_market_intelligence_node_fallback(monkeypatch):
    # Ensure offline demo mode so the node returns success fallback
    monkeypatch.setenv("OFFLINE_DEMO", "true")
    # Monkeypatch service to raise, to trigger fallback
    import src.agentic.nodes.market_intelligence as node_mod

    class Boom:
        def __init__(self, *args, **kwargs):
            pass

        async def get_pair_intelligence(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(node_mod, "MarketIntelligenceService", lambda *a, **k: Boom())

    state = initialize_state("Convert 1000 USD to EUR", base_currency="USD", quote_currency="EUR")
    out = await market_intelligence_node(state)
    assert out["intelligence_status"] == "success"
    assert out["intelligence_report"]["pair"] == "USD/EUR"
