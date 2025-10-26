import pytest

from src.agentic.nodes.prediction import prediction_node
from src.agentic.state import initialize_state
from src.prediction.models import PredictionResponse, PredictionQuality, HorizonPrediction


@pytest.mark.asyncio
async def test_prediction_node_happy_path(monkeypatch):
    # Monkeypatch MLPredictor to avoid network and heavy lifting
    import src.agentic.nodes.prediction as node_mod

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            pass

        async def predict(self, request):
            return PredictionResponse(
                status="success",
                confidence=0.6,
                processing_time_ms=50,
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={
                    1: HorizonPrediction(1, 0.12, {"p10": -0.2, "p50": 0.12, "p90": 0.4}, 0.55)
                },
                latest_close=1.2345,
                features_used=["sma_5"],
                quality=PredictionQuality(0.6, True, {}),
                model_id="unit_test",
                warnings=[],
            )

    monkeypatch.setattr(node_mod, "MLPredictor", FakePredictor)

    state = initialize_state(
        "Should I convert now?",
        base_currency="USD",
        quote_currency="EUR",
        timeframe="1_day",
    )
    out = await prediction_node(state)
    assert out["prediction_status"] == "success"
    assert out["price_forecast"]["predictions"]["1"]["mean_change_pct"] == 0.12
