
import pytest


@pytest.mark.integration
def test_full_graph_prediction_node_updates_state(monkeypatch):
    # Ensure offline demo for upstream nodes (market data, intelligence)
    monkeypatch.setenv("OFFLINE_DEMO", "true")

    from src.agentic.graph import create_graph
    from src.agentic.state import initialize_state
    import src.agentic.nodes.prediction as node_mod
    from src.prediction.models import PredictionResponse, PredictionQuality, HorizonPrediction

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            pass

        async def predict(self, request):
            return PredictionResponse(
                status="success",
                confidence=0.65,
                processing_time_ms=40,
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions={
                    1: HorizonPrediction(1, 0.1, {"p10": -0.1, "p50": 0.1, "p90": 0.3}, 0.56)
                },
                latest_close=1.111,
                features_used=["sma_5"],
                quality=PredictionQuality(0.65, True, {}),
                model_id="unit_test_graph",
                warnings=[],
            )

    # Monkeypatch predictor in the node module
    monkeypatch.setattr(node_mod, "MLPredictor", FakePredictor)

    graph = create_graph()
    state = initialize_state(
        "Convert 1000 USD to EUR",
        base_currency="USD",
        quote_currency="EUR",
        timeframe="1_day",
    )
    result = graph.invoke(state)

    assert result.get("prediction_status") == "success"
    assert result.get("price_forecast") is not None
    assert "predictions" in result["price_forecast"]
    assert "1" in result["price_forecast"]["predictions"]

