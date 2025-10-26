from src.agentic.nodes.decision import decision_node


def test_decision_node_updates_state_success():
    state = {
        "correlation_id": "test-123",
        "currency_pair": "USD/EUR",
        "base_currency": "USD",
        "quote_currency": "EUR",
        "amount": 5000,
        "risk_tolerance": "moderate",
        "urgency": "normal",
        "timeframe": "1_week",
        "warnings": [],
        "market_snapshot": {
            "quality": {"fresh": True, "dispersion_bps": 10.0},
            "indicators": {"atr_14": 0.005, "rsi_14": 55, "macd": 0.001, "macd_signal": 0.0},
            "regime": {"trend_direction": "up", "bias": "bullish"},
        },
        "intelligence_report": {"calendar": {"events_extracted": []}},
        "price_forecast": {
            "status": "success",
            "confidence": 0.8,
            "latest_close": 1.0,
            "predictions": {"7": {"mean_change_pct": 0.3, "quantiles": {"p10": -0.2, "p50": 0.3, "p90": 0.5}}},
        },
    }
    out = decision_node(state)
    assert out["decision_status"] == "success"
    rec = out["recommendation"]
    assert rec["action"] in {"wait", "staged_conversion", "convert_now"}
    assert 0.0 <= rec["confidence"] <= 1.0
    assert isinstance(rec["rationale"], list) and len(rec["rationale"]) > 0

