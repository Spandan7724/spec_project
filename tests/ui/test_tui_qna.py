from __future__ import annotations

from src.ui.tui.app import CurrencyAssistantTUI
from src.supervisor.models import ExtractedParameters


def make_reco_with_evidence():
    return {
        "action": "convert_now",
        "confidence": 0.6,
        "timeline": "Immediate",
        "evidence": {
            "market": {
                "mid_rate": 1.085,
                "bid": 1.0848,
                "ask": 1.0852,
                "rate_timestamp": "2025-10-27T12:00:00Z",
                "indicators": {"rsi_14": 55.2, "macd": 0.0005, "macd_signal": 0.0003},
                "regime": {"trend_direction": "up", "bias": "bullish"},
            },
            "prediction": {"horizon_key": "1", "mean_change_pct": 0.2, "quantiles": {"p10": -0.3, "p90": 0.7}},
            "predictions_all": {
                "1": {"mean_change_pct": 0.2, "quantiles": {"p10": -0.3, "p50": 0.1, "p90": 0.7}},
                "7": {"mean_change_pct": 0.8, "quantiles": {"p10": 0.1, "p50": 0.6, "p90": 1.2}},
            },
            "model": {"top_features": {"rsi_14": 1.2, "sma_20": 0.7}},
            "calendar": [
                {"currency": "USD", "event": "NFP", "importance": "high", "source_url": "https://...", "proximity_minutes": 1440}
            ],
        },
    }


def test_qna_rate_and_forecast():
    tui = CurrencyAssistantTUI()
    reco = make_reco_with_evidence()
    params = ExtractedParameters(currency_pair="USD/EUR")

    rate_ans = tui._answer_question("what's the current rate?", reco, params)
    assert "Current rate" in rate_ans

    fc_ans = tui._answer_question("what's the forecast?", reco, params)
    assert "Forecast mean change" in fc_ans or "No forecast" not in fc_ans
