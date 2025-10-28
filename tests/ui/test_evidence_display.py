from __future__ import annotations

from src.ui.tui.display import create_evidence_panel


def test_evidence_panel_renders():
    evidence = {
        "market": {
            "providers": ["yfinance"],
            "quality_notes": ["fresh"],
            "indicators": {"rsi_14": 50.0},
            "regime": {"trend_direction": "flat", "bias": "neutral"},
        },
        "news": [{"source": "Reuters", "title": "Headline", "url": "https://example.com"}],
        "calendar": [{"currency": "USD", "event": "CPI", "importance": "high", "source_url": "https://example.com"}],
        "model": {"top_features": {"rsi_14": 1.0}},
        "prediction": {"horizon_key": "1", "mean_change_pct": 0.1, "quantiles": {"0.05": -0.2, "0.95": 0.4}},
        "intelligence": {"pair_bias": 0.1, "news_confidence": "medium", "policy_bias": 0.0},
    }

    panel = create_evidence_panel(evidence)
    assert hasattr(panel, "__rich_console__")
