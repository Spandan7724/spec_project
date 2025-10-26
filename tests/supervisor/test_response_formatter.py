from src.supervisor.response_formatter import ResponseFormatter


def sample_recommendation(action: str = "convert_now"):
    return {
        "status": "success",
        "action": action,
        "confidence": 0.76,
        "timeline": "Immediate execution recommended",
        "rationale": [
            "Prediction indicates favorable short-term movement",
            "Market spread within normal range",
        ],
        "warnings": ["High-impact event tomorrow"],
        "expected_outcome": {
            "expected_rate": 0.9231,
            "range_low": 0.9180,
            "range_high": 0.9280,
            "expected_improvement_bps": 12.5,
        },
        "risk_summary": {
            "risk_level": "moderate",
            "realized_vol_30d": 6.71,
            "var_95": 1.85,
            "event_risk": "elevated",
            "event_details": "ECB minutes release",
        },
        "cost_estimate": {
            "spread_bps": 5.0,
            "fee_bps": 0.0,
            "total_bps": 5.0,
        },
    }


def test_format_convert_now_action():
    f = ResponseFormatter()
    txt = f.format_recommendation(sample_recommendation("convert_now"))
    assert "RECOMMENDATION FOR CURRENCY CONVERSION" in txt
    assert "ACTION: CONVERT NOW" in txt
    assert "CONFIDENCE: 0.76 (High)" in txt
    assert "TIMELINE:" in txt
    assert "RATIONALE:" in txt
    assert "WARNINGS:" in txt


def test_format_staged_action_includes_tranches():
    f = ResponseFormatter()
    rec = sample_recommendation("staged_conversion")
    rec["staged_plan"] = {
        "num_tranches": 2,
        "spacing_days": 2,
        "tranches": [
            {"tranche_number": 1, "percentage": 60, "execute_day": 0},
            {"tranche_number": 2, "percentage": 40, "execute_day": 2},
        ],
    }
    txt = f.format_recommendation(rec)
    assert "STAGED CONVERSION PLAN:" in txt
    assert "Tranche 1: 60%" in txt
    assert "Tranche 2: 40%" in txt


def test_format_wait_action():
    f = ResponseFormatter()
    txt = f.format_recommendation(sample_recommendation("wait"))
    assert "ACTION: WAIT" in txt


def test_confidence_levels():
    f = ResponseFormatter()
    rec = sample_recommendation()
    rec["confidence"] = 0.85
    assert "High" in f.format_recommendation(rec)
    rec["confidence"] = 0.55
    assert "Moderate" in f.format_recommendation(rec)
    rec["confidence"] = 0.35
    assert "Low" in f.format_recommendation(rec)


def test_format_error_response():
    f = ResponseFormatter()
    err = {"status": "error", "error": "Decision engine failed", "warnings": ["timeout"]}
    txt = f.format_recommendation(err)
    assert "ERROR GENERATING RECOMMENDATION" in txt
    assert "Decision engine failed" in txt
    assert "timeout" in txt

