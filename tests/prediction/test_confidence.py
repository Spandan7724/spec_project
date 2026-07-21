import pytest

from src.prediction.utils.confidence import (
    confidence_by_horizon,
    horizon_reliability_score,
    model_reliability_score,
)


def test_horizon_reliability_uses_direction_coverage_and_samples():
    metrics = {
        "directional_accuracy": 0.60,
        "quantile_coverage": 0.80,
        "n_samples": 500,
        "r2": 0.99,
    }

    assert horizon_reliability_score(metrics) == pytest.approx(0.60)


def test_horizon_reliability_penalizes_bad_coverage_and_small_sample():
    metrics = {
        "directional_accuracy": 0.60,
        "quantile_coverage": 0.40,
        "n_samples": 250,
    }

    # 0.60 direction × 0.50 coverage reliability × 0.50 sample reliability
    assert horizon_reliability_score(metrics) == pytest.approx(0.15)


def test_horizon_reliability_does_not_treat_r2_as_probability():
    assert horizon_reliability_score({"r2": 0.99, "n_samples": 500}) == 0.0


def test_model_reliability_can_target_the_selected_horizon():
    metrics = {
        "1d": {
            "directional_accuracy": 0.60,
            "quantile_coverage": 0.80,
            "n_samples": 500,
        },
        "7d": {
            "directional_accuracy": 0.50,
            "quantile_coverage": 0.40,
            "n_samples": 500,
        },
    }

    assert confidence_by_horizon(metrics, [1]) == {"1": pytest.approx(0.60)}
    assert model_reliability_score(metrics, [1]) == pytest.approx(0.60)
