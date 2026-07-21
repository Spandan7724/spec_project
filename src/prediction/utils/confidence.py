"""Validation-based reliability scores for prediction models.

The returned value is an operational reliability score, not a probability that
an individual forecast will be correct. It intentionally avoids treating R² as
a probability.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional


NOMINAL_INTERVAL_COVERAGE = 0.80


def _finite_unit_interval(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return max(0.0, min(1.0, float(value)))


def horizon_reliability_score(
    metrics: Dict[str, Any],
    *,
    min_samples_required: int = 500,
    nominal_interval_coverage: float = NOMINAL_INTERVAL_COVERAGE,
) -> float:
    """Score one horizon from direction accuracy and interval reliability.

    Directional accuracy is the empirical base rate. It is discounted when the
    p10-p90 interval misses its nominal 80% coverage or the evaluation sample is
    too small. Missing direction or coverage evidence produces a zero score.
    """
    direction_accuracy = _finite_unit_interval(metrics.get("directional_accuracy"))
    interval_coverage = _finite_unit_interval(metrics.get("quantile_coverage"))
    if direction_accuracy is None or interval_coverage is None:
        return 0.0

    nominal = max(1e-9, min(1.0, float(nominal_interval_coverage)))
    coverage_error = abs(interval_coverage - nominal)
    coverage_reliability = max(0.0, 1.0 - min(1.0, coverage_error / nominal))

    sample_count = metrics.get("n_samples", 0)
    if not isinstance(sample_count, (int, float)) or not math.isfinite(float(sample_count)):
        sample_count = 0
    required = max(1, int(min_samples_required))
    sample_reliability = max(0.0, min(1.0, float(sample_count) / required))

    return max(
        0.0,
        min(1.0, direction_accuracy * coverage_reliability * sample_reliability),
    )


def confidence_by_horizon(
    validation_metrics: Dict[str, Dict[str, Any]],
    horizons: Optional[Iterable[int]] = None,
    *,
    min_samples_required: int = 500,
) -> Dict[str, float]:
    """Return reliability scores keyed by normalized numeric horizon."""
    requested = {int(horizon) for horizon in horizons} if horizons is not None else None
    scores: Dict[str, float] = {}

    for raw_key, metrics in (validation_metrics or {}).items():
        try:
            horizon = int(str(raw_key).lower().removesuffix("d").removesuffix("h"))
        except (TypeError, ValueError):
            continue
        if requested is not None and horizon not in requested:
            continue
        scores[str(horizon)] = horizon_reliability_score(
            metrics or {},
            min_samples_required=min_samples_required,
        )

    return scores


def model_reliability_score(
    validation_metrics: Dict[str, Dict[str, Any]],
    horizons: Optional[Iterable[int]] = None,
    *,
    min_samples_required: int = 500,
) -> float:
    """Return the mean reliability for the requested model horizons."""
    scores = confidence_by_horizon(
        validation_metrics,
        horizons,
        min_samples_required=min_samples_required,
    )
    return sum(scores.values()) / len(scores) if scores else 0.0
