from typing import Dict


def check_quality_metrics(
    validation_metrics: Dict[str, Dict[str, float]],
    min_samples_required: int = 500,
    min_validation_coverage: float = 0.85,
    min_directional_accuracy: float = 0.52,
) -> Dict:
    """Evaluate simple quality gates over validation metrics.

    Returns summary with per-horizon checks and overall pass flag.
    """
    checks = {}
    all_passed = True
    for horizon, m in validation_metrics.items():
        samples_ok = m.get("n_samples", 0) >= min_samples_required
        coverage_ok = m.get("quantile_coverage", 0.0) >= min_validation_coverage
        diracc_ok = m.get("directional_accuracy", 0.0) >= min_directional_accuracy
        passed = samples_ok and coverage_ok and diracc_ok
        checks[horizon] = {
            "samples_ok": samples_ok,
            "coverage_ok": coverage_ok,
            "diracc_ok": diracc_ok,
            "passed": passed,
        }
        all_passed = all_passed and passed

    return {"passed": all_passed, "checks": checks}

