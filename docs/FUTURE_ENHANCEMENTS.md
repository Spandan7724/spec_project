
# Future Enhancements

## Hybrid Exact‑Day Forecasts

Goal: When a user provides a flexible timeframe (e.g., “in 10 days”), use an exact‑day prediction horizon if a trained model for that horizon is available; otherwise, fall back to the nearest calibrated bucket (e.g., 1/7/30 days).

### Rationale
- Exact horizons can better match user intent and scenario planning (24h/48h, 10d, 14d).
- Maintaining a fallback to calibrated buckets preserves accuracy, stability, and cache efficiency when exact horizons aren’t trained.

### Behavior
- If `timeframe_days` equals a horizon with a trained/registered model, include that horizon in the prediction request and prefer it in decision making.
- Otherwise, bucket using the current strategy (e.g., <=1→1d, <30→7d, else 30d).
- Expose the used horizon via `recommendation.meta.used_prediction_horizon_key` (already present).

### Prerequisites
- Trained models for specific additional horizons (e.g., 2d, 10d, 14d).
- Registry entries keyed by horizon (extend metadata with `horizon_days`, calibration metrics).
- Calibration and quality gates per horizon (direction probability, quantiles).

### Configuration
- Add a flag to enable hybrid routing:
  - `prediction.hybrid_exact_horizon: true|false` (default: false)
- Optional whitelist to limit allowed exact horizons:
  - `prediction.allowed_exact_horizons: [2, 10, 14]`

### Design Changes (minimal)
- Horizon selection logic (prediction layer):
  - Build the horizon set as: `configured_horizons ∪ {timeframe_days ∈ allowed_exact_horizons}`.
  - If the exact horizon lacks a trained model, skip silently and rely on buckets.
- Decision layer: prefer exact horizon if present; otherwise, nearest bucket (current behavior already surfaces the used key in `meta`).
- Caching: include horizon days in cache keys; expect slightly lower hit rate when exact horizons are in play.

### Registry & Training
- Registry (data/models/prediction_registry.json):
  - Add `horizon_days` to model metadata.
  - Expose a lookup by (pair, horizon_days, version/status).
- Training scripts:
  - Add CLI flag `--horizons` to include additional exact horizons.
  - Persist calibration metrics per horizon (MAPE, directional accuracy, calibration error).

### Explainability
- Ensure SHAP/feature importance support for exact horizons is identical to bucketed horizons.
- Surface top_features in evidence as we do now.

### Observability
- Metrics:
  - `prediction.hybrid.used_exact_horizon` (bool)
  - `prediction.hybrid.exact_horizon_days` (int)
  - Latency and cache hit rate by horizon
- Warnings:
  - If the exact horizon requested is not trained/available, optionally attach a low‑priority note to `recommendation.warnings`.

### Risks & Mitigations
- Risk: Fragmented cache, higher compute → Mitigate with `allowed_exact_horizons` and a config flag.
- Risk: Poor calibration at new horizons → Enforce horizon‑specific quality gates (minimum accuracy/confidence thresholds to allow usage).
- Risk: Inconsistent availability across pairs → Fall back to buckets automatically and record metrics.

### Implementation Steps
1. Config: add `hybrid_exact_horizon` and `allowed_exact_horizons`.
2. Registry: add `horizon_days` metadata; provide a `get_model_for_horizon(pair, horizon_days)` helper.
3. Predictor: union configured horizons with `timeframe_days` if in allowed set; skip if model missing.
4. Decision: prefer exact horizon if present; else bucket (no structural change required).
5. Evidence/meta: continue to publish `meta.used_prediction_horizon_key` and `evidence.prediction.horizon_key` for transparency.
6. Tests:
   - Unit: horizon selection with/without exact model present.
   - Integration: end‑to‑end recommendation preferring exact horizon; fallback works when omitted.
   - Performance: compare latency and cache hit deltas when hybrid is enabled.

### UI/UX Notes
- TUI/Web can show: “Forecast horizon used: X days (hybrid)” when exact horizon is used; otherwise “nearest calibrated horizon: Y days”.
- No changes required to user input; this is purely internal routing.
