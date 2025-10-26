# Phase 2 Implementation Summary

Concise overview of Phase 2 (Layer 2): price prediction agent with ML backends, registry, routing, fallback, and graph wiring.

## Phases Implemented
- 2.1 Data Pipeline & Feature Engineering
- 2.2 Model Registry & Storage
- 2.3 Model Backends (LightGBM + LSTM) + Explainability
- 2.4 Fallback Heuristics
- 2.5 Prediction Agent & Graph Wiring

## Files Created (by phase)

### 2.1 Data Pipeline
- `src/prediction/models.py` — Dataclasses for requests/responses/metadata.
- `src/prediction/config.py` — Prediction config loader (root `prediction:` + path normalization).
- `src/prediction/data_loader.py` — Async yfinance loader (1d + intraday intervals).
- `src/prediction/feature_builder.py` — Technical indicators + target generation.
- `tests/prediction/test_data_loader.py` — Loader smoke tests.
- `tests/prediction/test_feature_builder.py` — Features/RSI/targets tests.

### 2.2 Registry
- `src/prediction/registry.py` — JSON registry + pickle artifacts; latest-by-pair queries.
- `tests/prediction/test_registry.py` — Register/load/list/delete/persist tests.

### 2.3 Backends + Explainability
- `src/prediction/backends/base.py` — Unified backend interface.
- `src/prediction/backends/lightgbm_backend.py` — GBM mean/quantiles/direction + save/load.
- `src/prediction/backends/lstm_backend.py` — LSTM intraday w/ MC‑dropout quantiles + direction.
- `src/prediction/explainer.py` — SHAP explainer helpers for GBM.
- `src/prediction/utils/calibration.py` — Simple quality gates checker.
- `src/prediction/predictor.py` — Hybrid routing, caching, registry load, fallback hook.
- `tests/prediction/backends/test_lightgbm.py` — Train/predict/importance/save‑load.
- `tests/prediction/backends/test_lstm.py` — Train/predict/quantiles/save‑load.
- `tests/prediction/test_predictor_routing.py` — Horizon split sanity.
- `tests/prediction/test_predictor_registry_integration.py` — GBM load from registry.
- `src/prediction/training.py` — Helpers to train + register GBM/LSTM.
- `tests/prediction/test_training_helper.py` — Helper e2e test with mocks.

### 2.4 Fallback
- `src/prediction/utils/fallback.py` — MA crossover + RSI heuristics (conservative), quantiles by vol.
- `tests/prediction/test_fallback.py` — Basic/insufficient data tests.

### 2.5 Prediction Agent
- `src/agentic/nodes/prediction.py` — Async node; maps timeframe→horizons, calls predictor.
- `src/agentic/graph.py` — Wires real prediction node into LangGraph.
- `tests/unit/test_agentic/test_nodes/test_prediction.py` — Node unit test (mocked predictor).
- `tests/integration/test_agentic/test_prediction_full_graph.py` — Full graph integration (mocked predictor).
- `scripts_test/test_phase_2.py` — Phase 2 E2E runner; prints market/intel/prediction outputs.

### CLI & Paths
- `src/cli/main.py` — Training commands (interactive) + mounts sub‑CLI.
- `src/cli/train.py` — Dedicated training sub‑CLI (`train model`, `train lstm`).
- `pyproject.toml` — Adds `currency-assistant-train` entrypoint.
- `src/utils/paths.py` — Project root discovery + robust path resolution.
- `src/config.py` — Config path resolution irrespective of CWD.

## Configuration & Env (Phase 2 runtime knobs)
- `config.yaml` → `prediction:`
  - `predictor_backend`: "hybrid" | "lightgbm" | "lstm" (default routing).
  - `prediction_horizons`, `features_mode`, `technical_indicators`.
  - `model_registry_path`, `model_storage_dir` (normalized to absolute).
  - Quality gates + fallback strength.
- Env overrides (optional)
  - `CURRENCY_ASSISTANT_ROOT` — force repo root.
  - `CURRENCY_ASSISTANT_CONFIG` — point to specific config file.
  - `OFFLINE_DEMO` — upstream demo mode for quick runs.

## Phase Outcome
- Layer 2 complete:
  - Trainable, persisted models (GBM daily/weekly; LSTM intraday).
  - Hybrid routing with per‑request/backend override.
  - Uncertainty + direction per horizon (GBM; LSTM via MC‑dropout).
  - Fallback heuristics as last resort.
  - Prediction node integrated into the LangGraph workflow; Phase 2 runner demonstrates end‑to‑end outputs.

