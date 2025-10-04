# System Stabilization Plan

## Context
With the provider cost pipeline deferred, the immediate goal is to harden the existing agentic workflow so it remains reliable even with missing credentials, offline ML models, or absent tests. This document captures the priorities, rationale, and concrete tasks required to regain confidence in the core system before expanding functionality again.

## Objectives
1. Guarantee that the LangGraph workflow runs deterministically with clear logging, dependency management, and meaningful fallbacks.
2. Restore a minimal but effective automated test harness that validates critical behaviour without live network calls.
3. Ensure external integrations (API providers, GPUs, environment configuration) fail gracefully and provide actionable diagnostics.
4. Improve or temporarily replace the ML prediction layer so downstream agents receive sane signals.
5. Align documentation and configuration with the codebase so contributors can bootstrap the project quickly.

## Workstreams & Tasks

### 1. Agent Workflow Hardening
- [ ] Refactor agent nodes to accept injected dependencies instead of instantiating services internally (`MarketAnalysisAgent`, `EconomicAnalysisAgent`, `RiskAssessmentAgent`, `DecisionCoordinatorAgent`).
- [ ] Introduce consistent correlation IDs and structured logging sourced from `AgentMeta` for every node invocation.
- [ ] Audit error handling so each agent appends warnings to state without raising (e.g., rate collector failures, indicator gaps, ML unavailability) and decision logic surfaces them predictably.

### 2. Core Test Suite Restoration
- [ ] Recreate a lightweight `tests/test_agentic_core.py` (or similar) that mocks rate/economic/LLM dependencies and validates happy-path recommendations.
- [ ] Add regression tests for common failure scenarios (no rates, economic calendar empty, ML predictor unavailable) to prove fallbacks work.
- [ ] Provide shared fixtures/mocks for key services so tests run offline and quickly.

### 3. External Dependency Resilience
- [ ] Remove duplicate `load_dotenv()` calls and centralize environment loading to avoid unintended overrides.
- [ ] Implement configuration guards that log missing API keys without crashing provider initialization (`MultiProviderRateCollector`, calendar collectors, LLM manager).
- [ ] Allow the ML predictor to operate in CPU-only mode or skip training with a clear error, replacing the hard CUDA requirement during inference/training.
- [ ] Document required environment variables and sample `.env` values for local development.

### 4. ML Layer Remediation
- [ ] Investigate current LSTM artifacts (negative R², low directional accuracy) and capture root causes (data leakage, insufficient samples, overfitting).
- [ ] Decide on short-term strategy: retrain with cleaned data or provide a statistical fallback (e.g., moving-average drift) that passes sanity checks.
- [ ] Ensure the market agent clearly differentiates between real ML output and fallback heuristics for downstream reasoning.

### 5. Documentation & Configuration Alignment
- [ ] Update `docs/project_todo.md` and related planning docs to reflect the implemented LangGraph agents and the new stabilization plan.
- [ ] Author a quick-start section that explains how to run the CLI with mocks/tests, including environment preparation.
- [ ] Capture logging/observability expectations (log levels, correlation IDs, sample log snippet) to guide future instrumentation.

## Suggested Sequence
1. Start with Workstream 1 (agents) to stabilise behaviour and capture logging.
2. Immediately follow with Workstream 2 (tests) so subsequent changes remain covered.
3. Tackle Workstream 3 to make the system resilient when credentials or GPUs are unavailable.
4. Address Workstream 4 to raise the quality of predictive inputs consumed by the market agent.
5. Close with Workstream 5 to keep plans and onboarding material synchronized with the code.

## Progress Tracking
Update the checkboxes above as work completes; aim to keep the document in sync with pull requests or commits that move tasks forward.
