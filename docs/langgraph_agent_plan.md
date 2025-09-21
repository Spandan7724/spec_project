# LangGraph Agentic Implementation Plan

## Purpose
Establish a production-ready LangGraph workflow that orchestrates the existing data, ML, and LLM subsystems into an agentic currency-conversion advisor. This document captures the design decisions, implementation roadmap, and progress tracker for the agent build-out.

## Current Building Blocks
- **Data Collection**: Real-time FX collector, historical analytics, economic calendar (`src/data_collection/*`).
- **LLM Infrastructure**: Multi-provider manager with health checks/failover (`src/llm/manager.py`).
- **ML Scaffolding**: LSTM model, feature engineering, prediction cache (not yet wired into a serving loop) (`src/ml/*`).
- **Scraping & Tools**: Decision engine, cache manager, generic scraper (`src/tools/*`).
- **Documentation**: Tool communication contract (`docs/tool_communication_protocol.md`), system architecture references in existing docs.

These components will be wrapped by LangGraph agents that coordinate insights and produce user-facing recommendations.

## Target Outcomes
1. **Functional Graph**: LangGraph workflow that ingests a user conversion request, routes it through specialized agents, and emits an actionable recommendation.
2. **Agent Specialization**: Each agent focuses on a domain (market, economic, risk, provider costs) and returns structured outputs.
3. **State Management**: Shared graph state tracks inputs, intermediate signals, and metadata for transparency/replay.
4. **Resilience & Fallbacks**: Graceful degradation when ML models or external providers are unavailable.
5. **Testability**: Deterministic unit/integration tests for agent logic and graph execution paths.

## Agent Definitions
### 1. Market Analysis Agent
- **Inputs**: Currency pair, amount, historical data, technical indicators, ML forecasts (if available).
- **Responsibilities**:
  - Pull fresh rates via `MultiProviderRateCollector`.
  - Fetch 30–90 day historical dataset and indicators.
  - Query ML predictor for horizons `[1,7,30]`; fallback to rule-based technical signals when ML unavailable.
  - Produce market regime classification (trending/ranging), directional bias, and confidence score.
- **Outputs**: Structured dict (`market_analysis`) with key metrics, raw references, and error flags.

### 2. Economic Analysis Agent
- **Inputs**: Currency pair, timeline horizons, economic calendar cache.
- **Responsibilities**:
  - Use `EconomicCalendar` collectors to enumerate upcoming events and recent releases.
  - Score event impact vs currency pair relevance.
  - Provide qualitative bias (hawkish/dovish, risk-on/off) leveraging LLM summarization when necessary.
- **Outputs**: `economic_analysis` block with upcoming events, aggregated bias, risk timeline markers.

### 3. Risk Assessment Agent
- **Inputs**: Market signals, volatility metrics, user risk tolerance.
- **Responsibilities**:
  - Compute risk metrics using volatility + historical drawdown data.
  - Evaluate scenario ranges (best/base/worst) for requested time horizon.
  - Determine recommended conversion tranche sizing or hedging notes.
- **Outputs**: `risk_assessment` block with VaR estimates, volatility regime, suggested caution level.

### 4. Provider Cost Agent (Future Integration)
- **Status**: Not implemented; ensure the graph operates without this agent for now while leaving hooks ready for future insertion.
- **Integration Strategy**: Reserve state slots (`provider_costs`) and decision-coordinator inputs so a cost analyzer can plug in later without refactoring core flows.
- **Inputs** (future): Amount, currencies, baseline mid-market rate once provider modules exist.
- **Responsibilities** (future):
  - Invoke provider cost analyzer to compare effective rates and fees.
  - Return best provider recommendations, quote validity, and expected savings.
  - When unavailable, populate `provider_costs` with `status: "unavailable"` and rationale.
- **Outputs**: Placeholder `provider_costs` block today, easily replaced with real comparison data later.

- **Inputs**: Amount, currencies, baseline mid-market rate.
- **Responsibilities**:
  - Invoke provider cost analyzer (once built) or return placeholder with status `unavailable`.
  - When available, supply effective-rate comparison and best provider recommendation.
- **Outputs**: `provider_costs` data with comparison table and chosen provider (future).

### 5. Decision Coordinator Agent
- **Inputs**: Aggregated state from previous agents, user preferences, provider cost info.
- **Responsibilities**:
  - Resolve conflicting signals (e.g., bullish market vs high event risk).
  - Produce final action (`convert_now`, `wait`, `stage_conversions`) with rationale and timeline.
  - Format output for downstream interfaces (CLI/API) and capture citations.
- **Outputs**: Final `recommendation` payload.

## Graph Topology
```
User Request → Input Validation → Parallel Agents (Market, Economic, Risk) → Decision Coordinator → Response Formatter (provider agent plugs in later)
```
- **Concurrency**: Market, economic, and risk agents run in parallel where dependencies allow. Risk agent depends on market output for volatility metrics; the future provider agent will hook in after risk completes.
- **State Object**: Python dataclass or TypedDict containing:
  - `request`: raw user request, normalized currency pair, amount, risk tolerance.
  - `market_analysis`, `economic_analysis`, `risk_assessment`, `provider_costs`.
  - `meta`: timestamps, errors, fallbacks used.
  - `recommendation`: final outcome, reasoning, confidence, required follow-ups.
- **Error Propagation**: Each agent stores its own `errors` list. Decision coordinator inspects and adjusts confidence/next steps.

## Tool & Service Integration Plan
- **LLM Access**: Decision coordinator uses `LLMManager.chat()` for narrative reasoning with failover; other agents can opt-in as needed.
- **ML Predictions**: Market agent calls `MLPredictor.predict()` with caching and fallbacks from the existing prediction module.
- **Data Fetching**:
  - Market agent uses `MultiProviderRateCollector` and `HistoricalDataCollector`.
  - Economic agent reuses `calendar_collector` and caches results to avoid redundant API hits.
  - Risk agent leverages volatility calculations already present in analytics modules.
- **Scraping Tool**: Decision coordinator or economic agent can trigger `AgentScrapingInterface` when knowledge gaps/time-sensitive cues appear.

## Observability & Logging
- Structured logging per agent (e.g., `logger = logging.getLogger("agent.market")`).
- Graph-level correlation IDs for tracing a single request across agents.
- Optional telemetry hook to persist recommendation outcomes for evaluation.

## Testing Strategy
- **Unit Tests**: Agent-level tests with mocked dependencies to validate decision logic.
- **Graph Simulation Tests**: Use deterministic fixtures to run entire LangGraph workflow and assert final recommendations.
- **Error Handling Tests**: Simulate failures (missing ML model, provider outage) and ensure graceful fallbacks.

## Implementation Phases & TODO Tracker
### Phase A – Foundations
- [x] Define shared state schema and request validation helpers.
- [x] Introduce LangGraph dependency and base graph setup.
- [x] Implement agent skeletons with dependency injection (no external calls yet).

### Phase B – Agent Integrations
- [x] Wire Market Analysis Agent to rate collector, historical analytics, ML predictor fallback.
- [x] Wire Economic Analysis Agent to calendar collectors (LLM summarizer optional).
- [x] Implement Risk Assessment Agent using volatility/risk metrics.
- [ ] (Optional) Integrate Provider Cost Agent once cost analyzer exists.

### Phase C – Coordination & Output
- [x] Implement Decision Coordinator aggregation logic and recommendation formatting.
- [x] Add response formatter suitable for CLI/API outputs (JSON + human-readable summary).
- [x] Ensure fallbacks for missing agent data.

### Phase D – Reliability & Testing
- [ ] Add structured logging and metrics collection hooks.
- [ ] Create unit tests for each agent with mocked dependencies.
- [ ] Build integration tests covering happy path, partial failures, and failure fallbacks.
- [ ] Document configuration knobs (timeouts, provider order, cache TTLs).

### Phase E – Enhancements (Post-MVP)
- [ ] Integrate provider cost analyzer once implemented.
- [ ] Add continuous evaluation loop storing decisions and outcomes for accuracy tracking.
- [ ] Expose LangGraph workflow via FastAPI endpoint (ties into future Phase 4 work).

## Acceptance Criteria for Agentic MVP
- LangGraph flow accepts `{currency_pair, amount, risk_tolerance, timeframe}` and returns structured recommendation with confidence.
- All primary agents populate their sections when dependencies succeed; errors are surfaced without crashing the graph.
- Tests cover ≥80 % of agent code paths, including fallback scenarios.
- Documentation updated to reflect deployment steps and configuration.

---
*Maintainer: (assign owner). Review cadence: weekly until MVP achieved.*
