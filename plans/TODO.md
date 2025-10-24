# Currency Assistant - Implementation TODO

**Last Updated**: October 24, 2025  
**Total Tasks**: 32  
**Estimated Duration**: 6-8 weeks

---

## Progress Overview

- [ ] Phase 0: Foundation & Infrastructure (3 tasks)
- [ ] Phase 1: Layer 1 Agents - Data Collection (6 tasks)
- [ ] Phase 2: Price Prediction Agent (5 tasks)
- [ ] Phase 3: Decision Engine Agent (4 tasks)
- [ ] Phase 4: Supervisor Agent & Orchestration (4 tasks)
- [ ] Phase 5: User Interfaces (3 tasks)
- [ ] Phase 6: Testing & Polish (3 tasks)
- [ ] Phase 7: Deployment & Production (3 tasks)
- [ ] Phase 8: ML Model Training (2 tasks)

---

## 📋 Phase 0: Foundation & Infrastructure

### [ ] 0.1 Project Structure & Configuration
**Priority**: 🔴 Critical | **Time**: 2-4 hours

**Tasks**:
- [ ] Create directory structure for all modules
- [ ] Create `config.yaml` with all agent configurations
- [ ] Create `.env.example` template with all required API keys
- [ ] Create `src/config.py` for centralized config loading
- [ ] Add configuration validation on startup

**Files to Create**:
- `config.yaml`
- `.env.example`
- `src/config.py`
- `src/utils/errors.py`
- `src/utils/logging.py`
- `src/utils/validation.py`

**Validation**:
```bash
python -c "from src.config import load_config; print(load_config())"
```

---

### [ ] 0.2 Database Schema & ORM
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create PostgreSQL schema (or SQLite for dev)
- [ ] Set up Alembic for migrations
- [ ] Create SQLAlchemy ORM models
- [ ] Create database connection manager
- [ ] Write initial migration

**Tables**:
- `conversations` - Session history for Supervisor
- `market_data_cache` - Cached exchange rates
- `prediction_history` - ML predictions for tracking
- `agent_metrics` - Performance monitoring
- `system_logs` - Audit trail

**Files to Create**:
- `src/database/models.py`
- `src/database/connection.py`
- `src/database/session.py`
- `alembic/versions/001_initial_schema.py`
- `alembic.ini`

**Validation**:
```bash
alembic upgrade head
python -c "from src.database.models import Conversation; print('DB models loaded')"
```

---

### [ ] 0.3 LangGraph State Design
**Priority**: 🔴 Critical | **Time**: 2-3 hours

**Tasks**:
- [ ] Define complete state schema (TypedDict)
- [ ] Create state initialization function
- [ ] Create state validation utilities
- [ ] Design state transitions between nodes
- [ ] Add conversation history to state

**Files to Create**:
- `src/agentic/state.py`
- `src/agentic/graph.py` (skeleton)

**Validation**:
```python
from src.agentic.state import AgentState, initialize_state
state = initialize_state("Convert 5000 USD to EUR")
```

---

## 📋 Phase 1: Layer 1 Agents - Data Collection

### [ ] 1.1 Market Data - Provider Clients
**Priority**: 🔴 Critical | **Time**: 4-6 hours | **Ref**: `market-data-agent-plan.md`

**Tasks**:
- [ ] Create base provider interface
- [ ] Implement ExchangeRate.host client
- [ ] Implement yfinance client
- [ ] Add provider health checks
- [ ] Add rate limiting and retries
- [ ] Write unit tests for each provider

**Files to Create**:
- `src/data_collection/providers/__init__.py`
- `src/data_collection/providers/base.py`
- `src/data_collection/providers/exchange_rate_host.py`
- `src/data_collection/providers/yfinance_client.py`
- `tests/providers/test_exchange_rate_host.py`
- `tests/providers/test_yfinance.py`

**Validation**:
```python
from src.data_collection.providers import ExchangeRateHostClient
client = ExchangeRateHostClient()
rate = await client.get_rate("USD", "EUR")
```

---

### [ ] 1.2 Market Data - Aggregation & Indicators
**Priority**: 🔴 Critical | **Time**: 4-6 hours

**Tasks**:
- [ ] Create rate aggregator (median consensus)
- [ ] Implement technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
- [ ] Create regime classifier (trend + bias)
- [ ] Add data quality metrics (dispersion, freshness)
- [ ] Implement caching layer (5s TTL)
- [ ] Create snapshot builder
- [ ] Write integration tests

**Files to Create**:
- `src/data_collection/market_data/aggregator.py`
- `src/data_collection/market_data/indicators.py`
- `src/data_collection/market_data/regime.py`
- `src/data_collection/market_data/snapshot.py`
- `src/data_collection/market_data/cache.py`
- `tests/market_data/test_aggregator.py`
- `tests/market_data/test_indicators.py`

**Validation**:
```python
from src.data_collection.market_data import get_market_snapshot
snapshot = await get_market_snapshot("USD/EUR")
print(snapshot.mid_rate, snapshot.indicators.rsi_14)
```

---

### [ ] 1.3 Market Data - LangGraph Node
**Priority**: 🔴 Critical | **Time**: 2-3 hours

**Tasks**:
- [ ] Create Market Data agent node
- [ ] Integrate with LangGraph state
- [ ] Add error handling and fallbacks
- [ ] Add performance logging
- [ ] Write node integration test

**Files to Create**:
- `src/agentic/nodes/market_data.py`
- `tests/agentic/nodes/test_market_data.py`

**Validation**:
```python
from src.agentic.nodes.market_data import market_data_node
from src.agentic.state import initialize_state
state = initialize_state("Convert USD to EUR")
result = await market_data_node(state)
assert result["market_snapshot"] is not None
```

---

### [ ] 1.4 Market Intelligence - Serper Integration
**Priority**: 🔴 Critical | **Time**: 4-5 hours | **Ref**: `market-intelligence.md`

**Tasks**:
- [ ] Create Serper API client
- [ ] Implement economic calendar search
- [ ] Implement news search
- [ ] Add retry and error handling
- [ ] Write unit tests

**Files to Create**:
- `src/data_collection/market_intelligence/serper_client.py`
- `src/data_collection/market_intelligence/calendar_collector.py`
- `src/data_collection/market_intelligence/news_collector.py`
- `tests/market_intelligence/test_serper.py`

**Validation**:
```python
from src.data_collection.market_intelligence import SerperClient
client = SerperClient()
results = await client.search("US economic calendar October 2025")
```

---

### [ ] 1.5 Market Intelligence - LLM Extraction
**Priority**: 🔴 Critical | **Time**: 5-6 hours

**Tasks**:
- [ ] Create calendar event extractor (gpt-5-mini)
- [ ] Create news sentiment classifier (gpt-5-mini)
- [ ] Create narrative generator (gpt-4o)
- [ ] Implement JSON schema validation
- [ ] Add LLM error handling and retries
- [ ] Write extraction tests with mock LLM

**Files to Create**:
- `src/data_collection/market_intelligence/extractors/calendar_extractor.py`
- `src/data_collection/market_intelligence/extractors/news_classifier.py`
- `src/data_collection/market_intelligence/extractors/narrative_generator.py`
- `src/data_collection/market_intelligence/models.py`
- `tests/market_intelligence/test_extraction.py`

**Validation**:
```python
from src.data_collection.market_intelligence.extractors import extract_calendar_events
events = await extract_calendar_events(search_results, "USD", "EUR")
```

---

### [ ] 1.6 Market Intelligence - Aggregation & Node
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create intelligence aggregator
- [ ] Compute policy bias from events
- [ ] Calculate next high-impact event ETA
- [ ] Create Market Intelligence node
- [ ] Write integration tests

**Files to Create**:
- `src/data_collection/market_intelligence/aggregator.py`
- `src/data_collection/market_intelligence/bias_calculator.py`
- `src/agentic/nodes/market_intelligence.py`
- `tests/agentic/nodes/test_market_intelligence.py`

**Validation**:
```python
from src.agentic.nodes.market_intelligence import market_intelligence_node
result = await market_intelligence_node(state)
print(result["intelligence_report"]["overall_bias"])
```

---

## 📋 Phase 2: Price Prediction Agent

### [ ] 2.1 Data Pipeline & Feature Engineering
**Priority**: 🔴 Critical | **Time**: 4-6 hours | **Ref**: `price-prediction.md`

**Tasks**:
- [ ] Create historical OHLC data loader (yfinance)
- [ ] Implement technical feature builder (SMA, EMA, RSI, etc.)
- [ ] Add optional Market Intelligence features
- [ ] Create feature validation
- [ ] Write data pipeline tests

**Files to Create**:
- `src/prediction/data_loader.py`
- `src/prediction/feature_builder.py`
- `src/prediction/models.py`
- `tests/prediction/test_data_loader.py`
- `tests/prediction/test_features.py`

**Validation**:
```python
from src.prediction.data_loader import load_historical_data
from src.prediction.feature_builder import build_features
data = await load_historical_data("USDEUR=X", days=90)
features = build_features(data, mode="price_only")
```

---

### [ ] 2.2 Model Registry & Storage
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create model registry (JSON + pickle)
- [ ] Implement model save/load
- [ ] Add model metadata tracking
- [ ] Create model versioning
- [ ] Write registry tests

**Files to Create**:
- `src/prediction/registry.py`
- `src/prediction/config.py`
- `models/` directory
- `tests/prediction/test_registry.py`

**Validation**:
```python
from src.prediction.registry import ModelRegistry
registry = ModelRegistry()
registry.register_model("usd_eur_v1", model_obj, metadata)
```

---

### [ ] 2.3 LightGBM Backend
**Priority**: 🔴 Critical | **Time**: 5-7 hours

**Tasks**:
- [ ] Create base predictor interface
- [ ] Implement LightGBM backend
- [ ] Add quantile regression
- [ ] Add direction probability estimation
- [ ] Implement quality gates
- [ ] Create calibration utilities
- [ ] Write backend tests

**Files to Create**:
- `src/prediction/backends/base.py`
- `src/prediction/backends/lightgbm_backend.py`
- `src/prediction/utils/calibration.py`
- `tests/prediction/backends/test_lightgbm.py`

**Validation**:
```python
from src.prediction.backends import LightGBMBackend
backend = LightGBMBackend(config)
predictions = await backend.predict(features, horizons=[1, 7, 30])
```

---

### [ ] 2.4 Fallback Heuristics
**Priority**: 🟡 Important | **Time**: 2-3 hours

**Tasks**:
- [ ] Implement MA crossover heuristic
- [ ] Implement RSI reversion heuristic
- [ ] Add fallback confidence scoring
- [ ] Write fallback tests

**Files to Create**:
- `src/prediction/utils/fallback.py`
- `tests/prediction/test_fallback.py`

**Validation**:
```python
from src.prediction.utils.fallback import fallback_predict
pred = fallback_predict(indicators, horizons=[1, 7, 30])
```

---

### [ ] 2.5 Prediction Agent & Caching
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create main predictor with caching
- [ ] Implement quality gate checks
- [ ] Add prediction node for LangGraph
- [ ] Write integration tests

**Files to Create**:
- `src/prediction/predictor.py`
- `src/agentic/nodes/prediction.py`
- `tests/prediction/test_predictor.py`
- `tests/agentic/nodes/test_prediction.py`

**Validation**:
```python
from src.agentic.nodes.prediction import prediction_node
result = await prediction_node(state)
print(result["price_forecast"]["predictions"]["7"])
```

---

## 📋 Phase 3: Decision Engine Agent

### [ ] 3.1 Decision Model Core
**Priority**: 🔴 Critical | **Time**: 5-6 hours | **Ref**: `decision-engine.plan.md`

**Tasks**:
- [ ] Create utility calculation model
- [ ] Implement action scoring (convert_now, staged, wait)
- [ ] Add risk penalty calculation
- [ ] Add urgency fit calculation
- [ ] Write decision model tests

**Files to Create**:
- `src/decision/models.py`
- `src/decision/utility.py`
- `src/decision/risk_calculator.py`
- `src/decision/config.py`
- `tests/decision/test_utility.py`

**Validation**:
```python
from src.decision.utility import calculate_utility
utility = calculate_utility(expected_improvement=0.5, risk_penalty=0.2, ...)
```

---

### [ ] 3.2 Staging Algorithm
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create staging calculator
- [ ] Implement tranche sizing
- [ ] Add event-aware spacing
- [ ] Write staging tests

**Files to Create**:
- `src/decision/staging.py`
- `tests/decision/test_staging.py`

**Validation**:
```python
from src.decision.staging import create_staged_plan
plan = create_staged_plan(amount=5000, timeframe_days=7, urgency="normal")
```

---

### [ ] 3.3 Heuristic Fallbacks
**Priority**: 🟡 Important | **Time**: 2-3 hours

**Tasks**:
- [ ] Implement event-gating rules
- [ ] Implement momentum-based heuristics
- [ ] Add conservative/aggressive adjustments
- [ ] Write heuristic tests

**Files to Create**:
- `src/decision/heuristics.py`
- `tests/decision/test_heuristics.py`

**Validation**:
```python
from src.decision.heuristics import heuristic_decision
decision = heuristic_decision(indicators, intelligence, user_params)
```

---

### [ ] 3.4 Decision Engine Node
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create main decision engine
- [ ] Integrate all inputs (market, intelligence, prediction)
- [ ] Generate rationale and confidence
- [ ] Create decision node for LangGraph
- [ ] Write integration tests

**Files to Create**:
- `src/decision/engine.py`
- `src/agentic/nodes/decision.py`
- `tests/agentic/nodes/test_decision.py`

**Validation**:
```python
from src.agentic.nodes.decision import decision_node
result = await decision_node(state)
print(result["recommendation"]["action"])
```

---

## 📋 Phase 4: Supervisor Agent & Orchestration

### [ ] 4.1 NLU Parameter Extraction
**Priority**: 🔴 Critical | **Time**: 4-6 hours | **Ref**: `supervisor-agent.plan.md`

**Tasks**:
- [ ] Create parameter extractor using LLM (gpt-4o)
- [ ] Implement currency pair parsing
- [ ] Implement amount extraction
- [ ] Implement risk/urgency/timeframe inference
- [ ] Add validation and clarification prompts
- [ ] Write extraction tests

**Files to Create**:
- `src/agentic/nlu/extractor.py`
- `src/agentic/nlu/prompts.py`
- `src/agentic/nlu/validation.py`
- `tests/agentic/nlu/test_extraction.py`

**Validation**:
```python
from src.agentic.nlu.extractor import extract_parameters
params = await extract_parameters("I need to convert 5000 USD to EUR today")
```

---

### [ ] 4.2 Conversation Manager
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create conversation session manager
- [ ] Implement memory storage/retrieval
- [ ] Add clarification flow
- [ ] Write conversation tests

**Files to Create**:
- `src/agentic/conversation/session.py`
- `src/agentic/conversation/memory.py`
- `tests/agentic/conversation/test_session.py`

**Validation**:
```python
from src.agentic.conversation import ConversationSession
session = ConversationSession(user_id="test")
session.add_turn("user", "Convert USD to EUR")
```

---

### [ ] 4.3 Response Generator
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create response formatter
- [ ] Generate friendly recommendation text
- [ ] Add visualization data preparation
- [ ] Write response tests

**Files to Create**:
- `src/agentic/response/generator.py`
- `src/agentic/response/formatter.py`
- `tests/agentic/response/test_generator.py`

**Validation**:
```python
from src.agentic.response import generate_response
response = generate_response(recommendation, state)
```

---

### [ ] 4.4 LangGraph Orchestration
**Priority**: 🔴 Critical | **Time**: 4-6 hours

**Tasks**:
- [ ] Build complete LangGraph workflow
- [ ] Add conditional routing logic
- [ ] Implement Layer 1 parallel dispatch
- [ ] Add error handling and retries
- [ ] Create Supervisor node
- [ ] Write end-to-end tests

**Files to Create**:
- `src/agentic/graph.py` (complete)
- `src/agentic/nodes/supervisor.py`
- `src/agentic/routing.py`
- `tests/agentic/test_graph.py`

**Validation**:
```python
from src.agentic.graph import create_graph
graph = create_graph()
result = await graph.ainvoke({"user_query": "Convert 5000 USD to EUR"})
```

---

## 📋 Phase 5: User Interfaces

### [ ] 5.1 TUI (Terminal User Interface)
**Priority**: 🔴 Critical | **Time**: 6-8 hours | **Ref**: `tui-interface.plan.md`

**Tasks**:
- [ ] Set up Rich library
- [ ] Create main TUI loop
- [ ] Implement multi-turn conversation
- [ ] Add visual feedback (spinners, progress)
- [ ] Display recommendation with tables
- [ ] Add parameter review/edit
- [ ] Write TUI tests

**Files to Create**:
- `src/ui/tui/app.py`
- `src/ui/tui/display.py`
- `src/ui/tui/input_handler.py`
- `src/ui/tui/renderer.py`
- `src/ui/tui/config.py`
- `tests/ui/test_tui.py`

**Validation**:
```bash
python -m src.ui.tui.app
```

---

### [ ] 5.2 FastAPI Backend
**Priority**: 🟡 Important | **Time**: 4-5 hours | **Ref**: `web-ui.plan.md`

**Tasks**:
- [ ] Create FastAPI app
- [ ] Add REST endpoints (POST /recommend, GET /session/{id})
- [ ] Add SSE endpoint for streaming
- [ ] Add health check endpoint
- [ ] Add CORS middleware
- [ ] Write API tests

**Files to Create**:
- `src/ui/web/main.py`
- `src/ui/web/routes/recommendations.py`
- `src/ui/web/routes/sessions.py`
- `src/ui/web/middleware.py`
- `tests/ui/web/test_api.py`

**Validation**:
```bash
uvicorn src.ui.web.main:app --reload
curl -X POST http://localhost:8000/recommend -d '{"query": "Convert USD to EUR"}'
```

---

### [ ] 5.3 Web Frontend (HTMX)
**Priority**: 🟡 Important | **Time**: 6-8 hours

**Tasks**:
- [ ] Create HTML templates (Jinja2)
- [ ] Add HTMX interactions
- [ ] Create chart components (Chart.js)
- [ ] Add loading states
- [ ] Style with CSS
- [ ] Test in browser

**Files to Create**:
- `src/ui/web/templates/base.html`
- `src/ui/web/templates/chat.html`
- `src/ui/web/templates/recommendation.html`
- `src/ui/web/static/css/style.css`
- `src/ui/web/static/js/charts.js`

**Validation**:
```bash
# Open http://localhost:8000 in browser
```

---

## 📋 Phase 6: Testing & Polish

### [ ] 6.1 Integration Testing
**Priority**: 🔴 Critical | **Time**: 4-6 hours

**Tasks**:
- [ ] Write end-to-end test scenarios
- [ ] Test with mock providers
- [ ] Test error handling paths
- [ ] Test partial failures
- [ ] Test offline mode

**Files to Create**:
- `tests/integration/test_full_workflow.py`
- `tests/integration/test_error_scenarios.py`
- `tests/integration/fixtures.py`

---

### [ ] 6.2 Performance Testing
**Priority**: 🟡 Important | **Time**: 2-3 hours

**Tasks**:
- [ ] Measure end-to-end latency
- [ ] Test cache effectiveness
- [ ] Test concurrent requests
- [ ] Optimize slow paths

**Files to Create**:
- `tests/performance/test_latency.py`
- `tests/performance/test_cache.py`

---

### [ ] 6.3 Documentation
**Priority**: 🟡 Important | **Time**: 4-5 hours

**Tasks**:
- [ ] Write API documentation (OpenAPI/Swagger)
- [ ] Write deployment guide
- [ ] Write user guide
- [ ] Add inline code documentation
- [ ] Create README with quick start

**Files to Create**:
- `README.md`
- `docs/API.md`
- `docs/DEPLOYMENT.md`
- `docs/USER_GUIDE.md`

---

## 📋 Phase 7: Deployment & Production

### [ ] 7.1 Docker Setup
**Priority**: 🔴 Critical | **Time**: 3-4 hours

**Tasks**:
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Add PostgreSQL service
- [ ] Add Redis service
- [ ] Test Docker build and run

**Files to Create**:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Validation**:
```bash
docker-compose up --build
```

---

### [ ] 7.2 CI/CD Pipeline
**Priority**: 🟡 Important | **Time**: 2-3 hours

**Tasks**:
- [ ] Create GitHub Actions workflow
- [ ] Add linting (black, ruff)
- [ ] Add type checking (mypy)
- [ ] Add test running
- [ ] Add coverage reporting

**Files to Create**:
- `.github/workflows/test.yml`
- `.github/workflows/deploy.yml`
- `pyproject.toml` (update)

---

### [ ] 7.3 Monitoring & Logging
**Priority**: 🟡 Important | **Time**: 3-4 hours

**Tasks**:
- [ ] Add Prometheus metrics
- [ ] Add health check dashboard
- [ ] Set up log aggregation
- [ ] Add alerting rules

**Files to Create**:
- `src/monitoring/metrics.py`
- `src/monitoring/health.py`
- `prometheus.yml`

---

## 📋 Phase 8: ML Model Training

### [ ] 8.1 Training Pipeline
**Priority**: 🟡 Important | **Time**: 6-8 hours

**Tasks**:
- [ ] Create training script for LightGBM
- [ ] Collect historical data for major pairs
- [ ] Implement cross-validation
- [ ] Track experiments
- [ ] Save trained models to registry

**Files to Create**:
- `scripts/train_model.py`
- `scripts/evaluate_model.py`
- `scripts/backtest_model.py`
- `notebooks/model_exploration.ipynb`

**Validation**:
```bash
python scripts/train_model.py --pair USD/EUR --horizons 1,7,30
```

---

### [ ] 8.2 Model Deployment
**Priority**: 🟡 Important | **Time**: 2-3 hours

**Tasks**:
- [ ] Train models for top 5 currency pairs
- [ ] Validate quality gates
- [ ] Deploy to model registry
- [ ] Test in production

**Validation**:
```python
from src.prediction import MLPredictor
predictor = MLPredictor()
pred = await predictor.predict(...)
```

---

## 📊 Summary Statistics

| Phase | Tasks | Estimated Time | Status |
|-------|-------|----------------|--------|
| Phase 0 | 3 | 7-11 hours | ⏳ Not Started |
| Phase 1 | 6 | 22-30 hours | ⏳ Not Started |
| Phase 2 | 5 | 17-24 hours | ⏳ Not Started |
| Phase 3 | 4 | 13-17 hours | ⏳ Not Started |
| Phase 4 | 4 | 14-20 hours | ⏳ Not Started |
| Phase 5 | 3 | 16-21 hours | ⏳ Not Started |
| Phase 6 | 3 | 10-14 hours | ⏳ Not Started |
| Phase 7 | 3 | 8-11 hours | ⏳ Not Started |
| Phase 8 | 2 | 8-11 hours | ⏳ Not Started |
| **Total** | **32** | **115-159 hours** | **0% Complete** |

---

## 🎯 Current Focus

**Next Task**: Phase 0.1 - Project Structure & Configuration

**Command to Start**:
```bash
mkdir -p src/{config,data_collection/{providers,market_data,market_intelligence},prediction,decision,agentic/{nodes,nlu,conversation,response},ui/{tui,web},utils,database}
```

---

## 📝 Notes

- Each task builds on previous tasks - follow sequentially
- Mark tasks complete with `[x]` as you finish them
- Update estimated time based on actual experience
- Add notes about blockers or issues encountered
- Reference detailed plans in `.cursor/plans/` for implementation guidance

---

**For detailed implementation steps, see**: `.cursor/plans/IMPLEMENTATION_ROADMAP.md`

