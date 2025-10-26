# Currency Assistant - Implementation Roadmap

**Status**: In Progress  
**Start Date**: October 24, 2025  
**Estimated Duration**: 6-8 weeks  
**Current Phase**: Phase 3 - Decision Engine Agent

---

## 🎯 Overview

This roadmap provides a step-by-step implementation guide for the Currency Assistant multi-agent system. Each phase builds on the previous one, with clear deliverables and validation steps.

### System Architecture
```
User Query (TUI/WebUI)
    ↓
Supervisor Agent (NLU + Orchestration)
    ↓
┌───────────────────┬─────────────────────┬──────────────────┐
│   Market Data     │  Market Intelligence │  Price Prediction │
│   Agent (Layer 1) │  Agent (Layer 1)     │  Agent (Layer 2) │
└───────────────────┴─────────────────────┴──────────────────┘
    ↓
Decision Engine Agent (Layer 3)
    ↓
Supervisor (Response Generation)
    ↓
User Response
```

---

## 📋 Phase 0: Foundation & Infrastructure (Week 1)

**Goal**: Set up core infrastructure that all agents will use

### 0.1: Project Structure & Configuration
**Priority**: 🔴 Critical  
**Time Estimate**: 2-4 hours

**Tasks**:
- [ ] Create directory structure for all modules
- [ ] Set up `config.yaml` with all agent configurations
- [ ] Create `.env.example` template with all required API keys
- [ ] Create `src/config.py` for centralized config loading
- [ ] Add configuration validation on startup

**Deliverables**:
```
src/
├── config.py                    # Config loader
├── llm/                         # ✅ Already exists
├── data_collection/
│   ├── providers/              # API clients
│   ├── market_data/            # Market Data Agent
│   └── market_intelligence/    # Market Intelligence Agent
├── prediction/                  # Price Prediction Agent
├── decision/                    # Decision Engine Agent
├── agentic/
│   ├── state.py                # LangGraph state schema
│   ├── graph.py                # LangGraph graph builder
│   └── nodes/                  # Agent node implementations
├── ui/
│   ├── tui/                    # Terminal UI
│   └── web/                    # Web UI
└── utils/
    ├── logging.py              # Structured logging
    ├── errors.py               # Custom exceptions
    └── validation.py           # Input validation
```

**Files to Create**:
1. `config.yaml` - Complete system configuration
2. `.env.example` - API key template
3. `src/config.py` - Config loader with validation
4. `src/utils/errors.py` - Standard error classes
5. `src/utils/logging.py` - Logger setup with correlation IDs

**Validation**:
```bash
python -c "from src.config import load_config; print(load_config())"
```

---

### 0.2: Database Schema & ORM
**Priority**: 🔴 Critical  
**Time Estimate**: 2-3 hours (simplified with SQLite)

**Tasks**:
- [ ] Create SQLite database file
- [ ] Set up Alembic for migrations
- [ ] Create SQLAlchemy ORM models
- [ ] Create database connection manager
- [ ] Write initial migration

**Tables**:
1. `conversations` - Session history for Supervisor
2. `prediction_history` - ML predictions for tracking
3. `agent_metrics` - Performance monitoring
4. `system_logs` - Audit trail

**Files to Create**:
1. `src/database/models.py` - ORM models
2. `src/database/connection.py` - SQLite connection
3. `src/database/session.py` - Session management
4. `src/cache.py` - In-memory cache implementation
5. `alembic/versions/001_initial_schema.py` - (Optional; skipping for SQLite)
6. `alembic.ini` - (Optional; skipping for SQLite)

**Validation**:
```bash
python -c "from src.database.models import Conversation; print('DB models loaded')"
python -c "from src.cache import SimpleCache; cache = SimpleCache(); print('Cache ready')"
```

---

### 0.3: LangGraph State Design
**Priority**: 🔴 Critical  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Define complete state schema (TypedDict)
- [ ] Create state initialization function
- [ ] Create state validation utilities
- [ ] Design state transitions between nodes
- [ ] Add conversation history to state

**State Schema** (outline):
```python
class AgentState(TypedDict):
    # User Request
    correlation_id: str
    user_query: str
    currency_pair: str
    amount: float
    risk_tolerance: str
    urgency: str
    timeframe: str
    
    # Layer 1: Market Data
    market_snapshot: Optional[Dict]
    market_data_status: str
    
    # Layer 1: Market Intelligence
    intelligence_report: Optional[Dict]
    intelligence_status: str
    
    # Layer 2: Prediction
    price_forecast: Optional[Dict]
    prediction_status: str
    
    # Layer 3: Decision
    recommendation: Optional[Dict]
    decision_status: str
    
    # Conversation
    conversation_history: List[Dict]
    clarifications_needed: List[str]
    
    # System
    warnings: List[str]
    errors: List[str]
    processing_stage: str
```

**Files to Create**:
1. `src/agentic/state.py` - State schema and utilities
2. `src/agentic/graph.py` - LangGraph graph skeleton

**Validation**:
```python
from src.agentic.state import AgentState, initialize_state
state = initialize_state("Convert 5000 USD to EUR")
```

---

### 0.4: Logging & Error Handling
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Set up structured logging (JSON format)
- [ ] Implement correlation ID propagation
- [ ] Create custom exception hierarchy
- [ ] Add error recovery decorators
- [ ] Create health check endpoint

**Files to Create**:
1. `src/utils/logging.py` - Logger with correlation IDs
2. `src/utils/errors.py` - Custom exceptions
3. `src/utils/decorators.py` - Retry, timeout decorators
4. `src/health.py` - System health check

**Validation**:
```python
from src.utils.logging import get_logger
logger = get_logger(__name__, correlation_id="test-123")
logger.info("Test log", extra={"key": "value"})
```

---

## 📋 Phase 1: Layer 1 Agents - Data Collection (Week 2)

**Goal**: Implement Market Data and Market Intelligence agents

### 1.1: Market Data Agent - Provider Clients
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Reference**: `plans/market-data-agent-plan.md`

**Tasks**:
- [x] Create base provider interface
- [x] Implement ExchangeRate.host client
- [x] Implement yfinance client
- [x] Add provider health checks
- [x] Add rate limiting and retries
- [x] Write unit tests for each provider

**Files to Create**:
1. `src/data_collection/providers/__init__.py`
2. `src/data_collection/providers/base.py` - Base provider interface
3. `src/data_collection/providers/exchange_rate_host.py`
4. `src/data_collection/providers/yfinance_client.py`
5. `tests/unit/test_providers/test_exchange_rate_host.py`
6. `tests/unit/test_providers/test_yfinance.py`

**Validation**:
```python
from src.data_collection.providers import ExchangeRateHostClient
client = ExchangeRateHostClient()
rate = await client.get_rate("USD", "EUR")
print(rate)
```

---

### 1.2: Market Data Agent - Aggregation & Indicators
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Tasks**:
- [x] Create rate aggregator (median consensus)
- [x] Implement technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
- [x] Create regime classifier (trend + bias)
- [x] Add data quality metrics (dispersion, freshness)
- [x] Implement caching layer (5s TTL)
- [x] Create snapshot builder
- [x] Write integration tests

**Files to Create**:
1. `src/data_collection/market_data/aggregator.py`
2. `src/data_collection/market_data/indicators.py`
3. `src/data_collection/market_data/regime.py`
4. `src/data_collection/market_data/snapshot.py`
5. `src/data_collection/market_data/cache.py`
6. `tests/unit/test_market_data/test_aggregator.py`
7. `tests/unit/test_market_data/test_indicators.py`

**Validation**:
```python
from src.data_collection.market_data import get_market_snapshot
snapshot = await get_market_snapshot("USD/EUR")
print(snapshot.mid_rate, snapshot.indicators.rsi_14)
```

---

### 1.3: Market Data Agent - LangGraph Node
**Priority**: 🔴 Critical  
**Time Estimate**: 2-3 hours

**Tasks**:
- [x] Create Market Data agent node
- [x] Integrate with LangGraph state
- [x] Add error handling and fallbacks
- [x] Add performance logging
- [x] Write node integration test

**Files to Create**:
1. `src/agentic/nodes/market_data.py`
2. `tests/unit/test_agentic/test_nodes/test_market_data.py`

**Validation**:
```python
from src.agentic.nodes.market_data import market_data_node
from src.agentic.state import initialize_state
state = initialize_state("Convert USD to EUR")
result = await market_data_node(state)
assert result["market_snapshot"] is not None
```

---

### 1.4: Market Intelligence Agent - Serper Integration
**Priority**: 🔴 Critical  
**Time Estimate**: 4-5 hours

**Reference**: `plans/market-intelligence.md`

**Tasks**:
- [x] Create Serper API client
- [x] Implement economic calendar search
- [x] Implement news search
- [x] Add retry and error handling
- [x] Write unit tests

**Files to Create**:
1. `src/data_collection/market_intelligence/serper_client.py`
2. `src/data_collection/market_intelligence/calendar_collector.py`
3. `src/data_collection/market_intelligence/news_collector.py`
4. `tests/unit/test_market_intelligence/test_serper_client.py`

**Validation**:
```python
from src.data_collection.market_intelligence import SerperClient
client = SerperClient()
results = await client.search("US economic calendar October 2025")
print(len(results))
```

---

### 1.5: Market Intelligence Agent - LLM Extraction
**Priority**: 🔴 Critical  
**Time Estimate**: 5-6 hours

**Tasks**:
- [x] Create calendar event extractor (gpt-5-mini)
- [x] Create news sentiment classifier (gpt-5-mini)
- [x] Create narrative generator (gpt-4o)
- [x] Implement JSON schema validation
- [x] Add LLM error handling and retries
- [x] Write extraction tests with mock LLM

**Files to Create**:
1. `src/data_collection/market_intelligence/extractors/calendar_extractor.py`
2. `src/data_collection/market_intelligence/extractors/news_classifier.py`
3. `src/data_collection/market_intelligence/extractors/narrative_generator.py`
4. `src/data_collection/market_intelligence/models.py` - Data contracts
5. `tests/market_intelligence/test_extraction.py`

**Validation**:
```python
from src.data_collection.market_intelligence.extractors import extract_calendar_events
events = await extract_calendar_events(search_results, "USD", "EUR")
print(events[0].event_name, events[0].importance)
```

---

### 1.6: Market Intelligence Agent - Aggregation & Node
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [x] Create intelligence aggregator
- [x] Compute policy bias from events
- [x] Calculate next high-impact event ETA
- [x] Create Market Intelligence node
- [x] Write integration tests

**Files to Create**:
1. `src/data_collection/market_intelligence/aggregator.py`
2. `src/data_collection/market_intelligence/bias_calculator.py`
3. `src/agentic/nodes/market_intelligence.py`
4. `tests/unit/test_agentic/test_nodes/test_market_intelligence.py`

**Validation**:
```python
from src.agentic.nodes.market_intelligence import market_intelligence_node
result = await market_intelligence_node(state)
print(result["intelligence_report"]["overall_bias"])
```

---

## 📋 Phase 2: Layer 2 Agent - Price Prediction (Week 3) ✅ COMPLETE

**Goal**: Implement ML-based price forecasting

### 2.1: Data Pipeline & Feature Engineering ✅ COMPLETE
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Reference**: `plans/price-prediction.md`

**Tasks**:
- [x] Create historical OHLC data loader (yfinance)
- [x] Implement technical feature builder (SMA, EMA, RSI, etc.)
- [x] Add optional Market Intelligence features
- [x] Create feature validation
- [x] Write data pipeline tests

**Files to Create**:
1. `src/prediction/data_loader.py`
2. `src/prediction/feature_builder.py`
3. `src/prediction/models.py` - Data contracts
4. `tests/prediction/test_data_loader.py`
5. `tests/prediction/test_features.py`

**Validation**:
```python
from src.prediction.data_loader import load_historical_data
from src.prediction.feature_builder import build_features
data = await load_historical_data("USDEUR=X", days=90)
features = build_features(data, mode="price_only")
print(features.columns)
```

---

### 2.2: Model Registry & Storage ✅ COMPLETE
**Priority**: 🔴 Critical  
**Time Estimate**: 2-3 hours (simplified with JSON + pickle)

**Tasks**:
- [x] Create simple JSON model registry
- [x] Implement model save/load with pickle
- [x] Add model metadata tracking (accuracy, currency_pair, version)
- [x] Write registry tests

**Files to Create**:
1. `src/prediction/registry.py` - JSON + pickle based registry
2. `src/prediction/config.py`
3. `data/models/prediction/` - Directory for model files
4. `data/models/prediction_registry.json` - Metadata file
5. `tests/prediction/test_registry.py`

**Implementation Approach**:
- Save models as `.pkl` files using pickle
- Track metadata in single `registry.json` file
- No external services needed (MLflow skipped for simplicity)

**Validation**:
```python
from src.prediction.registry import ModelRegistry
registry = ModelRegistry()
registry.save_model("usd_eur_v1", model_obj, {
    "accuracy": 0.68,
    "currency_pair": "USD/EUR",
    "created_at": "2025-10-25"
})
loaded = registry.load_model("usd_eur_v1")
metadata = registry.get_metadata("usd_eur_v1")
```

---

### 2.3: Model Backends (LightGBM + LSTM) + Explainability ✅ COMPLETE
**Priority**: 🔴 Critical  
**Time Estimate**: 5-7 hours

**Tasks**:
- [x] Create base predictor interface
- [x] Implement LightGBM backend (daily/weekly)
- [x] Implement LSTM backend (intraday 1h/4h/24h)
- [x] Add quantile regression (LightGBM)
- [x] Add direction probability estimation (LightGBM)
- [x] Implement quality gates
- [x] Create calibration utilities
- [x] Integrate SHAP for model explainability (LightGBM)
- [x] Generate SHAP visualizations (waterfall, force plots)
- [x] Write backend tests for both backends

**Files to Create**:
1. `src/prediction/backends/base.py`
2. `src/prediction/backends/lightgbm_backend.py`
3. `src/prediction/backends/lstm_backend.py`
4. `src/prediction/explainer.py` - SHAP integration for web UI
5. `src/prediction/utils/calibration.py`
6. `tests/prediction/backends/test_lightgbm.py`
7. `tests/prediction/backends/test_lstm.py`

**Validation**:
```python
from src.prediction.backends import LightGBMBackend
from src.prediction.explainer import PredictionExplainer

backend = LightGBMBackend(config)
predictions = backend.predict(features, horizons=[1, 7, 30])
print(predictions.mean_change)

# SHAP explanations for web UI
explainer = PredictionExplainer(backend.model)
shap_plot_base64 = explainer.generate_waterfall_plot(features, feature_names)
feature_importance = explainer.get_feature_importance(top_n=5)

# LSTM backend (intraday)
from src.prediction.backends.lstm_backend import LSTMBackend
lstm = LSTMBackend()
# predictions_intraday = lstm.predict(intraday_features, horizons=[1])  # to be implemented
```

**Note**: SHAP provides visual explanations for web UI, making predictions more interpretable.

---

### 2.4: Fallback Heuristics ✅ COMPLETE
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [x] Implement MA crossover heuristic
- [x] Implement RSI reversion heuristic
- [x] Add fallback confidence scoring
- [x] Write fallback tests

**Files to Create**:
1. `src/prediction/utils/fallback.py`
2. `tests/prediction/test_fallback.py`

**Validation**:
```python
from src.prediction.utils.fallback import fallback_predict
pred = fallback_predict(indicators, horizons=[1, 7, 30])
print(pred.mean_change)
```

---

### 2.5: Prediction Agent & Caching ✅ COMPLETE
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [x] Create main predictor with caching
- [x] Implement quality gate checks
- [x] Add prediction node for LangGraph
- [x] Write integration tests

**Files to Create**:
1. `src/prediction/predictor.py`
2. `src/agentic/nodes/prediction.py`
3. `tests/prediction/test_predictor.py`
4. `tests/agentic/nodes/test_prediction.py`

**Validation**:
```python
from src.agentic.nodes.prediction import prediction_node
result = await prediction_node(state)
print(result["price_forecast"]["predictions"]["7"])
```

---

## 📋 Phase 3: Layer 3 Agent - Decision Engine (Week 4)

**Goal**: Implement recommendation logic

### 3.1: Decision Model Core
**Priority**: 🔴 Critical  
**Time Estimate**: 5-6 hours

**Reference**: `/plans/decision-engine.plan.md`

**Tasks**:
- [ ] Create utility calculation model
- [ ] Implement action scoring (convert_now, staged, wait)
- [ ] Add risk penalty calculation
- [ ] Add urgency fit calculation
- [ ] Write decision model tests

**Files to Create**:
1. `src/decision/models.py` - Data contracts
2. `src/decision/utility.py` - Utility model
3. `src/decision/risk_calculator.py`
4. `src/decision/config.py`
5. `tests/decision/test_utility.py`

**Validation**:
```python
from src.decision.utility import calculate_utility
utility = calculate_utility(
    expected_improvement=0.5,
    risk_penalty=0.2,
    cost=0.1,
    urgency_fit=0.3,
    weights={"profit": 0.4, "risk": 0.3, "cost": 0.2, "urgency": 0.1}
)
```

---

### 3.2: Staging Algorithm
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [ ] Create staging calculator
- [ ] Implement tranche sizing
- [ ] Add event-aware spacing
- [ ] Write staging tests

**Files to Create**:
1. `src/decision/staging.py`
2. `tests/decision/test_staging.py`

**Validation**:
```python
from src.decision.staging import create_staged_plan
plan = create_staged_plan(
    amount=5000,
    timeframe_days=7,
    urgency="normal",
    high_impact_events=[2, 5]
)
print(plan.tranches, plan.spacing_days)
```

---

### 3.3: Heuristic Fallbacks
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Implement event-gating rules
- [ ] Implement momentum-based heuristics
- [ ] Add conservative/aggressive adjustments
- [ ] Write heuristic tests

**Files to Create**:
1. `src/decision/heuristics.py`
2. `tests/decision/test_heuristics.py`

**Validation**:
```python
from src.decision.heuristics import heuristic_decision
decision = heuristic_decision(indicators, intelligence, user_params)
print(decision.action, decision.rationale)
```

---

### 3.4: Decision Engine Node
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [ ] Create main decision engine
- [ ] Integrate all inputs (market, intelligence, prediction)
- [ ] Generate rationale and confidence
- [ ] Create decision node for LangGraph
- [ ] Write integration tests

**Files to Create**:
1. `src/decision/engine.py`
2. `src/agentic/nodes/decision.py`
3. `tests/agentic/nodes/test_decision.py`

**Validation**:
```python
from src.agentic.nodes.decision import decision_node
result = await decision_node(state)
print(result["recommendation"]["action"])
print(result["recommendation"]["rationale"])
```

---

## 📋 Phase 4: Supervisor Agent & Orchestration (Week 5)

**Goal**: Implement NLU and workflow orchestration

### 4.1: NLU Parameter Extraction
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Reference**: `plans/supervisor-agent.plan.md`

**Tasks**:
- [ ] Create parameter extractor using LLM (gpt-4o)
- [ ] Implement currency pair parsing
- [ ] Implement amount extraction
- [ ] Implement risk/urgency/timeframe inference
- [ ] Add validation and clarification prompts
- [ ] Write extraction tests

**Files to Create**:
1. `src/agentic/nlu/extractor.py`
2. `src/agentic/nlu/prompts.py`
3. `src/agentic/nlu/validation.py`
4. `tests/agentic/nlu/test_extraction.py`

**Validation**:
```python
from src.agentic.nlu.extractor import extract_parameters
params = await extract_parameters("I need to convert 5000 USD to EUR today")
print(params.currency_pair, params.amount, params.urgency)
```

---

### 4.2: Conversation Manager
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [ ] Create conversation session manager
- [ ] Implement memory storage/retrieval
- [ ] Add clarification flow
- [ ] Write conversation tests

**Files to Create**:
1. `src/agentic/conversation/session.py`
2. `src/agentic/conversation/memory.py`
3. `tests/agentic/conversation/test_session.py`

**Validation**:
```python
from src.agentic.conversation import ConversationSession
session = ConversationSession(user_id="test")
session.add_turn("user", "Convert USD to EUR")
session.add_turn("assistant", "How much?")
print(session.get_history())
```

---

### 4.3: Response Generator
**Priority**: 🔴 Critical  
**Time Estimate**: 3-4 hours

**Tasks**:
- [ ] Create response formatter
- [ ] Generate friendly recommendation text
- [ ] Add visualization data preparation
- [ ] Write response tests

**Files to Create**:
1. `src/agentic/response/generator.py`
2. `src/agentic/response/formatter.py`
3. `tests/agentic/response/test_generator.py`

**Validation**:
```python
from src.agentic.response import generate_response
response = generate_response(recommendation, state)
print(response.message)
```

---

### 4.4: LangGraph Orchestration
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Tasks**:
- [ ] Build complete LangGraph workflow
- [ ] Add conditional routing logic
- [ ] Implement Layer 1 parallel dispatch
- [ ] Add error handling and retries
- [ ] Create Supervisor node
- [ ] Write end-to-end tests

**Files to Create**:
1. `src/agentic/graph.py` - Complete graph
2. `src/agentic/nodes/supervisor.py`
3. `src/agentic/routing.py` - Conditional edges
4. `tests/agentic/test_graph.py`

**Validation**:
```python
from src.agentic.graph import create_graph
graph = create_graph()
result = await graph.ainvoke({"user_query": "Convert 5000 USD to EUR"})
print(result["recommendation"])
```

---

## 📋 Phase 5: User Interfaces (Week 6)

**Goal**: Build user-facing interfaces

### 5.1: TUI (Terminal User Interface)
**Priority**: 🔴 Critical  
**Time Estimate**: 6-8 hours

**Reference**: `plans/tui-interface.plan.md` and `plans/impli/phase_5/phase-5-1.plan.md`

**Tasks**:
- [ ] Set up Rich library
- [ ] Create main TUI loop
- [ ] Implement multi-turn conversation
- [ ] Add visual feedback (spinners, progress)
- [ ] Display recommendation with tables
- [ ] Add parameter review/edit
- [ ] Write TUI tests

**Files to Create**:
1. `src/ui/tui/app.py`
2. `src/ui/tui/display.py`
3. `src/ui/tui/input_handler.py`
4. `src/ui/tui/renderer.py`
5. `src/ui/tui/config.py`
6. `tests/ui/test_tui.py`

**Validation**:
```bash
python -m src.ui.tui.app
# Test conversation flow
```

---

### 5.2: FastAPI Backend
**Priority**: 🟡 Important  
**Time Estimate**: 4-5 hours

**Reference**: `plans/web-ui.plan.md` and `plans/impli/phase_5/phase-5-2.plan.md`

**Tasks**:
- [ ] Create FastAPI app
- [ ] Add REST endpoints (POST /recommend, GET /session/{id})
- [ ] Add SSE endpoint for streaming
- [ ] Add health check endpoint
- [ ] Add CORS middleware
- [ ] Write API tests

**Files to Create**:
1. `src/ui/web/main.py`
2. `src/ui/web/routes/recommendations.py`
3. `src/ui/web/routes/sessions.py`
4. `src/ui/web/middleware.py`
5. `tests/ui/web/test_api.py`

**Validation**:
```bash
uvicorn src.ui.web.main:app --reload
curl -X POST http://localhost:8000/recommend -d '{"query": "Convert USD to EUR"}'
```

---

### 5.3: Web Frontend (Next.js + TypeScript)
**Priority**: 🟡 Important  
**Time Estimate**: 6-8 hours

**Reference**: `plans/impli/phase_5/phase-5-3.plan.md`

**Tasks**:
- [ ] Initialize Next.js 14+ project with TypeScript and TailwindCSS
- [ ] Install and configure shadcn/ui component library
- [ ] Create API client layer with axios
- [ ] Implement TypeScript types for API requests/responses
- [ ] Create React Query hooks for data fetching
- [ ] Implement Zustand store for chat state management
- [ ] Build chat UI components (ChatContainer, ChatMessage, ChatInput)
- [ ] Build analysis components (RecommendationCard, ConfidenceGauge, StagingPlan)
- [ ] Implement visualization components with Recharts
- [ ] Write component tests with React Testing Library
- [ ] Test full user flow end-to-end with FastAPI backend

**Files to Create**:
1. `frontend/app/page.tsx` - Main chat interface
2. `frontend/components/chat/ChatContainer.tsx`
3. `frontend/components/chat/ChatMessage.tsx`
4. `frontend/components/chat/ChatInput.tsx`
5. `frontend/components/analysis/RecommendationCard.tsx`
6. `frontend/components/analysis/ConfidenceGauge.tsx`
7. `frontend/components/visualizations/PriceChart.tsx`
8. `frontend/lib/api/client.ts` - Axios setup
9. `frontend/lib/api/conversation.ts` - API functions
10. `frontend/lib/types/api.ts` - TypeScript types
11. `frontend/lib/hooks/useConversation.ts` - React Query hook
12. `frontend/lib/store/chatStore.ts` - Zustand store

**Validation**:
```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
# Test conversation flow with backend running
```

---

## 📋 Phase 6: Testing & Polish (Week 7)

**Goal**: Comprehensive testing and bug fixes

### 6.1: Integration Testing
**Priority**: 🔴 Critical  
**Time Estimate**: 4-6 hours

**Tasks**:
- [ ] Write end-to-end test scenarios
- [ ] Test with mock providers
- [ ] Test error handling paths
- [ ] Test partial failures
- [ ] Test offline mode

**Files to Create**:
1. `tests/integration/test_full_workflow.py`
2. `tests/integration/test_error_scenarios.py`
3. `tests/integration/fixtures.py`

---

### 6.2: Performance Testing
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Measure end-to-end latency
- [ ] Test cache effectiveness
- [ ] Test concurrent requests
- [ ] Optimize slow paths

**Files to Create**:
1. `tests/performance/test_latency.py`
2. `tests/performance/test_cache.py`

---

### 6.3: Documentation
**Priority**: 🟡 Important  
**Time Estimate**: 4-5 hours

**Tasks**:
- [ ] Write API documentation (OpenAPI/Swagger)
- [ ] Write deployment guide
- [ ] Write user guide
- [ ] Add inline code documentation
- [ ] Create README with quick start

**Files to Create**:
1. `README.md`
2. `docs/API.md`
3. `docs/DEPLOYMENT.md`
4. `docs/USER_GUIDE.md`

---

## 📋 Phase 7: Deployment & Production (Week 8)

**Goal**: Prepare for production deployment

### 7.1: Docker Setup (Optional)
**Priority**: 🟡 Optional  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Test Docker build and run

**Files to Create**:
1. `Dockerfile`
2. `docker-compose.yml`
3. `.dockerignore`

**Note**: This is optional. The app works fine with just a virtual environment.
SQLite and in-memory cache don't need separate services.

**Validation**:
```bash
docker build -t currency-assistant .
docker run currency-assistant
```

---

### 7.2: CI/CD Pipeline
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Create GitHub Actions workflow
- [ ] Add linting (black, ruff)
- [ ] Add type checking (mypy)
- [ ] Add test running
- [ ] Add coverage reporting

**Files to Create**:
1. `.github/workflows/test.yml`
2. `.github/workflows/deploy.yml`
3. `pyproject.toml` (update with dev dependencies)

---

### 7.3: Monitoring & Logging
**Priority**: 🟡 Important  
**Time Estimate**: 3-4 hours

**Tasks**:
- [ ] Add Prometheus metrics
- [ ] Add health check dashboard
- [ ] Set up log aggregation
- [ ] Add alerting rules

**Files to Create**:
1. `src/monitoring/metrics.py`
2. `src/monitoring/health.py`
3. `prometheus.yml`

---

## 📋 Phase 8: ML Model Training (Ongoing)

**Goal**: Train and deploy initial ML models

### 8.1: Training Pipeline
**Priority**: 🟡 Important  
**Time Estimate**: 6-8 hours

**Tasks**:
- [ ] Create training script for LightGBM
- [ ] Collect historical data for major pairs
- [ ] Implement cross-validation
- [ ] Track experiments
- [ ] Save trained models to registry

**Files to Create**:
1. `scripts/train_model.py`
2. `scripts/evaluate_model.py`
3. `scripts/backtest_model.py`
4. `notebooks/model_exploration.ipynb`

**Validation**:
```bash
python scripts/train_model.py --pair USD/EUR --horizons 1,7,30
python scripts/evaluate_model.py --model-id usd_eur_v1
```

---

### 8.2: Model Deployment
**Priority**: 🟡 Important  
**Time Estimate**: 2-3 hours

**Tasks**:
- [ ] Train models for top 5 currency pairs
- [ ] Validate quality gates
- [ ] Deploy to model registry
- [ ] Test in production

**Validation**:
```python
from src.prediction import MLPredictor
predictor = MLPredictor()
pred = await predictor.predict(PredictionRequest(
    currency_pair="USD/EUR",
    horizons=[1, 7, 30]
))
```

---

## 📊 Progress Tracking

### Checklist Summary
- [ ] **Phase 0**: Foundation (14 tasks)
- [ ] **Phase 1**: Layer 1 Agents (22 tasks)
- [ ] **Phase 2**: Price Prediction (16 tasks)
- [ ] **Phase 3**: Decision Engine (12 tasks)
- [ ] **Phase 4**: Supervisor (16 tasks)
- [ ] **Phase 5**: UI (18 tasks)
- [ ] **Phase 6**: Testing (10 tasks)
- [ ] **Phase 7**: Deployment (10 tasks)
- [ ] **Phase 8**: ML Training (8 tasks)

**Total Tasks**: ~126

---

## 🎯 Quick Start (For Implementation)

### Day 1: Foundation
```bash
# 1. Create directory structure
mkdir -p src/{config,data_collection/{providers,market_data,market_intelligence},prediction,decision,agentic/{nodes,nlu,conversation,response},ui/{tui,web},utils,database}

# 2. Create config files
touch config.yaml .env.example

# 3. Create core utilities
touch src/config.py src/utils/{logging.py,errors.py,validation.py}

# 4. Set up database
alembic init alembic
touch src/database/{models.py,connection.py,session.py}

# 5. Create LangGraph state
touch src/agentic/{state.py,graph.py,routing.py}
```

### Day 2-3: Market Data Agent
```bash
# Follow Phase 1.1-1.3 tasks
```

### Day 4-5: Market Intelligence Agent
```bash
# Follow Phase 1.4-1.6 tasks
```

### Week 2: Price Prediction Agent
```bash
# Follow Phase 2 tasks
```

### Week 3: Decision Engine
```bash
# Follow Phase 3 tasks
```

### Week 4: Supervisor & Orchestration
```bash
# Follow Phase 4 tasks
```

### Week 6: UI
```bash
# Follow Phase 5 tasks (TUI, FastAPI, Next.js frontend)
```

---

## 🔑 Critical Dependencies

### Environment Variables (.env)
```bash
# LLM
COPILOT_ACCESS_TOKEN=your_token

# Market Data
EXCHANGE_RATE_HOST_API_KEY=your_key

# Market Intelligence
SERPER_API_KEY=your_key

# Database (SQLite - no credentials needed!)
DATABASE_PATH=data/currency_assistant.db

# Logging
LOG_LEVEL=INFO
```

### Python Dependencies (add to pyproject.toml)
```toml
[project.dependencies]
# Existing LLM deps...

# Agents & Orchestration
langgraph = ">=0.2.0"
langchain = ">=0.3.0"

# Data Providers
yfinance = ">=0.2.0"
httpx = ">=0.27.0"

# ML
lightgbm = ">=4.0.0"
scikit-learn = ">=1.3.0"
pandas = ">=2.0.0"
numpy = ">=1.24.0"
shap = ">=0.44.0"  # For model explainability visualizations

# Database
sqlalchemy = ">=2.0.0"
alembic = ">=1.12.0"
# SQLite is built into Python - no extra package needed!

# Caching - using in-memory Python cache (no Redis needed)

# API
fastapi = ">=0.110.0"
uvicorn = ">=0.27.0"
jinja2 = ">=3.1.0"  # Optional for API docs/templates

# TUI
rich = ">=13.0.0"

# Frontend (Next.js) - Run in separate directory with npm
# next, react, typescript, tailwindcss, @tanstack/react-query, zustand, axios, recharts

# Testing
pytest = ">=8.0.0"
pytest-asyncio = ">=0.23.0"
pytest-cov = ">=4.1.0"
```

---

## ✅ Definition of Done

For each phase, consider it complete when:

1. **Code Written**: All files listed in phase created
2. **Tests Pass**: Unit tests passing with >80% coverage
3. **Integration Works**: Component integrates with existing system
4. **Documentation**: Docstrings and comments added
5. **Validation**: Manual testing confirms expected behavior
6. **No Linter Errors**: Code passes black, ruff, mypy
7. **Committed**: Changes committed to version control

---

## 🚀 Getting Started

1. **Review this plan** - Understand the full scope
2. **Set up Phase 0** - Foundation is critical
3. **Work sequentially** - Each phase builds on previous
4. **Test continuously** - Don't accumulate technical debt
5. **Document as you go** - Future you will thank you

**Ready to start? Begin with Phase 0, Task 0.1!**
