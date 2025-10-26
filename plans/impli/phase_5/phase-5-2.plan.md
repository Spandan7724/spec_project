<!-- f67714c1-a54f-4e8d-9617-16955f212afc 8ee4663c-dc00-4f18-9c03-51070d5c74a5 -->
# Phase 5.2: FastAPI Backend Implementation

## Overview

Build a FastAPI REST API backend to expose Currency Assistant functionality via HTTP endpoints. The API will support conversation management, analysis execution, session tracking, and real-time updates via Server-Sent Events (SSE).

## Purpose

- Provide HTTP API for web frontend and external integrations
- Enable asynchronous analysis execution with progress tracking
- Support conversation sessions and state management
- Auto-generate OpenAPI/Swagger documentation
- Enable CORS for web frontend access

## Architecture

```
src/ui/web/
├── __init__.py
├── main.py                          # FastAPI app initialization
├── routes/
│   ├── __init__.py
│   ├── conversation.py              # Conversation endpoints
│   ├── analysis.py                  # Analysis execution endpoints
│   ├── visualization.py             # Data for charts (optional)
│   └── health.py                    # Health check endpoints
├── models/
│   ├── __init__.py
│   ├── requests.py                  # API request models (Pydantic)
│   └── responses.py                 # API response models (Pydantic)
├── middleware.py                    # CORS, logging, error handling
└── dependencies.py                  # Dependency injection

tests/ui/web/
├── test_api_conversation.py
├── test_api_analysis.py
└── test_api_integration.py
```

## File Descriptions

### 1. `src/ui/web/main.py`

**Purpose**: FastAPI application initialization and configuration.

**Components**:

- FastAPI app instance with metadata
- CORS middleware configuration
- Router registration
- Exception handlers
- Startup/shutdown events

**Endpoints Registered**:

- `/api/conversation/*` - Conversation routes
- `/api/analysis/*` - Analysis routes
- `/api/viz/*` - Visualization data (optional)
- `/health` - Health check

**Configuration**:

- Title: "Currency Assistant API"
- Version from config
- OpenAPI docs at `/docs`
- ReDoc at `/redoc`

### 2. `src/ui/web/routes/conversation.py`

**Purpose**: Handle multi-turn conversation and parameter extraction.

**Endpoints**:

**POST `/api/conversation/message`**

- Input: `ConversationInput` (user_input, session_id)
- Output: `ConversationOutput` (session_id, state, message, requires_input, parameters)
- Purpose: Process user message through NLU and conversation manager
- Behavior: Extract parameters, ask clarifications, validate inputs

**POST `/api/conversation/reset/{session_id}`**

- Input: session_id (path parameter)
- Output: Status confirmation
- Purpose: Clear conversation session and start fresh

**GET `/api/conversation/session/{session_id}`**

- Input: session_id (path parameter)
- Output: Session state and history
- Purpose: Retrieve conversation history and current state

**Integration**:

- Uses `ConversationManager` for session management
- Uses `extract_parameters` for NLU
- Stores sessions in memory (or database for persistence)

### 3. `src/ui/web/routes/analysis.py`

**Purpose**: Execute analysis and track progress.

**Endpoints**:

**POST `/api/analysis/start`**

- Input: `AnalysisRequest` (session_id, correlation_id, parameters)
- Output: `AnalysisStartResponse` (correlation_id, status)
- Purpose: Initiate background analysis task
- Behavior: Validate parameters, queue analysis, return correlation_id for tracking

**GET `/api/analysis/status/{correlation_id}`**

- Input: correlation_id (path parameter)
- Output: `AnalysisStatus` (status, progress, message)
- Purpose: Check analysis progress
- Behavior: Return current status (pending/processing/completed/error)

**GET `/api/analysis/result/{correlation_id}`**

- Input: correlation_id (path parameter)
- Output: `AnalysisResult` (recommendation, market_data, intelligence, prediction)
- Purpose: Retrieve completed analysis results
- Behavior: Return full recommendation with all agent outputs

**GET `/api/analysis/stream/{correlation_id}` (SSE)**

- Input: correlation_id (path parameter)
- Output: Server-Sent Events stream
- Purpose: Real-time progress updates during analysis
- Behavior: Stream progress messages as agents execute

**Integration**:

- Uses `AgentOrchestrator` to run agent workflow
- Uses BackgroundTasks or task queue for async execution
- Stores results temporarily (in-memory dict or Redis)

### 4. `src/ui/web/routes/visualization.py` (Optional)

**Purpose**: Provide data for frontend charts and visualizations.

**Endpoints**:

**GET `/api/viz/confidence/{correlation_id}`**

- Output: Confidence breakdown data (overall, components)
- Purpose: Data for confidence gauge chart

**GET `/api/viz/technical/{currency_pair}`**

- Output: Technical indicators (RSI, MACD, SMA, etc.)
- Purpose: Data for technical indicators dashboard

**GET `/api/viz/calendar/{currency}`**

- Output: Upcoming economic calendar events
- Purpose: Data for calendar timeline

**GET `/api/viz/history/{currency_pair}`**

- Output: Historical price data
- Purpose: Data for price chart

**Integration**:

- Queries market data agent for current indicators
- Queries intelligence agent for calendar events
- Queries database for historical data

### 5. `src/ui/web/routes/health.py`

**Purpose**: Health check and monitoring endpoints.

**Endpoints**:

**GET `/health`**

- Output: `HealthResponse` (status, components)
- Purpose: Overall system health
- Behavior: Check database, cache, config, external APIs

**GET `/health/ready`**

- Output: Ready status (200 or 503)
- Purpose: Kubernetes readiness probe

**GET `/health/live`**

- Output: Live status (200 or 503)
- Purpose: Kubernetes liveness probe

**Integration**:

- Uses `get_health_status()` from `src/health.py`
- Returns appropriate HTTP status codes

### 6. `src/ui/web/models/requests.py`

**Purpose**: Pydantic models for API requests.

**Models**:

**ConversationInput**:

- `user_input: str` - User message
- `session_id: Optional[str]` - Session identifier

**AnalysisRequest**:

- `session_id: str` - Session identifier
- `correlation_id: str` - Unique request ID
- `currency_pair: str` - e.g., "USD/EUR"
- `base_currency: str` - e.g., "USD"
- `quote_currency: str` - e.g., "EUR"
- `amount: float` - Conversion amount
- `risk_tolerance: str` - conservative/moderate/aggressive
- `urgency: str` - urgent/normal/flexible
- `timeframe: str` - immediate/1_day/1_week/1_month
- `timeframe_days: int` - Numeric timeframe

**Validation**:

- Currency codes: 3-letter uppercase
- Amount: positive float
- Enums for risk_tolerance, urgency, timeframe

### 7. `src/ui/web/models/responses.py`

**Purpose**: Pydantic models for API responses.

**Models**:

**ConversationOutput**:

- `session_id: str`
- `state: str` - COLLECTING/CONFIRMING/PROCESSING/COMPLETED/ERROR
- `message: str` - Assistant response
- `requires_input: bool` - Whether user input is needed
- `parameters: Optional[Dict]` - Extracted parameters

**AnalysisStartResponse**:

- `correlation_id: str`
- `status: str` - "pending"

**AnalysisStatus**:

- `status: str` - pending/processing/completed/error
- `progress: int` - 0-100
- `message: str` - Current progress message

**AnalysisResult**:

- `status: str` - "completed"
- `recommendation: Dict` - Decision engine output
- `market_snapshot: Dict` - Market data
- `intelligence_report: Dict` - Market intelligence
- `price_forecast: Dict` - Predictions
- `processing_time_ms: int` - Total execution time

**HealthResponse**:

- `status: str` - healthy/degraded/unhealthy
- `components: Dict[str, Dict]` - Component health details

### 8. `src/ui/web/middleware.py`

**Purpose**: Middleware for CORS, logging, and error handling.

**Middleware**:

**CORSMiddleware**:

- Allow origins: configurable (default: localhost)
- Allow methods: GET, POST, PUT, DELETE
- Allow headers: Content-Type, Authorization

**LoggingMiddleware**:

- Log all requests with correlation ID
- Log response times
- Log errors with stack traces

**ErrorHandlerMiddleware**:

- Catch all exceptions
- Return structured error responses
- Log errors appropriately

### 9. `src/ui/web/dependencies.py`

**Purpose**: Dependency injection for routes.

**Dependencies**:

**get_conversation_manager()**:

- Returns ConversationManager instance
- Singleton or per-request

**get_orchestrator()**:

- Returns AgentOrchestrator instance
- Singleton or per-request

**get_db_session()**:

- Returns database session
- Handles cleanup

**verify_session(session_id: str)**:

- Validates session exists
- Returns session or raises 404

## Key Features

### 1. Asynchronous Analysis

- Background task execution
- Non-blocking API responses
- Progress tracking via correlation_id

### 2. Session Management

- In-memory session storage (or database)
- Session timeout handling
- Session history retrieval

### 3. Error Handling

- Structured error responses
- HTTP status codes (400, 404, 500, 503)
- Error logging with correlation IDs

### 4. CORS Support

- Configurable origins
- Support for web frontend on different port
- Pre-flight request handling

### 5. OpenAPI Documentation

- Auto-generated from Pydantic models
- Interactive docs at `/docs`
- ReDoc at `/redoc`

## Implementation Steps

1. **Create web directory structure**
   ```bash
   mkdir -p src/ui/web/{routes,models} tests/ui/web
   ```

2. **Implement request/response models** (`models/requests.py`, `models/responses.py`)

3. **Implement middleware** (`middleware.py`)

4. **Implement health routes** (`routes/health.py`)

5. **Implement conversation routes** (`routes/conversation.py`)

6. **Implement analysis routes** (`routes/analysis.py`)

7. **Implement visualization routes** (`routes/visualization.py`) - Optional

8. **Implement main FastAPI app** (`main.py`)

9. **Write API tests** (test all endpoints)

10. **Test with curl/Postman/httpie**

## Dependencies

```toml
# Already in dependencies
fastapi = ">=0.110.0"
uvicorn = ">=0.27.0"
pydantic = ">=2.0.0"

# May need to add
python-multipart = ">=0.0.6"  # For form data
sse-starlette = ">=1.8.0"     # For Server-Sent Events
```

## Running the API

### Development

```bash
# Run with auto-reload
uvicorn src.ui.web.main:app --reload --port 8000

# Access API
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health: http://localhost:8000/health
```

### Production

```bash
# With multiple workers
uvicorn src.ui.web.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended)
gunicorn src.ui.web.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Example API Usage

### 1. Start Conversation

```bash
curl -X POST http://localhost:8000/api/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"user_input": "I need to convert 5000 USD to EUR"}'

# Response
{
  "session_id": "abc-123",
  "state": "CONFIRMING",
  "message": "I understand you want to convert 5000 USD to EUR. What is your risk tolerance?",
  "requires_input": true,
  "parameters": {
    "currency_pair": "USD/EUR",
    "amount": 5000
  }
}
```

### 2. Start Analysis

```bash
curl -X POST http://localhost:8000/api/analysis/start \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "correlation_id": "req-456",
    "currency_pair": "USD/EUR",
    "base_currency": "USD",
    "quote_currency": "EUR",
    "amount": 5000,
    "risk_tolerance": "moderate",
    "urgency": "normal",
    "timeframe": "1_week",
    "timeframe_days": 7
  }'

# Response
{
  "correlation_id": "req-456",
  "status": "pending"
}
```

### 3. Check Status

```bash
curl http://localhost:8000/api/analysis/status/req-456

# Response
{
  "status": "processing",
  "progress": 50,
  "message": "Generating price predictions..."
}
```

### 4. Get Result

```bash
curl http://localhost:8000/api/analysis/result/req-456

# Response
{
  "status": "completed",
  "recommendation": {
    "action": "WAIT",
    "confidence": 0.72,
    "timeline": "wait_24_48_hours",
    "rationale": ["Prediction shows 0.5% improvement..."]
  },
  "market_snapshot": {...},
  "intelligence_report": {...},
  "price_forecast": {...},
  "processing_time_ms": 4523
}
```

### 5. Health Check

```bash
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "components": {
    "database": {"status": "healthy"},
    "cache": {"status": "healthy"},
    "config": {"status": "healthy"}
  }
}
```

## Testing Strategy

- **Unit tests**: Test each route independently with mocked dependencies
- **Integration tests**: Test complete API flows with real agent execution
- **Load tests**: Test concurrent requests and performance

## Success Criteria

- All endpoints return correct response models
- OpenAPI docs generated correctly
- CORS works with frontend
- Async analysis executes in background
- Progress tracking works correctly
- Error handling returns appropriate status codes
- Health checks report component status
- API can handle concurrent requests
- Response times meet targets (<5s for analysis)

## Integration Points

- `src.agentic.conversation.session.ConversationSession`: Session management
- `src.agentic.nlu.extractor.extract_parameters`: Parameter extraction
- `src.agentic.graph.create_graph`: LangGraph workflow execution
- `src.agentic.response.generator.generate_response`: Response formatting
- `src.health.get_health_status`: Health monitoring

## Security Considerations

- Add rate limiting for production
- Add authentication/authorization if needed
- Validate all inputs with Pydantic
- Sanitize error messages (don't leak internal details)
- Configure CORS origins properly for production

## Notes

- API is stateless (session data stored separately)
- Use correlation_id for request tracking and logging
- Consider Redis for production session storage
- Consider message queue (Celery) for production task handling
- SSE endpoint is optional but useful for real-time updates
- Visualization endpoints can be added in Phase 5.3 when needed

### To-dos

- [ ] Create FastAPI application in src/ui/web/main.py with CORS middleware
- [ ] Implement conversation routes in src/ui/web/routes/conversation.py (message, reset, session)
- [ ] Implement analysis routes in src/ui/web/routes/analysis.py (start, status, result, stream SSE)
- [ ] Create Pydantic models for requests and responses in src/ui/web/models/
- [ ] Implement health check routes in src/ui/web/routes/health.py
- [ ] Create middleware for CORS, logging, and error handling
- [ ] Write comprehensive API tests for all endpoints
- [ ] Test API with curl/Postman for complete workflow