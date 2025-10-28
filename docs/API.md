
# Currency Assistant API (FastAPI)

This document describes the backend HTTP API exposed by the FastAPI service in `backend/` to support the TUI and Web UI.

- Base URL (dev): `http://localhost:8000`
- Root endpoint: `GET /` → `{ "status": "ok", "service": "currency-assistant-api" }`
- CORS: permissive for development (`*`). Tighten for production.
- Auth: none (development). Add auth in production as needed.

## Endpoints Overview

- Conversation
  - `POST /api/conversation/message` – process a user message
  - `GET /api/conversation/session/{session_id}` – fetch session state/history
  - `POST /api/conversation/reset/{session_id}` – reset a session
- Analysis
  - `POST /api/analysis/start` – start analysis (asynchronous)
  - `GET /api/analysis/status/{correlation_id}` – check progress
  - `GET /api/analysis/result/{correlation_id}` – fetch final recommendation
  - `GET /api/analysis/stream/{correlation_id}` – Server‑Sent Events (SSE) progress stream
- Visualization (derived views)
  - `GET /api/viz/confidence/{correlation_id}` – overall + component confidences
- Health
  - `GET /health` – service health and component status

---

## Conversation

### POST /api/conversation/message
Process a single user turn through the Supervisor’s ConversationManager.

- Request body (JSON):
```
{
  "user_input": "string",
  "session_id": "string | null"
}
```
- Response (JSON):
```
{
  "session_id": "string",
  "state": "initial | collecting_currency_pair | collecting_amount | collecting_risk | collecting_urgency | collecting_timeframe | confirming | processing | completed | error",
  "message": "string",          // assistant’s reply
  "requires_input": true|false,  // true → UI should prompt user for another message
  "parameters": {                // ExtractedParameters (may be partial)
    "currency_pair": "USD/EUR",
    "base_currency": "USD",
    "quote_currency": "EUR",
    "amount": 5000,
    "risk_tolerance": "moderate",
    "urgency": "normal",
    "timeframe": "1_week",        // or null when using flexible/canonical fields
    "timeframe_days": 7,           // may be set by NLU
    "timeframe_mode": "duration|deadline|immediate",
    "deadline_utc": "2025-11-15T00:00:00Z",
    "window_days": {"start":3,"end":5},
    "time_unit": "hours|days",
    "timeframe_hours": 12
  }
}
```
- Curl:
```
curl -sS -X POST "$BASE/api/conversation/message"   -H 'Content-Type: application/json'   -d '{"user_input":"I need to convert 1000 USD to EUR","session_id":"test-session-123"}'
```

### GET /api/conversation/session/{session_id}
Fetch session state and history (useful for restoring UI state).
- Response (JSON):
```
{
  "session_id": "string",
  "state": "string",
  "parameters": { ... ExtractedParameters ... },
  "history": [ {"role":"user|assistant","content":"...","timestamp":"..."}, ... ]
}
```

### POST /api/conversation/reset/{session_id}
Reset the session. Returns `{ status: "reset", session_id }`.

---

## Analysis

Analysis is asynchronous. Clients provide a `correlation_id` (UUID recommended) and use it for status/result/SSE.

### POST /api/analysis/start
Start the analysis background job.

- Request body (JSON): legacy + flexible timeframe supported:
```
{
  "session_id": "string",
  "correlation_id": "string",    // unique per request

  // core params
  "currency_pair": "USD/EUR",     // optional if base/quote supplied
  "base_currency": "USD",
  "quote_currency": "EUR",
  "amount": 5000,
  "risk_tolerance": "conservative|moderate|aggressive",
  "urgency": "urgent|normal|flexible",

  // EITHER legacy categorical timeframe ...
  "timeframe": "immediate|1_day|1_week|1_month",
  "timeframe_days": 7,             // optional legacy numeric mapping

  // OR flexible free‑text timeframe ...
  "timeframe_text": "in 10 days | by 2025-11-15 | 3-5 days | in 12 hours",

  // OR canonical fields (if you already parsed client‑side)
  "timeframe_mode": "duration|deadline|immediate",
  "deadline_utc": "2025-11-15T00:00:00Z",
  "window_days": {"start":3,"end":5},
  "time_unit": "hours|days",
  "timeframe_hours": 12,
  "timeframe_days": 10
}
```
- Behavior:
  - If canonical fields present → they are trusted.
  - Else if `timeframe_text` present → normalized via Supervisor NLU (no LLM network usage).
  - Else → legacy categorical timeframe is used (and days derived when needed).
- Response:
```
{ "correlation_id": "string", "status": "pending" }
```
- Curl:
```
# Legacy categorical
curl -sS -X POST "$BASE/api/analysis/start"   -H 'Content-Type: application/json'   -d '{
    "session_id":"'$SID'",
    "correlation_id":"'$CORR'",
    "currency_pair":"USD/EUR",
    "base_currency":"USD",
    "quote_currency":"EUR",
    "amount":5000,
    "risk_tolerance":"moderate",
    "urgency":"normal",
    "timeframe":"1_week"
  }'

# Flexible free‑text
curl -sS -X POST "$BASE/api/analysis/start"   -H 'Content-Type: application/json'   -d '{
    "session_id":"'$SID'",
    "correlation_id":"'$CORR'",
    "currency_pair":"USD/EUR",
    "base_currency":"USD",
    "quote_currency":"EUR",
    "amount":5000,
    "risk_tolerance":"moderate",
    "urgency":"normal",
    "timeframe_text":"in 10 days"
  }'
```

### GET /api/analysis/status/{correlation_id}
Check progress.
- Response (JSON):
```
{ "status": "pending|processing|completed|error", "progress": 0-100, "message": "string" }
```

### GET /api/analysis/result/{correlation_id}
Fetch the final recommendation (available when `status` is `completed`).
- Response (JSON): recommendation envelope produced by the orchestrator, e.g.:
```
{
  "status": "success|error",
  "action": "convert_now|staged_conversion|wait",
  "confidence": 0.72,
  "timeline": "Execute in 3 tranches over 7 days",
  "rationale": [ ... ],
  "warnings": [ ... ],
  "metadata": { "correlation_id": "...", "timestamp": "..." },
  "staged_plan": { ... },
  "expected_outcome": { ... },
  "risk_summary": { ... },
  "cost_estimate": { ... },
  "evidence": { "market": { ... }, "news": [ ... ], "calendar": [ ... ], "intelligence": { ... }, "model": { ... }, "prediction": { ... }, "predictions_all": { ... } },
  "meta": { "prediction_horizon_days": 7, "used_prediction_horizon_key": "7" },
  "utility_scores": { ... },
  "component_confidences": { ... }
}
```

### GET /api/analysis/stream/{correlation_id}
Server‑Sent Events (SSE) progress stream. Emits `status` events until `completed|error`.
- Media type: `text/event-stream`
- Event format:
```
event: status
data: {"status":"processing","progress":25,"message":"Analyzing market conditions..."}

```
- Curl:
```
curl -N "$BASE/api/analysis/stream/$CORR"
```

---

## Visualization

### GET /api/viz/confidence/{correlation_id}
Returns overall recommendation confidence and component confidences for visualization.
- Response (JSON):
```
{ "overall": 0.72, "components": { "market": 0.80, "intelligence": 0.65, "prediction": 0.70 } }
```

---

## Health

### GET /health
- Response (JSON):
```
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2025-10-28T10:12:08.990308",
  "components": {
    "database": {"status":"healthy","message":"..."},
    "cache": {"status":"healthy","message":"..."},
    "config": {"status":"healthy","message":"..."}
  }
}
```

---

## Error Handling

- 400 Bad Request – invalid request or analysis not completed
- 404 Not Found – unknown `session_id`/`correlation_id`
- 500 Internal Server Error – unexpected exception; message included in `detail`

---

## Notes for Frontend Integration

- Correlation IDs
  - Generate on the client (UUID recommended) and pass to `/api/analysis/start`.
  - Use the same ID to poll status, fetch results, or stream SSE.

- Flexible timeframe input
  - Prefer sending `timeframe_text` for free‑text (e.g., “in 10 days”).
  - If you already parsed on the client, send canonical fields (`timeframe_days`, `deadline_utc`, `window_days`, `timeframe_mode`, `time_unit`, `timeframe_hours`).
  - Legacy categorical (`timeframe`) still works.

- SSE vs. polling
  - Use `EventSource` on `/api/analysis/stream/{correlation_id}` for real‑time progress.
  - Fallback to polling `/api/analysis/status/{correlation_id}` every 500–1000ms.

- CORS
  - Development: `*`. Configure allowed origins in production.

- Offline/LLM behavior
  - `OFFLINE_DEMO=true` allows graceful degradation. NLU normalization of timeframe_text does not require LLM network access.

---

## Quick Smoke (curl)

```
export BASE=http://localhost:8000
export SID=$(uuidgen)
export CORR=$(uuidgen)

# Start analysis
curl -sS -X POST "$BASE/api/analysis/start"   -H 'Content-Type: application/json'   -d '{
    "session_id":"'$SID'",
    "correlation_id":"'$CORR'",
    "currency_pair":"USD/EUR",
    "base_currency":"USD",
    "quote_currency":"EUR",
    "amount":5000,
    "risk_tolerance":"moderate",
    "urgency":"normal",
    "timeframe_text":"in 10 days"
  }'

# Stream progress
curl -N "$BASE/api/analysis/stream/$CORR"

# Or poll status
curl -sS "$BASE/api/analysis/status/$CORR"

# Get result
curl -sS "$BASE/api/analysis/result/$CORR"
```
