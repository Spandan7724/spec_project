# Currency Assistant

AI-powered assistant that guides when and how to convert one currency into another. It combines real-time market data, news and economic events intelligence, ML price predictions, and a decision engine into clear, actionable recommendations, exposed over a FastAPI backend with a React (Vite + TS) frontend and a Rich-powered terminal UI.

- Backend: FastAPI, SQLAlchemy, httpx, pandas/numpy, LightGBM, PyTorch (LSTM), yfinance
- Frontend: React 18, Vite, TypeScript, Tailwind, shadcn/ui, Recharts
- LLM layer: pluggable providers with automatic failover (GitHub Copilot, OpenAI, Claude)
- Storage: SQLite (default) + local model registry under `data/`


## Table of Contents

- Overview
- Architecture
- Repository Layout
- Quick Start
- Environment Variables
- Configuration (`config.yaml`)
- Running (Backend, Frontend, TUI)
- API Overview
- Models and Training
- Testing
- Troubleshooting
- Roadmap and Docs


## Overview

Currency Assistant helps decide: convert now, stage over time, or wait. It:
- Parses user intent and parameters (pair, amount, risk, urgency, timeframe)
- Collects market snapshot (live quotes, indicators, regime)
- Gathers market intelligence (news sentiment, economic calendar; LLM-assisted)
- Produces ML forecasts (daily + optional intraday) with explanations
- Aggregates into a recommendation with confidence, risk, cost, and timeline
- Streams progress and returns evidence suitable for visualization


## Architecture

High-level flow (Agent Orchestrator):
1) Conversation + NLU (Supervisor)
   - Extracts `currency_pair`, `amount`, `risk_tolerance`, `urgency`, `timeframe` (incl. flexible text like “in 10 days”)
2) Layer 1 (parallel)
   - Market Data node: live mid/bid/ask, indicators (SMA/EMA/RSI/MACD/BB/ATR), regime
   - Market Intelligence node: news classification + narrative, economic calendar extraction, policy bias
3) Layer 2
   - Prediction node: LightGBM (daily) + LSTM (intraday) via a simple hybrid router; optional SHAP explanations
4) Layer 3
   - Decision node: risk-aware utility, staging plans, cost estimate; outputs action, confidence, rationale
5) Results + Evidence
   - Routes format complete recommendation + supporting evidence for UI visualizations

Key modules:
- Orchestrator: `src/supervisor/agent_orchestrator.py`
- Nodes: `src/agentic/nodes/{market_data,market_intelligence,prediction,decision}.py`
- Decision engine: `src/decision/*`
- Prediction system: `src/prediction/*` with registry `data/models/*`
- Intelligence: `src/data_collection/market_intelligence/*`
- Market data: `src/data_collection/market_data/*`, providers in `src/data_collection/providers/*`
- LLM manager + providers: `src/llm/*`
- Backend API: `backend/*` (routes + DB layer)
- Frontend: `frontend/*`
- TUI: `src/ui/tui/*`


## Repository Layout

```
backend/                # FastAPI app and routes
  main.py               # Application factory + router wiring
  routes/               # conversation, analysis, viz, models, health
  database/             # API-layer models + repositories (uses src.database)
frontend/               # Vite + React + TS web UI
src/
  agentic/              # Nodes, graph, agent state
  data_collection/      # Market data + intelligence collection
  decision/             # Decision engine
  llm/                  # LLM config, manager, providers
  prediction/           # ML predictors, training, registry
  database/             # SQLAlchemy base + connection
  ui/tui/               # Terminal UI
  utils/, cache.py      # Logging, decorators, errors, helpers
scripts/                # Helpers (e.g., API smoke test)
tests/                  # Unit + integration tests
data/                   # SQLite DB + local model registry storage
config.yaml             # Central app + agents + prediction + LLM config
.env.example            # Template for environment variables
```


## Quick Start

Prerequisites:
- Python 3.12+
- Node.js 20+ and npm
- Optional but recommended: `uv` (https://docs.astral.sh/uv/) for fast Python installs

1) Clone and enter the project
```
git clone <this-repo>
cd currency_assistant
```

2) Configure environment
```
cp .env.example .env
# Edit `.env` and add required API keys (see Environment Variables)
```

3) Install Python dependencies
- Using uv
```
uv venv
source .venv/bin/activate
uv pip install -e .
```
- Or using pip
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

4) Install frontend deps
```
cd frontend
npm install
cd ..
```

5) Run backend (dev)
```
python dev_backend.py
# FastAPI on http://127.0.0.1:8000
```

6) Run frontend (dev)
```
cd frontend
npm run dev
# Vite on http://127.0.0.1:5173 (proxies /api → 127.0.0.1:8000)
```

7) Open the app
- Web: http://127.0.0.1:5173
- TUI: `currency-assistant-tui`


## Environment Variables

Set these in `.env` (copy from `.env.example`). Only enable what you plan to use.

LLM providers (one or more):
- `COPILOT_ACCESS_TOKEN` — GitHub Copilot (used by provider names starting with `copilot`)  
- `OPENAI_API_KEY` — OpenAI
- `ANTHROPIC_API_KEY` — Anthropic Claude

Market data and intelligence:
- `EXCHANGE_RATE_HOST_API_KEY` — optional; ExchangeRate.host live endpoint
- `SERPER_API_KEY` — required for news/calendar scraping via Serper
- `SERPER_ENABLE_WHITELIST` — true|false; restricts Serper news domains (see `config.yaml` whitelist)

App settings:
- `LOG_LEVEL` — INFO|DEBUG…
- `DEBUG` — true|false
- `OFFLINE_DEMO` — true|false. When true, nodes return safe fallbacks on failure; useful for demos without keys

Frontend:
- `VITE_API_BASE_URL` — override API base (defaults to relative `/api` in dev and Vite proxy to 127.0.0.1:8000)


## Configuration (`config.yaml`)

`config.yaml` centralizes application, agents, prediction, and LLM configuration.

Highlights:
- `app`, `database`, `cache` — basic runtime + SQLite path (`data/currency_assistant.db`)
- `api` — upstreams (ExchangeRate.host, Serper) and news domain whitelist
- `agents.market_data` — providers and short cache TTLs for live quotes
- `agents.market_intelligence` — knobs for article/event limits and concurrency
- `prediction` — horizons, features, quality gates, explanations, and local model registry paths
- `decision` — risk profiles, thresholds, staging patterns, costs, heuristics toggle
- `llm` — multiple provider variants (e.g., `copilot_main`, `copilot_fast`, `openai_main`, `claude_main`) and failover

Notes:
- Any provider name that starts with `copilot` uses `COPILOT_ACCESS_TOKEN`; `openai_*` uses `OPENAI_API_KEY`; `claude_*` uses `ANTHROPIC_API_KEY`.
- You can override which YAML file to load via `CURRENCY_ASSISTANT_CONFIG`.


## Running

Backend (dev):
- `python dev_backend.py` (hot reload `backend/` and `src/`)
- Or `uvicorn backend.main:app --reload --port 8000`

Health checks:
- `GET http://127.0.0.1:8000/health` → database, cache, config status

Frontend (dev):
- `cd frontend && npm run dev` → http://127.0.0.1:5173 (proxies `/api` to backend)
- Optional: set `VITE_API_BASE_URL=http://127.0.0.1:8000` to bypass proxy

Terminal UI:
- `currency-assistant-tui` launches a guided terminal experience end-to-end

Smoke test:
- `bash scripts/api_smoke_test.sh` (set `BASE` if not default)

Data + storage:
- SQLite DB created at `data/currency_assistant.db` on first run
- Model registry metadata at `data/models/prediction_registry.json`, binaries in `data/models/prediction/`


## API Overview

Base URL (dev): `http://127.0.0.1:8000`

Conversation (`backend/routes/conversation.py`):
- `POST /api/conversation/message` — turn-based supervisor; returns `state`, `message`, `parameters`
- `POST /api/conversation/message/stream` — SSE stream of growing message for immediate feedback
- `GET /api/conversation/session/{session_id}` — session state + history (+ result summary if available)
- `POST /api/conversation/reset/{session_id}` — clear session
- `GET /api/conversation/sessions/active` — list active sessions

Analysis (`backend/routes/analysis.py`):
- `POST /api/analysis/start` — begin background analysis (supports legacy, free-text, or canonical timeframe)
- `GET /api/analysis/status/{correlation_id}` — progress
- `GET /api/analysis/result/{correlation_id}` — complete recommendation + evidence
- `GET /api/analysis/stream/{correlation_id}` — SSE status events
- `GET /api/analysis/list` — list analyses (filter/paginate)
- `DELETE /api/analysis/{correlation_id}` — delete one
- `POST /api/analysis/cleanup` — delete expired

Visualization (`backend/routes/visualization.py`):
- `GET /api/viz/confidence/{correlation_id}` — overall + component confidences
- `GET /api/viz/risk-breakdown/{correlation_id}` — risk details
- `GET /api/viz/cost-breakdown/{correlation_id}` — cost details
- `GET /api/viz/timeline-data/{correlation_id}` — execution timeline points
- `GET /api/viz/prediction-chart/{correlation_id}` — prediction means + quantiles
- `GET /api/viz/prediction-quantiles/{correlation_id}` — fan-chart structure
- `GET /api/viz/evidence/{correlation_id}` — news, calendar, market summary
- `GET /api/viz/historical-prices/{correlation_id}` — OHLC history + indicators
- `GET /api/viz/technical-indicators/{correlation_id}` — RSI/MACD/ATR/vol
- `GET /api/viz/market-regime/{correlation_id}` — regime timeline
- `GET /api/viz/sentiment-timeline/{correlation_id}` — sentiment snapshot + synthetic series
- `GET /api/viz/events-timeline/{correlation_id}` — normalized event list for timeline
- `GET /api/viz/shap-explanations/{correlation_id}` — SHAP features and optional waterfall

Models (`backend/routes/models.py`):
- `POST /api/models/train` — train LightGBM or LSTM in background; returns job_id
- `GET /api/models/train/status/{job_id}` — job progress
- `GET /api/models/` — list registered models (filters supported)
- `GET /api/models/{model_id}` — model metadata
- `DELETE /api/models/{model_id}` — delete a model
- `GET /api/models/registry/info` — registry stats

Health:
- `GET /health` — overall system health

Full reference: `docs/API.md`


## Models and Training

Local model registry is file-based under `data/models/` and is used by the prediction node.

Two paths to train:

1) REST API
```
POST /api/models/train
{
  "currency_pair": "USD/EUR",
  "model_type": "lightgbm" | "lstm",
  "horizons": [1,7,30],           # for lightgbm (days) or lstm (hours)
  "version": "1.0",
  # Optional GBM knobs
  "gbm_rounds": 120,
  "gbm_patience": 10,
  "gbm_learning_rate": 0.05,
  "gbm_num_leaves": 31,
  # Optional LSTM knobs
  "lstm_epochs": 5,
  "lstm_hidden_dim": 64,
  "lstm_seq_len": 64,
  "lstm_lr": 0.001
}
```
Then poll `GET /api/models/train/status/{job_id}` until `completed`.

2) CLI (Typer)
```
# LightGBM daily models
currency-assistant train-model --pair USD/EUR --horizons 1 --horizons 7 --horizons 30

# LSTM intraday models
currency-assistant train-lstm --pair USD/EUR --horizon-hours 1 --horizon-hours 4 --horizon-hours 24
```

Prediction routing:
- `prediction.predictor_backend` can be `lightgbm`, `lstm`, or `hybrid` (default) for serving
- The node also adapts horizons to the user timeframe and config (`include_intraday_for_1_day`)

Explanations:
- If enabled in config, LightGBM models can return top features and SHAP waterfall payloads for UI


## Testing

Install dev deps (already declared in `pyproject.toml`) and run:
```
pytest -q
```
Selectively:
```
pytest tests/unit -q
pytest tests/integration -q
```
Quick backend sanity:
```
bash scripts/api_smoke_test.sh
```


## Troubleshooting

- Missing API keys: LLM providers and Serper require keys. Set in `.env`. You can set `OFFLINE_DEMO=true` to allow fallback behavior for demos.
- SSE not streaming through proxies: ensure reverse proxies disable buffering for `text/event-stream` (backend sets `X-Accel-Buffering: no`).
- CORS in dev: frontend uses Vite proxy for `/api` to avoid CORS; otherwise configure `VITE_API_BASE_URL` and backend CORS.
- SQLite path: backend creates `data/` automatically; ensure the process user can write to it.
- yfinance rate limiting: historical downloads may throttle; retry or reduce calls.
- Domain whitelist: set `SERPER_ENABLE_WHITELIST=false` to accept a broader set of news domains.


## Roadmap and Docs

- Implementation plans: `plans/*.md` (architecture, phased plans, quick start for LLM selection)
- API reference: `docs/API.md`
- Phase write-ups: `docs/PHASE*.md`

Contributions: keep changes focused and consistent with existing style; see tests and plans for intent. If you want a Dockerfile or deployment guide, open an issue or request and we can add it.

