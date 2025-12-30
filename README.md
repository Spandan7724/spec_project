# Currency Assistant

An AI assistant that helps people decide when and how to convert currencies, using live market data, news signals, and ML forecasts to produce a clear recommendation with supporting evidence.

It combines live market data, news and economic signals, ML forecasts, and a decision engine to recommend when to convert and why. Ships as a FastAPI backend, a Vite/React frontend, and a Rich-powered TUI.

## What it does

- Parses user intent (pair, amount, risk, urgency, timeframe)
- Pulls quotes/indicators plus news + calendar signals
- Runs LightGBM (daily) and optional LSTM (intraday) forecasts
- Produces an action, confidence, and evidence for UI charts

## Stack

- Backend: FastAPI, SQLAlchemy, httpx, pandas/numpy, LightGBM, PyTorch, yfinance
- Frontend: React, Vite, TypeScript, Tailwind, Recharts
- LLM: pluggable providers with failover (Copilot/OpenAI/Claude)
- Storage: SQLite + local model registry under `data/`

## Quick start

Prereqs: Python 3.12+, Node 20+, npm. Optional: `uv`.

1) Configure environment
```
cp .env.example .env
# Add API keys you plan to use
```

2) Install Python deps
```
uv venv && source .venv/bin/activate
uv pip install -e .
```
Or:
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3) Install frontend deps
```
cd frontend
npm install
cd ..
```

4) Run backend + frontend
```
python dev_backend.py
```
```
cd frontend
npm run dev
```

Open:
- Web: http://127.0.0.1:5173
- TUI: `currency-assistant-tui`

## Configuration

- `.env` for API keys and toggles; `OFFLINE_DEMO=true` enables safe fallbacks.
- `config.yaml` for app/agents/prediction/LLM settings.
- Override config path with `CURRENCY_ASSISTANT_CONFIG`.

Key env vars:
- `COPILOT_ACCESS_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `SERPER_API_KEY` (news/calendar), `EXCHANGE_RATE_HOST_API_KEY` (optional)
- `VITE_API_BASE_URL` (frontend API base override)

## Commands

- Backend: `python dev_backend.py` or `uvicorn backend.main:app --reload --port 8000`
- Frontend: `cd frontend && npm run dev`
- TUI: `currency-assistant-tui`
- Train models (CLI):
  - `currency-assistant train-model --pair USD/EUR --horizons 1 --horizons 7`
  - `currency-assistant train-lstm --pair USD/EUR --horizon-hours 1 --horizon-hours 4`
- API docs: `docs/API.md`

## Tests

```
pytest -q
```
```
bash scripts/api_smoke_test.sh
```

## Repo layout

```
backend/     FastAPI app and routes
frontend/    Vite + React UI
src/         Agents, data collection, prediction, decision, TUI
data/        SQLite DB + model registry
config.yaml  Central configuration
docs/        API + design/phase notes
```

   
