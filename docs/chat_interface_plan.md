# Conversational Currency Assistant Plan

## Goal
Extend the current agentic workflow into a full conversational assistant so users can ask follow-up questions, request clarifications, and receive updated advice in a multi-turn chat. The agent should reuse structured insights (market analysis, ML forecasts, economic events, risk assessment) while leveraging the LLM for natural language explanations.

## High-Level Architecture

```
Client (CLI/Web) ⇄ Chat API ⇄ Chat Manager
                          │
                          ├── Agentic Workflow (LangGraph)
                          │      ├ Market / Economic / Risk nodes
                          │      └ Decision Coordinator (LLM backed)
                          │
                          ├── Stored Session State (structured results)
                          └── Knowledge/LMM fallback
```

### Components
- **Chat API**: fast path for receiving user messages (FastAPI, CLI, or both).
- **Chat Manager**: orchestrates sessions, intent detection, state retrieval, and calls to the agentic workflow or LLM.
- **Session Store**: retains the latest `AgentGraphState`, ML forecasts, risk scenarios, and prior conversation history.
- **Agentic Workflow**: reused as-is for new conversion requests or refreshes; returns structured data for the manager to summarise.
- **LLM Layer**: generates conversational responses using up-to-date structured insights, or handles non-conversion queries.

## Conversation Lifecycle
1. **Receive message**: Client sends `{session_id, message}`.
2. **Parse intent**:
   - `request_conversion`: user provides pair/amount/timeframe.
   - `ask_followup`: questions about recent recommendation ("why wait?", "what changed?").
   - `ask_update`: user wants a refreshed analysis or rerun.
   - `general_query`: generic FX/economy question requiring LLM knowledge.
3. **Retrieve session state** (if any) to reuse context.
4. **Execute**:
   - `request_conversion` → invoke `run_agentic_workflow` with parsed parameters (consider reusing cached results if recent).
   - `ask_followup` → use stored structured outputs to answer without re-running heavy computations.
   - `ask_update` → call workflow with `force_refresh=True` or re-fetch data.
   - `general_query` → route to LLM with structured context and/or external knowledge base.
5. **Store updated state** (new `AgentGraphState`, timeline, ML predictions).
6. **Respond via LLM**: craft a response that references structured metrics (market summary, risk numbers, ML bias) and mention actions already recommended.

## State Management
- Store in-memory (dict keyed by session ID) for prototypes; consider Redis or a database for production.
- Fields to retain per session:
  - `agent_state`: last `AgentGraphState` returned.
  - `recommendation`: action, rationale, timeline.
  - `ml_summary`: forecasts per horizon, model confidence.
  - `risk_snapshot`: scenarios, volatility metrics.
  - `market_snapshot`: current rate, technical regime, bias.
  - `event_list`: high-impact events with timestamps.
  - `conversation_history`: list of user/assistant messages for LLM context.
  - `last_updated`: timestamp for refresh logic.

## Chat Manager Responsibilities
- Validate user inputs and map to structured fields (pair normalization, numeric amounts, timeframe).
- Gracefully handle errors: unavailable models (`train first` message), provider timeouts, or missing data.
- Manage caching: reuse previous agentic results if the request hasn’t changed and data is still fresh.
- Provide concise explanations: e.g., highlight ML forecast direction for the requested horizon.
- Support commands (optional): `/help`, `/history`, `/refresh`.

## LLM Prompt Design
- **System prompt**: set the assistant role and expectations (grounded responses, mention confidence, highlight next steps).
- **Context**: supply structured JSON from the latest `AgentGraphState`, including horizon-specific forecasts, economic events filtered to the timeframe, and risk metrics.
- **User message**: original query plus conversation history (truncated for token limits).
- **Output requirements**: instruct the LLM to respond in plain language with short sections (Action, Reasoning, Events, ML Forecast, Next Steps).

## Intent Detection
- Basic rule-based for MVP using simple keyword matching (`"convert"`, `"why"`, `"update"`).
- For robustness, add a lightweight classifier (on top of the LLM or a smaller model) to detect intent and entities (currency, amount, timeframe, risk).

## Integration with Agentic Workflow
- The chat manager should call `run_agentic_workflow` only when necessary:
  - On new or updated conversion requests.
  - On refresh commands or when stale data exceeds freshness thresholds.
- Ensure the workflow’s output (structured state) is serializable and stored for follow-up use.
- Add a summarization helper to convert structured risk/ML data into human-readable bullet points.

## Handling Missing Models / Auto-Train
- If the predictor reports missing models, respond with a clear message (“Model for USD/INR not trained yet”) and optionally queue a training job for offline processing.
- Avoid auto-training during chat to keep latency low.

## Refresh Logic
- Use timestamps to decide when data is stale (e.g., refresh if >10 minutes for live rates, >24 hours for economic calendar).
- Provide user controls (`/refresh`) to force a rerun.

## Session Persistence & Cleanup
- For initial versions, store sessions in-memory with an LRU eviction policy.
- Optionally persist to disk or use a datastore if chat histories need to survive restarts.

## CLI & Web Interfaces
- Extend the interactive CLI to behave like a chat: display assistant responses, keep history, allow commands.
- Add a FastAPI endpoint (`POST /chat`) for web/mobile clients; respond with JSON containing structured recommendation, reasoning, warnings, and raw data for UI rendering.
- Consider streaming responses if you enable LLM streaming.

## Testing Strategy
- Unit tests for intent parsing, state storage, and LLM prompt shaping.
- Integration tests that simulate multi-turn conversations and verify consistent agentic behaviour.
- Load tests to ensure the chat manager handles concurrent sessions without exhausting resources.

## Future Enhancements
- **User profiles**: persist user preferences (risk tolerance, recurring transfers) to tailor responses automatically.
- **Follow-up events**: schedule notifications when the agent recommends “wait”, and the key date arrives.
- **Voice integration**: connect to speech-to-text and text-to-speech services.
- **Analytics**: log decisions and user follow-up actions for model/heuristic improvement.

---
_Last updated: 2025-09-22_
