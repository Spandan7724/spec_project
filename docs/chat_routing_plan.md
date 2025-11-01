# Chat Routing Enhancement Plan

## Goals

- Allow the chat assistant to answer free-form currency questions without immediately entering the analysis workflow.
- Keep the existing parameter-collection + analysis flow intact, but only trigger it when the user explicitly requests an analysis.
- Reuse existing agentic components (market data, intelligence, prediction) when the question warrants it, without forcing the full workflow every time.
- Maintain configurable limits for conversation history; continue using `config.yaml` to control how much prior context the LLM receives.

## High-Level Architecture

```
User Message
   │
   ▼
Intent Router ──────────────┐
   │                        │
   ├─ Analysis Intent ──────┴─▶ Existing parameter collection → analysis pipeline → results
   │
   └─ General Question ──▶ Free-form responder (LLM + optional data fetch)
```

### Components Involved

- **ConversationManager** (`src/supervisor/conversation_manager.py`): orchestration layer that currently handles parameter collection and follow-up questions.
- **NLUExtractor** (`src/supervisor/nlu_extractor.py`): entity extraction for currency pair, amount, timeframe, etc. Can be leveraged for intent routing.
- **Agentic nodes** (`src/agentic/nodes/*`): async helpers for market data, intelligence, prediction, decision. Can be called individually for richer answers.
- **LLM manager** (`src/llm/manager.py`, `src/llm/agent_helpers.py`): used to generate natural-language responses.
- **Config** (`config.yaml` via `src/config.py`): hold new settings for routing thresholds or prompts.

## Detailed Steps

### 1. Add Router to ConversationManager

1. Extend `ConversationManager` with a method like `_classify_intent(message: str) -> Literal["analysis", "general", "reset"]`.
   - Use simple heuristics first: keyword checks (e.g., “convert”, “analysis”, “trade”) or rely on `NLUExtractor` outputs.
   - Optional: fallback to a lightweight LLM intent prompt if heuristics are inconclusive.
2. Modify `process_input` / `aprocess_input`:
   - After logging the user message, run the classifier.
   - If the intent is `"reset"`, reuse existing restart logic.
   - If `"analysis"`: proceed with the current parameter collection flow.
   - If `"general"` and the session has no active analysis, route to a new `handle_general_question` method (see below) and return the response without modifying `session.state`.
   - If the session is already in `RESULTS_READY`, keep using the existing follow-up handler (the analysis result remains the context of record).

### 2. Implement Free-Form Responder

Create a new method, for example:

```python
def handle_general_question(self, session: ConversationSession, question: str) -> SupervisorResponse:
    ...
```

Responsibilities:

- Decide what data to fetch:
  - Simple informational question → direct LLM answer using current config / heuristics.
  - Questions mentioning “rate”, “volatility”, etc. → optionally call `market_data_node` (maybe with caching).
  - If question implies research (“What’s the sentiment on EUR/USD?”), optionally call `market_intelligence` or `prediction` nodes.
  - Keep calls optional/configurable to avoid unnecessary latency.
- Build a concise prompt for `_answer_with_llm`:
  - Provide fetched data (if any) as structured context.
  - Provide recent chat history (use the existing `self.history_limit` logic).
- Return a `SupervisorResponse` with `state` unchanged (likely `INITIAL` or `RESULTS_READY`) and `requires_input=True` to keep the chat interactive.
- Append the assistant’s reply to `session.conversation_history`.

### 3. Explicit Trigger for Analysis

- Update UI copy and quick prompts to encourage phrases like “Start an analysis” or “Run an analysis for {pair}”.
- Consider adding a dedicated user command (`/analyze`, button press) that sends a particular keyword recognized by the router.
- When the router detects this trigger, switch `session.state` into the appropriate collection state (current behaviour).

### 4. Maintain Configurability

- Add `chat.router` block in `config.yaml` to control routing behaviour (thresholds, enable/disable market-data fetch for general questions, etc.).
- Extend `ConversationManager._load_history_limit()` if additional toggles are needed.
- Example:

```yaml
chat:
  history:
    messages: 8
    max: false
  router:
    enable_llm_intent: false
    analysis_keywords: ["convert", "analysis", "hedge", "swap"]
    general_keywords: ["outlook", "trend", "rate", "volatility"]
    fetch_market_data_for_general: true
```

### 5. Optional: Reuse Agentic Nodes

- Wrap the async nodes in sync helpers (using `asyncio.run`) similar to the existing pipeline, so the responder can:
  - fetch live rate snapshots,
  - summarise intelligence,
  - optionally run predictions.
- Keep these calls guarded by config flags to avoid hitting providers unnecessarily during casual chat.

### 6. Testing Strategy

- Unit tests:
  - Validate `_classify_intent` covers expected phrases/edge cases.
  - Ensure `handle_general_question` returns a response without mutating session state when no analysis is active.
- Integration tests (or manual QA):
  - Ask free-form questions at startup; confirm the system responds without requesting amount/timeframe.
  - Trigger an analysis and confirm the follow-up flow still works.
  - After the analysis completes, ask general questions; verify it stays in results-ready mode.
  - Try reset keywords and confirm the session restarts.
- Regression tests for existing analysis pipeline to ensure no behaviour change when the user explicitly starts an analysis.

## Future Enhancements

- Replace heuristic router with a dedicated intent classifier endpoint or small local model.
- Add memory of past analyses per session to answer “How does this compare with last week?”.
- Introduce rate limits or caching for free-form market data fetches to keep costs down.
- Feed market and intelligence data into vector stores for richer conversational capabilities.

## Summary

By inserting an intent router before the current conversation flow, the chat assistant can answer casual questions, only launching the full agentic workflow when the user explicitly signals an analysis request. The plan reuses existing LLM and market data components, keeps history limits configurable, and lays groundwork for future enhancements like more advanced intent classification and knowledge memories.
