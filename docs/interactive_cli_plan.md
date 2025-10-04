# Interactive Chat CLI Plan

## Motivation & Goals
The current CLI wrappers (`scripts/agentic_cli.py`, `scripts/agentic_cli_interactive.py`) collect a single request, run the workflow once, and render a static report. To bridge toward the eventual FastAPI + web frontend, we need a conversational CLI that:
- Maintains a running session so users can adjust parameters, ask follow-up questions, or request clarifications without restarting the process.
- Exposes the same capabilities that the future chat UI will require (session state, message history, structured commands).
- Stays fully offline-friendly and testable, reusing the stabilized mock harness where possible.
- Provides a clean contract for eventual reuse by FastAPI (e.g., a `ConversationSession` class that both CLI and API layers can drive).

## Current Baseline
- `agentic_cli.py` orchestrates a single-shot workflow and prints summaries.
- `agentic_cli_interactive.py` offers a text prompt loop but treats each run independently (no conversation memory or commands).
- There is no SDK-style abstraction that encapsulates session state, rendering, or command handling.

## High-Level Architecture
1. **Session Manager** (`currency_assistant.cli.session`)
   - Holds conversation history (list of user/assistant turns, normalized requests, serialized states).
   - Provides helpers to build payloads for the agentic workflow, applying defaults or last-known values.
   - Exposes methods `submit_request()`, `show_last()`, `reset()`, `export_history()`.

2. **Command Router** (`currency_assistant.cli.commands`)
   - Parses user input into domain commands (`/analyze`, `/set risk=low`, `/show market`, `/save transcript.json`, `/exit`).
   - Uses a simple grammar or regex-based parser to distinguish between (a) free-form follow-up questions and (b) prefixed commands.

3. **Runner / UI Loop** (`scripts/chat_cli.py`)
   - Event loop that reads input, routes to the command handler, invokes `agentic.run_agentic_workflow` when analysis is requested, and streams formatted output back to the terminal.
   - Supports optional streaming/step-by-step updates (e.g., show node progress with correlation ID).

4. **Renderer**
   - Converts `AgentGraphState` (or serialized dict) into terminal-friendly sections with optional color/indentation (respecting ASCII-only requirements).
   - Provides utilities to render incremental updates (market summary first, then economic, etc.).

5. **Persistence Layer (optional)**
   - Minimal JSON/markdown exporter so users can save a transcript or reload later.
   - Hooks Storage into CLI via `/save` and future web UI reuse.

## Detailed Implementation Plan

### Phase A – Foundations
- [ ] **Create `src/cli/session.py`**
  - Define `ConversationSession` with attributes: `history`, `defaults`, `last_state`, `correlation_id`.
  - Methods:
    - `build_payload(overrides: dict) -> dict`
    - `record_turn(request: dict, state: AgentGraphState)`
    - `get_last_state()` / `summaries()` helpers for reuse by renderer.
- [ ] **Add serialization helpers** to persist/load session history as JSON (leveraging existing `serialize_state`).

### Phase B – Command & Parser Layer
- [ ] **Design command syntax**: e.g.
  - `/analyze` (run workflow using current defaults)
  - `/set pair=USD/JPY amount=1500 timeframe=14 risk=low`
  - `/show market`, `/show recommendation`, `/history`, `/reset`, `/save filename`
  - `/help`, `/exit`
  - Any other input treated as free-form follow-up; the CLI can explain how to adjust parameters.
- [ ] **Implement command parser** (`src/cli/commands.py`).
  - Use simple parsing (split on spaces, parse `key=value`).
  - Return structured commands or `UserQuery` objects.

### Phase C – Runner & Integration
- [ ] **New script `scripts/chat_cli.py`** that:
  1. Initializes `ConversationSession` with default settings.
  2. Prints a welcome banner and help snippet.
  3. Repeatedly reads input (supports multi-line via trailing `\`?).
  4. Routes to command handler:
     - `/analyze` → build payload via session and call `run_agentic_workflow`.
     - `/set` → update session defaults, echo confirmation.
     - `/show` → render pieces of the last state (market/economic/risk/recommendation).
     - `/history` → list prior actions with timestamps/confidence.
     - `/save` → call session exporter.
     - `/reset` → clear history and defaults.
     - `/exit` → break loop.
     - Free-form query → run `analyze` with optional `user_notes` capturing query text.
  5. Displays progress using correlation IDs (e.g., `Running analysis [cid]...`).
- [ ] **Rendering utilities**
  - Extract formatting into `src/cli/render.py` so both CLI and tests can reuse them.
  - Include optional verbosity levels (short vs. detailed).

### Phase D – Streaming & Feedback (Nice-to-have)
- [ ] Hook into LangGraph callbacks (if available) to stream node completion messages.
- [ ] Allow toggling between streaming vs. final summary via `/set stream=true`.

### Phase E – Testing & Documentation
- [ ] Add unit tests for the command parser and session serialization (`tests/test_cli_commands.py`, `tests/test_cli_session.py`).
- [ ] Extend mocked workflow test to run an end-to-end CLI cycle: feed scripted inputs, capture outputs (use `io.StringIO`).
- [ ] Document usage in `docs/interactive_cli_plan.md` (this file) and cross-link from `docs/system_stabilization_plan.md`.
- [ ] Update README / quick start to mention `uv run python scripts/chat_cli.py`.

## Future Alignment with FastAPI/Web UI
- The `ConversationSession` and renderer (pure functions returning dict/JSON) become reusable service layers for the API.
- Command parser logic can map to HTTP endpoints (e.g., `/session/{id}/analyze`, `/session/{id}/update`).
- Streaming hooks and correlation IDs map naturally to WebSocket or SSE for the web frontend.

## Risks & Mitigations
- **CLI Parser Complexity**: Keep command grammar minimal; support `--` style options only if needed.
- **Session Memory Growth**: Provide `/history truncate` and limit stored states or allow file-backed persistence.
- **Blocking Calls**: Ensure CLI stays responsive by running the workflow asynchronously (use `asyncio.run` or `uvloop`) to avoid UI freezes.
- **Credential Handling**: Mirror existing `.env` pattern; show warnings if keys missing rather than failing silently.

## Milestones
1. Foundations (Session + Parser) – 1-2 days.
2. Runner integration & core commands – 2 days.
3. Rendering polish + streaming (optional) – 1 day.
4. Testing/documentation – 1 day.

Deliverable: `scripts/chat_cli.py` providing a conversational terminal experience, with reusable session/command modules that pave the way for the upcoming FastAPI backend.
