<!-- f67714c1-a54f-4e8d-9617-16955f212afc 0660becd-a9c9-4846-9b6e-bc0bbaa56a67 -->
# Phase 4.2: Conversation Manager

## Overview

Build the conversation manager that orchestrates multi-turn dialogues with users. It guides them through parameter collection, shows what was extracted, asks for missing information, handles confirmation and parameter changes, and maintains session state throughout the conversation.

## Reference

See `plans/supervisor-agent.plan.md` for conversation flow examples (lines 16-69) and conversation manager implementation (lines 392-883).

## Files to Create

### 1. `src/supervisor/conversation_manager.py`

**Purpose**: Manage multi-turn conversation flow and session state.

**Class**: `ConversationManager`

**Constructor**:

- Creates `NLUExtractor` instance
- Initializes `sessions` dict for in-memory session storage

**Main Method**: `process_input(request: SupervisorRequest) -> SupervisorResponse`

**Logic**:

1. Get or create session using session_id
2. Add user message to conversation history
3. Route to appropriate handler based on current `ConversationState`
4. Return `SupervisorResponse` with next message and state

**State Handlers**:

- `_handle_initial(session, user_input) -> SupervisorResponse`
  - **Purpose**: Process initial user query
  - **Logic**:

    1. Extract parameters using NLUExtractor
    2. Store extracted parameters in session
    3. Build response showing what was extracted (✓ marks)
    4. Determine missing parameters
    5. If all complete → move to confirmation
    6. Otherwise → ask for first missing parameter

  - **Output**: Response with extracted params and question for next param

- `_handle_collection(session, user_input) -> SupervisorResponse`
  - **Purpose**: Collect specific parameter based on current state
  - **Logic**:

    1. Determine which parameter being collected (from ConversationState)
    2. Extract that parameter from user input
    3. Validate extracted value
    4. If invalid → re-ask with error message
    5. If valid → update session parameters
    6. Confirm parameter set
    7. Check if all parameters now complete
    8. If complete → move to confirmation
    9. Otherwise → ask for next missing parameter

  - **States**: COLLECTING_CURRENCY_PAIR, COLLECTING_AMOUNT, COLLECTING_RISK, COLLECTING_URGENCY, COLLECTING_TIMEFRAME

- `_handle_confirmation(session, user_input) -> SupervisorResponse`
  - **Purpose**: Handle user confirmation, changes, or restart requests
  - **Logic**:
    - If "yes"/"confirm"/"proceed" → set state to PROCESSING, return parameters
    - If "restart"/"start over" → reset session, ask initial question
    - If "change <param>" → handle parameter change
    - Otherwise → repeat confirmation prompt
  - **Output**: Either proceed to processing or stay in confirmation

**Helper Methods**:

- `_get_or_create_session(session_id: Optional[str]) -> ConversationSession`
  - Returns existing session or creates new one with UUID
  - Stores session in sessions dict

- `_move_to_confirmation(session, prefix="") -> SupervisorResponse`
  - **Purpose**: Transition to confirmation state and show all parameters
  - **Logic**:

    1. Set state to CONFIRMING
    2. Build message showing all collected parameters with ✓ marks
    3. Add confirmation options (yes/change/restart)
    4. Add message to history
    5. Return response

  - **Output**: Formatted confirmation message

- `_ask_for_parameter(session, param_name, prefix_parts=[]) -> SupervisorResponse`
  - **Purpose**: Ask user for specific missing parameter
  - **Logic**:

    1. Get question template for parameter
    2. Set appropriate ConversationState
    3. Build message with options/examples
    4. Add to conversation history
    5. Return response

  - **Questions**:
    - currency_pair: "What currency pair? (e.g., USD/EUR)"
    - amount: "What amount? (e.g., 5000)"
    - risk_tolerance: "Risk tolerance? conservative/moderate/aggressive"
    - urgency: "Urgency? urgent/normal/flexible"
    - timeframe: "Timeframe? immediate/1_day/1_week/1_month"

- `_handle_parameter_change(session, user_input) -> SupervisorResponse`
  - **Purpose**: Handle "change <param> to <value>" requests
  - **Logic**:

    1. Parse change command (e.g., "change risk to aggressive")
    2. Map parameter name (risk → risk_tolerance)
    3. Validate parameter name
    4. Extract and validate new value
    5. Update session parameters
    6. Return to confirmation with updated params

  - **Supported changes**: risk, urgency, timeframe, amount

**Direct Parsers** (for single-word responses):

- `_parse_risk_direct(text) -> Optional[str]`
  - Matches: conservative, moderate, aggressive (and synonyms)

- `_parse_urgency_direct(text) -> Optional[str]`
  - Matches: urgent, normal, flexible (and synonyms)

- `_parse_timeframe_direct(text) -> Optional[str]`
  - Matches: immediate, 1_day, 1_week, 1_month

**Session Storage**:

- In-memory dict: `{session_id: ConversationSession}`
- MVP approach, no database persistence
- Sessions timeout after configured duration (handled separately)

### 2. `src/supervisor/session_manager.py`

**Purpose**: Handle session lifecycle and cleanup.

**Class**: `SessionManager`

**Constructor**: Takes `SupervisorConfig`

**Methods**:

- `cleanup_expired_sessions(sessions: Dict[str, ConversationSession]) -> int`
  - **Purpose**: Remove sessions older than timeout
  - **Logic**:

    1. Iterate through sessions
    2. Check if `last_updated` > timeout threshold
    3. Remove expired sessions
    4. Return count of removed sessions

  - **Schedule**: Should be called periodically (e.g., every 5 minutes)

- `get_session_stats(sessions: Dict) -> Dict[str, Any]`
  - **Purpose**: Return session statistics for monitoring
  - **Output**: 
    - total_sessions: int
    - active_sessions (updated in last 5 min): int
    - sessions_by_state: Dict[state, count]

### 3. `tests/supervisor/test_conversation_manager.py`

**Purpose**: Unit tests for conversation manager.

**Test Cases**:

**Initial Handling**:

- `test_handle_initial_full_query`: Query with all params → move to confirmation
- `test_handle_initial_partial_query`: Query with some params → ask for missing
- `test_handle_initial_minimal_query`: Query with only currency → ask for amount
- `test_handle_initial_no_currency`: Query without currency → ask for currency pair

**Collection Handling**:

- `test_collect_currency_pair_valid`: Valid currency → accepted
- `test_collect_currency_pair_invalid`: Invalid currency → re-ask
- `test_collect_amount_valid`: Valid number → accepted
- `test_collect_amount_invalid`: Non-number → re-ask
- `test_collect_risk_valid`: "moderate" → accepted
- `test_collect_risk_invalid`: Unrecognized word → re-ask
- `test_collect_urgency_direct`: "urgent" → accepted
- `test_collect_timeframe_direct`: "1_week" → accepted

**Confirmation Handling**:

- `test_confirmation_yes`: "yes" → proceed to processing
- `test_confirmation_restart`: "restart" → reset to initial
- `test_confirmation_change_risk`: "change risk to aggressive" → updated, stay in confirmation
- `test_confirmation_change_amount`: "change amount to 3000" → updated
- `test_confirmation_invalid`: Random text → repeat confirmation prompt

**Session Management**:

- `test_get_or_create_session_new`: Creates new session with UUID
- `test_get_or_create_session_existing`: Returns existing session
- `test_session_conversation_history`: Messages added to history
- `test_session_state_transitions`: State changes correctly through flow

**Full Conversation Flows**:

- `test_full_flow_complete_initial`: Full params in first message → confirmation → proceed
- `test_full_flow_collect_all`: Empty initial → collect each param → confirmation → proceed
- `test_full_flow_with_change`: Params → confirmation → change → confirmation → proceed

**Fixtures**:

- `conversation_manager`: ConversationManager instance
- `sample_session`: Pre-populated ConversationSession

### 4. `tests/supervisor/test_session_manager.py`

**Purpose**: Unit tests for session lifecycle management.

**Test Cases**:

- `test_cleanup_expired_sessions`: Old sessions removed
- `test_cleanup_keeps_active_sessions`: Recent sessions kept
- `test_get_session_stats_empty`: No sessions → all zeros
- `test_get_session_stats_populated`: Sessions present → correct counts
- `test_sessions_by_state_count`: Verify count per state

**Fixtures**:

- `session_manager`: SessionManager with test config
- `test_sessions`: Dict of sessions with various ages/states

### 5. `tests/supervisor/test_conversation_integration.py`

**Purpose**: Integration test for full conversation flows.

**Test Scenarios**:

**Scenario 1: Happy Path - Full Query**

- User: "Convert 5000 USD to EUR urgently, moderate risk, within a week"
- Expected: Extract all → show confirmation → user says yes → return processing

**Scenario 2: Incremental Collection**

- User: "USD to EUR"
- System: "What amount?"
- User: "5000"
- System: "Risk tolerance?"
- User: "moderate"
- System: "Urgency?"
- User: "normal"
- System: "Timeframe?"
- User: "1_week"
- System: Shows confirmation
- User: "yes"
- Expected: Return processing with all params

**Scenario 3: Change During Confirmation**

- User provides all params → confirmation
- User: "change risk to conservative"
- System: Updated confirmation
- User: "yes"
- Expected: Proceed with updated params

**Scenario 4: Restart Mid-Conversation**

- User starts conversation → provides some params
- User: "restart"
- System: Reset to initial
- Expected: Clean slate, can start over

## Validation

Manual validation script:

```python
from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.models import SupervisorRequest

# Create manager
manager = ConversationManager()

# Simulate conversation
print("=== Test Conversation 1: Full Query ===")
response = manager.process_input(SupervisorRequest(
    user_input="Convert 5000 USD to EUR urgently, moderate risk, within a week"
))
print(f"State: {response.state}")
print(f"Message:\n{response.message}\n")
print(f"Requires input: {response.requires_input}\n")

# Confirm
if response.requires_input:
    response = manager.process_input(SupervisorRequest(
        user_input="yes",
        session_id=response.session_id
    ))
    print(f"State: {response.state}")
    print(f"Parameters: {response.parameters}\n")

print("\n=== Test Conversation 2: Incremental ===")
session_id = None

# Initial query
response = manager.process_input(SupervisorRequest(user_input="USD to EUR"))
session_id = response.session_id
print(f"State: {response.state}")
print(f"Message:\n{response.message}\n")

# Provide amount
response = manager.process_input(SupervisorRequest(
    user_input="5000",
    session_id=session_id
))
print(f"State: {response.state}")
print(f"Message:\n{response.message}\n")

# Continue through remaining parameters...
# (risk, urgency, timeframe)

print("\n=== Test Conversation 3: Parameter Change ===")
response = manager.process_input(SupervisorRequest(
    user_input="Convert 10000 USD to GBP, conservative, urgent, immediate"
))
session_id = response.session_id
print(f"Message:\n{response.message}\n")

# Change risk
response = manager.process_input(SupervisorRequest(
    user_input="change risk to aggressive",
    session_id=session_id
))
print(f"Message:\n{response.message}\n")

# Confirm
response = manager.process_input(SupervisorRequest(
    user_input="yes",
    session_id=session_id
))
print(f"State: {response.state}")
print(f"Risk in params: {response.parameters.risk_tolerance}")
```

## Success Criteria

- [ ] ConversationManager routes to correct handler based on state
- [ ] Initial handler extracts parameters and asks for missing ones
- [ ] Collection handlers validate input and re-prompt on invalid input
- [ ] Confirmation handler recognizes yes/change/restart commands
- [ ] Parameter changes update session and return to confirmation
- [ ] Session state transitions correctly through conversation
- [ ] Conversation history records all messages with timestamps
- [ ] Session manager cleans up expired sessions
- [ ] All unit tests pass with >80% coverage
- [ ] Integration tests demonstrate complete conversation flows
- [ ] Manual validation shows natural conversation flow

## Key Design Decisions

1. **State Machine**: Clear states for each collection phase
2. **Flexible Input**: Accept direct answers or full sentences
3. **Confirmation Step**: Always show all parameters before proceeding
4. **Change Support**: Allow parameter modification at confirmation
5. **Restart Anytime**: Users can restart conversation at any point
6. **Session Storage**: In-memory for MVP, easy to swap to DB later
7. **Conversation History**: Track all turns for context and debugging
8. **Graceful Validation**: Re-prompt with helpful messages on invalid input

## Integration Points

- Uses NLUExtractor (Phase 4.1) for parameter extraction
- Returns ExtractedParameters to Agent Orchestrator (Phase 4.3)
- Conversation history can be displayed in UI (Phase 5)
- Session IDs used across TUI and Web UI

### To-dos

- [ ] Implement ConversationManager in src/supervisor/conversation_manager.py with state handlers
- [ ] Implement SessionManager in src/supervisor/session_manager.py for session cleanup
- [ ] Write comprehensive unit tests for conversation manager (initial, collection, confirmation)
- [ ] Write unit tests for session manager (cleanup, stats)
- [ ] Write integration tests for full conversation flows (happy path, incremental, changes, restart)