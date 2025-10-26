<!-- f67714c1-a54f-4e8d-9617-16955f212afc 0dd7544e-8e74-44d4-b2b2-a2420a391ad4 -->
# Phase 4.4: Supervisor Orchestration & Integration

## Overview

Integrate all supervisor components into a complete workflow: conversation management, agent orchestration, and response formatting. Create the agent orchestrator that dispatches and coordinates all Layer 1, 2, and 3 agents, and build the LangGraph supervisor node that serves as the entry and exit point for the entire system.

## Reference

See `plans/supervisor-agent.plan.md` for agent orchestrator (lines 885-1024) and overall architecture.

## Files to Create

### 1. `src/supervisor/agent_orchestrator.py`

**Purpose**: Orchestrate execution of all agents (Market Data, Market Intelligence, Prediction, Decision) in correct sequence with error handling.

**Class**: `AgentOrchestrator`

**Constructor**:

- Creates instances of all agent nodes:
  - `market_data_node` (from Phase 1.3)
  - `market_intelligence_node` (from Phase 1.6)
  - `prediction_node` (from Phase 2.5)
  - `decision_node` (from Phase 3.4)

**Main Method**: `run_analysis(parameters: ExtractedParameters, correlation_id: str) -> Dict[str, Any]`

**Inputs**:

- `parameters`: ExtractedParameters with all user-provided values
- `correlation_id`: Unique ID for tracking this request through logs

**Outputs**: Dict with:

- `status`: "success" or "error"
- `action`: Decision action (if success)
- `confidence`: Decision confidence (if success)
- `timeline`: Timeline string (if success)
- `rationale`: List of reasons (if success)
- `staged_plan`: Staged plan (if applicable)
- `expected_outcome`: Expected outcome (if available)
- `risk_summary`: Risk summary (if available)
- `cost_estimate`: Cost estimate (if available)
- `warnings`: List of warning messages
- `error`: Error message (if status is error)
- `metadata`: Dict with correlation_id, timestamp

**Execution Flow**:

1. **Initialize State**:

   - Create initial AgentState from parameters
   - Set correlation_id
   - Set processing_stage to "initialized"

2. **Layer 1: Parallel Execution** (Market Data + Market Intelligence):

   - Log start of Layer 1
   - Call market_data_node(state) asynchronously
   - Call market_intelligence_node(state) asynchronously
   - Wait for both to complete
   - Update state with results
   - Collect any errors/warnings from Layer 1
   - Continue even if one fails (graceful degradation)

3. **Layer 2: Prediction** (Sequential):

   - Log start of Layer 2
   - Call prediction_node(state) asynchronously
   - Update state with prediction results
   - Collect any errors/warnings
   - Continue even if prediction fails (fallback to heuristics)

4. **Layer 3: Decision** (Sequential):

   - Log start of Layer 3
   - Call decision_node(state) asynchronously
   - Update state with decision results
   - If decision fails → return error response
   - If decision succeeds → extract recommendation

5. **Extract Recommendation**:

   - Get decision from state
   - Build recommendation dict with all fields
   - Add warnings collected from all layers
   - Add metadata (correlation_id, timestamp)

6. **Error Handling**:

   - Try-except around each layer
   - Log errors with correlation_id
   - Collect errors in warnings list
   - Continue to next layer when possible
   - Only abort if decision layer fails

**Internal Methods**:

- `_initialize_state(parameters: ExtractedParameters, correlation_id: str) -> AgentState`
  - Creates initial state from parameters
  - Sets all required fields

- `_extract_recommendation(state: AgentState, warnings: List[str]) -> Dict[str, Any]`
  - Extracts decision response from state
  - Converts to dict format
  - Adds warnings and metadata

- `_handle_error(error: Exception, layer: str, correlation_id: str) -> str`
  - Logs error with context
  - Returns user-friendly error message

**Logging**:

- Log start/end of each layer with correlation_id
- Log execution times for performance monitoring
- Log errors with full context
- Use structured logging with extra fields

### 2. `src/agentic/nodes/supervisor.py`

**Purpose**: LangGraph node that serves as entry point (NLU + orchestration) and exit point (response formatting).

**Entry Node Function**: `supervisor_start_node(state: AgentState) -> Dict[str, Any]`

**Inputs**: Initial AgentState (may be minimal or have user_query)

**Logic**:

1. Generate correlation_id if not present
2. Extract parameters from state.user_query using NLU (if needed)
3. Validate parameters are complete
4. Update state with extracted parameters
5. Set processing_stage to "nlu_complete"
6. Return dict with updated state fields

**Exit Node Function**: `supervisor_end_node(state: AgentState) -> Dict[str, Any]`

**Inputs**: AgentState with recommendation from decision node

**Logic**:

1. Extract recommendation from state
2. Format recommendation using ResponseFormatter
3. Update state with formatted message
4. Set processing_stage to "complete"
5. Return dict with final_response and completion status

**Note**: For Phase 4, these are simple pass-through nodes. Full implementation with conversation management comes when integrating with UI in Phase 5.

### 3. Update `src/agentic/graph.py`

**Purpose**: Add supervisor nodes to complete the LangGraph workflow.

**Changes**:

1. **Import Supervisor Nodes**:
   ```python
   from src.agentic.nodes.supervisor import supervisor_start_node, supervisor_end_node
   ```

2. **Add Nodes to Graph**:

   - Add "supervisor_start" node (entry point)
   - Add "supervisor_end" node (exit point)

3. **Update Edges**:

   - START → supervisor_start
   - supervisor_start → market_data (and market_intelligence in parallel)
   - decision → supervisor_end
   - supervisor_end → END

**Complete Flow**:

```
START
  ↓
supervisor_start (NLU)
  ↓
┌─────────────────────────────┐
│  Layer 1 (Parallel)         │
│  ├─ market_data            │
│  └─ market_intelligence    │
└─────────────────────────────┘
  ↓
prediction (Layer 2)
  ↓
decision (Layer 3)
  ↓
supervisor_end (Format response)
  ↓
END
```

### 4. `src/supervisor/supervisor.py`

**Purpose**: Main Supervisor class that ties everything together.

**Class**: `Supervisor`

**Constructor**:

- Creates ConversationManager instance
- Creates AgentOrchestrator instance
- Creates ResponseFormatter instance
- Loads SupervisorConfig

**Main Method**: `process_request(request: SupervisorRequest) -> SupervisorResponse`

**Logic**:

1. Process user input through ConversationManager
2. Get SupervisorResponse from conversation
3. If requires_input → return immediately (need more info)
4. If state is PROCESSING:

   - Extract parameters from response
   - Generate correlation_id
   - Run agent orchestration
   - Format recommendation
   - Update session with result
   - Return final response

5. Return response

**This is the main entry point for the entire system in non-graph mode.**

### 5. `tests/supervisor/test_agent_orchestrator.py`

**Purpose**: Unit tests for agent orchestrator.

**Test Cases**:

**Successful Orchestration**:

- `test_run_analysis_success`: All agents succeed → complete recommendation
- `test_run_analysis_with_all_data`: All layers return data → full recommendation
- `test_layer_1_parallel_execution`: Market data and intelligence run in parallel
- `test_warnings_collected`: Warnings from all layers collected
- `test_correlation_id_propagated`: correlation_id flows through all layers

**Graceful Degradation**:

- `test_market_data_fails_continues`: Market data fails → continues with intelligence
- `test_market_intelligence_fails_continues`: Intelligence fails → continues with market data
- `test_prediction_fails_uses_heuristics`: Prediction fails → decision uses heuristics
- `test_partial_data_recommendation`: Some agents fail → still generates recommendation

**Error Handling**:

- `test_decision_failure_returns_error`: Decision fails → returns error response
- `test_all_layers_fail_returns_error`: All agents fail → returns error
- `test_exception_handling`: Exceptions caught and logged

**Fixtures**:

- `orchestrator`: AgentOrchestrator instance
- `sample_parameters`: Complete ExtractedParameters
- Mock agent nodes for controlled testing

### 6. `tests/unit/test_agentic/test_nodes/test_supervisor.py`

**Purpose**: Unit tests for supervisor LangGraph nodes.

**Test Cases**:

- `test_supervisor_start_node`: Initializes state correctly
- `test_supervisor_start_with_correlation_id`: Uses provided correlation_id
- `test_supervisor_start_generates_correlation_id`: Generates new correlation_id if missing
- `test_supervisor_end_node`: Formats recommendation correctly
- `test_supervisor_end_updates_stage`: Sets processing_stage to complete

**Fixtures**:

- `initial_state`: AgentState for testing start node
- `complete_state`: AgentState with recommendation for testing end node

### 7. `tests/supervisor/test_supervisor.py`

**Purpose**: Unit tests for main Supervisor class.

**Test Cases**:

- `test_process_request_needs_more_info`: Incomplete params → returns with requires_input=True
- `test_process_request_confirmation`: Confirmation state → returns confirmation message
- `test_process_request_processing`: Processing state → runs orchestration → returns recommendation
- `test_process_request_full_flow`: Complete query → orchestration → formatted response

**Fixtures**:

- `supervisor`: Supervisor instance
- Mock ConversationManager
- Mock AgentOrchestrator

### 8. `tests/integration/test_supervisor_integration.py`

**Purpose**: End-to-end integration test for complete supervisor workflow.

**Test Scenario**: Full conversation through to recommendation

**Test Case**: `test_complete_supervisor_flow`

**Steps**:

1. Create Supervisor instance
2. Submit initial query: "Convert 5000 USD to EUR urgently, moderate risk, within a week"
3. Verify parameters extracted
4. Verify confirmation shown
5. Confirm with "yes"
6. Verify agents orchestrated (mock or real)
7. Verify recommendation formatted
8. Verify final response complete

**Expected**:

- All conversation steps work
- Parameters extracted correctly
- Orchestration executes
- Recommendation formatted properly
- Session state managed correctly

### 9. `tests/integration/test_full_graph_execution.py`

**Purpose**: Integration test for complete LangGraph execution.

**Test Case**: `test_full_graph_workflow`

**Steps**:

1. Create complete graph using create_graph()
2. Initialize state with user query and parameters
3. Execute graph (invoke or stream)
4. Verify all nodes executed in correct order
5. Verify state updated at each step
6. Verify final recommendation in state

**Validation**:

- Assert supervisor_start executed
- Assert market_data and market_intelligence executed (parallel)
- Assert prediction executed
- Assert decision executed
- Assert supervisor_end executed
- Assert final state has formatted response

## Validation

Manual end-to-end validation:

```python
from src.supervisor.supervisor import Supervisor
from src.supervisor.models import SupervisorRequest
from src.agentic.graph import create_graph
from src.agentic.state import initialize_state

# Test 1: Supervisor class (conversation + orchestration)
print("=== Test 1: Supervisor Workflow ===")
supervisor = Supervisor()

# Initial request
response1 = supervisor.process_request(SupervisorRequest(
    user_input="Convert 5000 USD to EUR urgently, moderate risk, within a week"
))
print(f"State: {response1.state}")
print(f"Message:\n{response1.message}\n")

# Confirm
if response1.requires_input:
    response2 = supervisor.process_request(SupervisorRequest(
        user_input="yes",
        session_id=response1.session_id
    ))
    print(f"State: {response2.state}")
    print(f"Final Message:\n{response2.message}\n")

# Test 2: LangGraph execution
print("\n=== Test 2: LangGraph Workflow ===")
graph = create_graph()

# Initialize state
state = initialize_state("Convert 5000 USD to EUR")
state["amount"] = 5000
state["currency_pair"] = "USD/EUR"
state["risk_tolerance"] = "moderate"
state["urgency"] = "urgent"
state["timeframe"] = "1_week"
state["timeframe_days"] = 7

# Execute graph
print("Executing graph...")
final_state = await graph.ainvoke(state)

print(f"Processing stage: {final_state['processing_stage']}")
print(f"Decision status: {final_state['decision_status']}")
if final_state.get('recommendation'):
    print(f"Action: {final_state['recommendation']['action']}")
    print(f"Confidence: {final_state['recommendation']['confidence']}")

# Test 3: Agent orchestrator directly
print("\n=== Test 3: Agent Orchestrator ===")
from src.supervisor.agent_orchestrator import AgentOrchestrator
from src.supervisor.models import ExtractedParameters

orchestrator = AgentOrchestrator()
params = ExtractedParameters(
    currency_pair="USD/EUR",
    base_currency="USD",
    quote_currency="EUR",
    amount=5000,
    risk_tolerance="moderate",
    urgency="urgent",
    timeframe="1_week",
    timeframe_days=7
)

recommendation = await orchestrator.run_analysis(params, "test-correlation-id")
print(f"Status: {recommendation['status']}")
print(f"Action: {recommendation.get('action')}")
print(f"Confidence: {recommendation.get('confidence')}")
```

## Success Criteria

- [ ] AgentOrchestrator executes all agents in correct sequence
- [ ] Layer 1 agents run in parallel (market_data + market_intelligence)
- [ ] Layers 2 and 3 run sequentially (prediction → decision)
- [ ] Graceful degradation when agents fail
- [ ] Warnings collected from all layers
- [ ] Correlation ID propagated through all logs
- [ ] Supervisor class integrates conversation and orchestration
- [ ] Supervisor nodes (start/end) work in LangGraph
- [ ] Complete graph executes end-to-end
- [ ] All unit tests pass with >80% coverage
- [ ] Integration tests demonstrate full workflow
- [ ] Manual validation shows complete system working

## Key Design Decisions

1. **Orchestrator Pattern**: Centralized agent dispatch and coordination
2. **Graceful Degradation**: Continue on partial failures, only abort on critical errors
3. **Parallel Layer 1**: Market data and intelligence fetch simultaneously
4. **Sequential Layers 2-3**: Prediction needs Layer 1, decision needs all previous
5. **Correlation ID**: Track requests through entire system for debugging
6. **Warning Collection**: Aggregate warnings from all layers for user visibility
7. **Supervisor as Entry/Exit**: Clean separation of concerns in graph
8. **Flexible Integration**: Supervisor works standalone or via LangGraph

## Integration Points

- ConversationManager (Phase 4.2) handles parameter collection
- NLUExtractor (Phase 4.1) parses user input
- ResponseFormatter (Phase 4.3) formats final output
- Agent nodes from Phases 1, 2, 3 orchestrated in sequence
- LangGraph provides workflow structure
- TUI/Web UI (Phase 5) will use Supervisor as main entry point

## Phase 4 Complete

After this phase, the Supervisor Agent is fully operational and can:

- Conduct multi-turn conversations to collect parameters
- Extract parameters from natural language
- Orchestrate all agents in correct sequence with parallel execution
- Handle errors and partial failures gracefully
- Format recommendations for users
- Integrate with LangGraph workflow
- Provide complete end-to-end functionality

**Next**: Phase 5 will add user interfaces (TUI and Web) that interact with the Supervisor.

### To-dos

- [ ] Implement AgentOrchestrator in src/supervisor/agent_orchestrator.py with layer-based execution
- [ ] Create supervisor nodes in src/agentic/nodes/supervisor.py and update graph.py
- [ ] Implement main Supervisor class in src/supervisor/supervisor.py
- [ ] Write unit tests for agent orchestrator (success, degradation, errors)
- [ ] Write unit tests for supervisor nodes and main Supervisor class
- [ ] Write integration tests for complete supervisor workflow and full graph execution