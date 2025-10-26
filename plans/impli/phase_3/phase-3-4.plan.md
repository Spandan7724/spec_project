<!-- f67714c1-a54f-4e8d-9617-16955f212afc 5b8e8f9e-2f19-4ba9-9251-14587a92666a -->
# Phase 3.4: Decision Engine Integration & Node

## Overview

Integrate all decision components (utility scorer, staging planner, heuristics, confidence aggregator) into the main Decision Engine orchestrator. Create the LangGraph node that receives state from upstream agents and produces final recommendations. This is the culmination of Phase 3.

## Reference

See `plans/decision-engine.plan.md` for complete orchestration logic and decision flow (lines 305-315, 317-372).

## Files to Create

### 1. `src/decision/decision_maker.py`

**Purpose**: Main orchestrator that coordinates all decision components to produce final recommendations.

**Class**: `DecisionMaker`

**Constructor**: Takes `DecisionConfig`

**Attributes**:

- `config`: DecisionConfig
- `utility_scorer`: UtilityScorer instance
- `staging_planner`: StagingPlanner instance
- `cost_calculator`: CostCalculator instance
- `heuristic_maker`: HeuristicDecisionMaker instance
- `confidence_agg`: ConfidenceAggregator instance
- `risk_calculator`: RiskCalculator instance

**Main Method**: `make_decision(request: DecisionRequest) -> DecisionResponse`

**Inputs**: `DecisionRequest` with all upstream data

**Outputs**: Complete `DecisionResponse` with action, confidence, rationale, plans, etc.

**Decision Flow**:

1. **Validate Request**:

   - Check required fields (amount, risk_tolerance, urgency, timeframe, market)
   - Log component availability

2. **Calculate Risk Summary**:

   - Use RiskCalculator to assess volatility and event risk
   - Store for inclusion in response

3. **Determine Decision Mode**:

   - Check if prediction available and reliable:
     - Prediction exists AND status == "success" AND confidence >= min_threshold
   - If yes: Use **utility-based decision**
   - If no: Use **heuristic fallback**

4. **Utility-Based Decision Path**:

a. **Score All Actions**:

   - Call `utility_scorer.score_actions(request)`
   - Get scores for convert_now, staged, wait

b. **Apply Hard Constraints** (event-gating for conservative):

   - If risk_tolerance == "conservative" AND high-impact event within threshold:
     - Set convert_now score to -999 (effectively block it)

c. **Select Best Action**:

   - `best_action = argmax(utility_scores)`

d. **Generate Staging Plan** (if staged selected):

   - Call `staging_planner.create_staged_plan(request)`

e. **Calculate Expected Outcome**:

   - Extract expected_rate, range_low, range_high from prediction
   - Calculate expected_improvement_bps

f. **Aggregate Confidence**:

   - Call `confidence_agg.aggregate_confidence(request, utility_scores, is_heuristic=False)`

5. **Heuristic Fallback Path**:

a. **Make Heuristic Decision**:

   - Call `heuristic_maker.make_heuristic_decision(request)`
   - Get action, confidence, rationale, method

b. **Generate Staging Plan** (if staged selected):

   - Call `staging_planner.create_staged_plan(request)`

c. **Calculate Expected Outcome** (from intelligence or technical):

   - Use intelligence bias or technical indicators for rough estimate

d. **Aggregate Confidence**:

   - Call `confidence_agg.aggregate_confidence(request, {}, is_heuristic=True)`
   - Apply heuristic penalty

6. **Calculate Costs**:

   - Determine if staged (action == "staged_conversion")
   - Call `cost_calculator.calculate_cost_estimate(request, is_staged)`

7. **Generate Timeline String**:

   - convert_now: "Immediate execution recommended"
   - staged_conversion: "Execute in {num_tranches} tranches over {timeframe} days"
   - wait: "Wait until {timeframe} for better rate"

8. **Generate Rationale** (top 3-5 reasons):

   - For utility-based:
     - "Best utility score: {score}"
     - Mention expected improvement if positive
     - Mention risk factors (volatility, events)
     - Mention urgency alignment
     - Mention cost considerations
   - For heuristic:
     - Use rationale from heuristic_maker
     - Prepend "Using heuristic fallback (prediction unavailable)"

9. **Collect Warnings**:

   - Carry forward warnings from request
   - Add new warnings:
     - "Prediction unavailable, using heuristics"
     - "High volatility detected"
     - "High-impact event approaching"
     - "Low confidence in recommendation"

10. **Build DecisionResponse**:

    - Populate all fields
    - Include utility_scores (for transparency)
    - Include component_confidences
    - Set timestamp

**Internal Methods**:

- `_is_prediction_reliable(prediction: Optional[Dict]) -> bool`
  - Returns True if prediction exists, has good status, meets confidence threshold

- `_apply_hard_constraints(utility_scores: Dict, request: DecisionRequest) -> Dict`
  - Applies event-gating for conservative risk profile
  - Returns modified utility scores

- `_select_best_action(utility_scores: Dict) -> str`
  - Returns action with highest utility score

- `_generate_expected_outcome(request: DecisionRequest, action: str, is_heuristic: bool) -> ExpectedOutcome`
  - Extracts or estimates expected rates and improvements

- `_generate_timeline(action: str, staged_plan: Optional[StagedPlan], timeframe_days: int) -> str`
  - Creates human-readable timeline string

- `_generate_rationale(action: str, utility_scores: Dict, request: DecisionRequest, risk_summary: RiskSummary, is_heuristic: bool, heuristic_rationale: Optional[List[str]]) -> List[str]`
  - Generates top 3-5 reasons for recommendation

- `_collect_warnings(request: DecisionRequest, risk_summary: RiskSummary, confidence: float, is_heuristic: bool) -> List[str]`
  - Aggregates all warnings

**Error Handling**:

- Gracefully handle missing/partial data
- Log errors and include in warnings
- Never fail completely - always provide some recommendation
- Default to conservative action (WAIT or STAGED) if uncertain

### 2. `src/agentic/nodes/decision.py`

**Purpose**: LangGraph node that wraps DecisionMaker and integrates with agent state.

**Function**: `decision_node(state: AgentState) -> Dict[str, Any]`

**Inputs**: `AgentState` from LangGraph with:

- User parameters (from supervisor)
- Market snapshot (from market_data node)
- Intelligence report (from market_intelligence node, optional)
- Price forecast (from prediction node, optional)

**Process**:

1. **Extract Data from State**:

   - Build DecisionRequest from state fields
   - Map state fields to request fields:
     - amount, risk_tolerance, urgency, timeframe, timeframe_days
     - market: state.market_snapshot
     - intelligence: state.intelligence_report
     - prediction: state.price_forecast
     - warnings: state.warnings
     - components_available: track which agents succeeded

2. **Create DecisionMaker**:

   - Load config
   - Instantiate DecisionMaker

3. **Make Decision**:

   - Call `decision_maker.make_decision(request)`
   - Get DecisionResponse

4. **Update State**:

   - Return dict with updated fields:
     - `recommendation`: Convert DecisionResponse to dict (dataclass.asdict)
     - `decision_status`: "success" or "partial"
     - `processing_stage`: "decision_complete"
   - Preserve correlation_id and other state fields

5. **Error Handling**:

   - Try-except around decision making
   - On error:
     - Log error with correlation_id
     - Add error to state.errors
     - Set decision_status to "error"
     - Provide minimal fallback recommendation (conservative default)

6. **Logging**:

   - Log decision inputs (currency_pair, amount, risk_tolerance)
   - Log decision outputs (action, confidence)
   - Log execution time
   - Use correlation_id from state

**Outputs**: Dictionary with:

- `recommendation`: Full DecisionResponse as dict
- `decision_status`: Status string
- `processing_stage`: Updated stage

### 3. Update `src/agentic/graph.py`

**Purpose**: Add decision node to LangGraph workflow.

**Changes**:

1. **Import Decision Node**:
   ```python
   from src.agentic.nodes.decision import decision_node
   ```

2. **Add Node to Graph**:

   - Replace placeholder decision node with actual implementation
   - Node should execute after prediction node completes
   - Node name: "decision"

3. **Update Edges**:

   - Ensure prediction → decision edge exists
   - Ensure decision → supervisor_end edge exists

**Note**: The graph structure from Phase 0.3 already has placeholders; this just replaces the placeholder with real implementation.

### 4. `tests/decision/test_decision_maker.py`

**Purpose**: Unit tests for DecisionMaker orchestrator.

**Test Cases**:

**Utility-Based Decision Tests**:

- `test_decision_with_prediction`: Full prediction available → utility-based
- `test_conservative_event_blocking`: Conservative + nearby event → blocks convert_now
- `test_moderate_event_staging`: Moderate + nearby event → prefer staged
- `test_aggressive_allows_convert`: Aggressive + nearby event → allows convert_now
- `test_staged_generates_plan`: Action staged → creates StagedPlan
- `test_utility_scores_included`: Utility scores in response for transparency
- `test_expected_outcome_from_prediction`: Extract expected rates correctly

**Heuristic Fallback Tests**:

- `test_decision_without_prediction`: No prediction → heuristic fallback
- `test_low_confidence_prediction_triggers_heuristic`: Prediction confidence < threshold → heuristic
- `test_heuristic_has_lower_confidence`: Heuristic confidence < utility-based
- `test_heuristic_rationale_includes_method`: Rationale mentions heuristic method

**Integration Tests**:

- `test_full_decision_flow_convert_now`: End-to-end convert_now decision
- `test_full_decision_flow_staged`: End-to-end staged decision
- `test_full_decision_flow_wait`: End-to-end wait decision
- `test_partial_data_handles_gracefully`: Missing intelligence → still decides
- `test_warnings_propagated`: Warnings from request appear in response

**Edge Cases**:

- `test_missing_market_data`: No market data → neutral default
- `test_conflicting_signals`: Mixed signals → lower confidence
- `test_very_high_volatility`: High ATR → high risk, affects decision

**Fixtures**:

- `config`: DecisionConfig
- `decision_maker`: DecisionMaker instance

### 5. `tests/unit/test_agentic/test_nodes/test_decision.py`

**Purpose**: Unit tests for decision LangGraph node.

**Test Cases**:

- `test_decision_node_updates_state`: Verify state fields updated correctly
- `test_decision_node_with_all_data`: All upstream data available
- `test_decision_node_without_prediction`: Prediction missing, uses heuristic
- `test_decision_node_without_intelligence`: Intelligence missing, still works
- `test_decision_node_preserves_correlation_id`: correlation_id flows through
- `test_decision_node_error_handling`: Node doesn't crash on error
- `test_decision_node_logs_execution`: Verify logging occurs

**Fixtures**:

- `config`: Load config
- `base_state`: Minimal AgentState for testing

**Mock Data**: Create AgentState with various combinations of:

- Complete data (market + intelligence + prediction)
- Partial data (market only, market + intelligence)
- Edge cases (errors, missing required fields)

### 6. `tests/integration/test_decision_integration.py`

**Purpose**: Integration test for complete decision flow through LangGraph.

**Test Scenario**: Full workflow from initialized state through decision

**Test Case**: `test_complete_decision_workflow`

**Setup**:

- Initialize AgentState with user query
- Mock market_data_node to return market snapshot
- Mock market_intelligence_node to return intelligence report
- Mock prediction_node to return price forecast
- Execute decision_node

**Expected Behavior**:

- Decision node receives all upstream data
- Makes recommendation using utility model
- Returns complete DecisionResponse
- State properly updated with recommendation

**Validation**:

- Assert state.recommendation is not None
- Assert action in ["convert_now", "staged_conversion", "wait"]
- Assert 0.0 <= confidence <= 1.0
- Assert rationale is not empty
- Assert decision_status == "success"

**Test Case**: `test_decision_workflow_heuristic_fallback`

**Setup**: Same but prediction_node returns error

**Expected Behavior**:

- Decision falls back to heuristics
- Confidence is lower (0.3-0.6)
- Warning about heuristic fallback

### 7. `tests/integration/test_decision_scenarios.py`

**Purpose**: Test realistic decision scenarios end-to-end.

**Test Scenarios**:

**Scenario 1: Urgent Conversion Before Event**

- Urgency: urgent
- Event: Fed meeting tomorrow
- Prediction: Neutral
- Expected: CONVERT_NOW (urgency overrides event wait)

**Scenario 2: Conservative with Approaching Event**

- Risk tolerance: conservative
- Event: ECB meeting in 2 days
- Prediction: +0.3% improvement
- Expected: WAIT (conservative blocks near-event conversion)

**Scenario 3: Optimal Staging Opportunity**

- Timeframe: 7 days
- Prediction: +0.5% in 7 days
- Event: Jobs report in 3 days
- Expected: STAGED (3 tranches, avoid day 3)

**Scenario 4: High Volatility Wait**

- ATR: Very high (0.01)
- RSI: Neutral (50)
- Prediction: Slight positive
- Expected: WAIT or STAGED (risk aversion due to volatility)

**Scenario 5: Clear Uptrend Wait**

- Prediction: +0.8% in 7 days
- RSI: 45 (not overbought)
- Trend: Strong uptrend
- Expected: WAIT (capture predicted improvement)

Each scenario validates:

- Action makes sense given inputs
- Rationale explains reasoning
- Confidence is appropriate
- Staging plan (if applicable) is reasonable

## Validation

Manual end-to-end validation:

```python
from src.decision.decision_maker import DecisionMaker
from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest
from src.agentic.nodes.decision import decision_node
from src.agentic.state import initialize_state

# Test 1: Direct DecisionMaker
config = DecisionConfig.from_yaml()
decision_maker = DecisionMaker(config)

request = DecisionRequest(
    amount=5000,
    risk_tolerance="moderate",
    urgency="normal",
    timeframe="1_week",
    timeframe_days=7,
    market={
        "current_rate": 0.92,
        "indicators": {"rsi_14": 55, "atr_14": 0.005, "macd": {"value": 0.0003, "signal": 0.0002}},
        "regime": {"trend": "uptrend", "bias": "bullish"}
    },
    intelligence={
        "overall_bias": 5.0,
        "upcoming_events": [
            {"importance": "high", "days_until": 4, "event_name": "Fed Meeting"}
        ]
    },
    prediction={
        "status": "success",
        "predictions": {"7": {"mean_change_pct": 0.3, "confidence": 0.7}}
    }
)

response = decision_maker.make_decision(request)
print(f"Action: {response.action}")
print(f"Confidence: {response.confidence}")
print(f"Timeline: {response.timeline}")
print(f"Rationale:")
for r in response.rationale:
    print(f"  - {r}")

if response.staged_plan:
    print(f"\nStaged Plan: {response.staged_plan.num_tranches} tranches")
    for t in response.staged_plan.tranches:
        print(f"  Day {t.execute_day}: {t.percentage}%")

print(f"\nRisk: {response.risk_summary.risk_level}")
print(f"Cost: {response.cost_estimate.total_bps} bps")

# Test 2: Through LangGraph Node
state = initialize_state("Convert 5000 USD to EUR")
state["amount"] = 5000
state["currency_pair"] = "USD/EUR"
state["risk_tolerance"] = "moderate"
state["urgency"] = "normal"
state["timeframe"] = "1_week"
state["timeframe_days"] = 7
state["market_snapshot"] = request.market
state["intelligence_report"] = request.intelligence
state["price_forecast"] = request.prediction

updated_state = decision_node(state)
print(f"\n=== LangGraph Node Output ===")
print(f"Decision Status: {updated_state['decision_status']}")
print(f"Action: {updated_state['recommendation']['action']}")
print(f"Confidence: {updated_state['recommendation']['confidence']}")
```

## Success Criteria

- [ ] DecisionMaker orchestrates all components correctly
- [ ] Utility-based decision uses prediction when available
- [ ] Heuristic fallback activates when prediction unavailable/unreliable
- [ ] Hard constraints (event-gating) apply for conservative profile
- [ ] Staging plans generated when staged action selected
- [ ] Expected outcomes calculated from available data
- [ ] Confidence aggregated from multiple sources
- [ ] Rationale provides clear, actionable explanations
- [ ] Warnings propagated and new ones added appropriately
- [ ] Decision node integrates with LangGraph state correctly
- [ ] Error handling prevents crashes
- [ ] All unit tests pass with >80% coverage
- [ ] Integration tests demonstrate realistic scenarios
- [ ] Manual validation shows sensible decisions

## Key Design Decisions

1. **Prediction Reliability Check**: Threshold-based decision on utility vs heuristic
2. **Hard Constraints**: Only for conservative + nearby events
3. **Always Decide**: Never fail to provide recommendation, even with minimal data
4. **Transparency**: Include utility_scores and component_confidences in response
5. **Graceful Degradation**: Heuristic → Technical → Neutral fallback chain
6. **State Integration**: Clean mapping between LangGraph state and DecisionRequest
7. **Error Recovery**: Log and warn, but always return usable recommendation
8. **Comprehensive Rationale**: Top 3-5 reasons covering all decision factors

## Integration Points

- **Consumes**: Market Data Agent output, Market Intelligence Agent output, Prediction Agent output
- **Produces**: DecisionResponse for Supervisor Agent to format for user
- **Dependencies**: All Phase 3.1, 3.2, 3.3 components
- **LangGraph**: Integrated as "decision" node in Layer 3

## Files Summary

**Created**:

- `src/decision/decision_maker.py` - Main orchestrator
- `src/agentic/nodes/decision.py` - LangGraph node wrapper
- `tests/decision/test_decision_maker.py` - Orchestrator tests
- `tests/unit/test_agentic/test_nodes/test_decision.py` - Node tests
- `tests/integration/test_decision_integration.py` - Workflow integration test
- `tests/integration/test_decision_scenarios.py` - Realistic scenario tests

**Updated**:

- `src/agentic/graph.py` - Replace placeholder decision node

## Phase 3 Complete

After this phase, the Decision Engine Agent is fully operational and integrated into the LangGraph workflow. It can:

- Process inputs from all upstream agents
- Make intelligent recommendations using utility model or heuristics
- Generate staging plans when appropriate
- Provide confidence scores and clear rationale
- Handle errors and partial data gracefully
- Integrate seamlessly with LangGraph orchestration

### To-dos

- [ ] Implement DecisionMaker orchestrator in src/decision/decision_maker.py that coordinates all decision components
- [ ] Create decision_node for LangGraph in src/agentic/nodes/decision.py that wraps DecisionMaker
- [ ] Add utility-based decision path when prediction is available and reliable
- [ ] Add heuristic fallback path when prediction is unavailable or unreliable
- [ ] Implement hard constraints (event-gating) for conservative risk profile
- [ ] Add staging plan generation when staged action is selected
- [ ] Implement expected outcome calculation from available data sources
- [ ] Add confidence aggregation from multiple sources (market, intelligence, prediction)
- [ ] Write comprehensive unit tests for DecisionMaker (>80% coverage)
- [ ] Write unit tests for decision LangGraph node
- [ ] Write integration tests for complete decision workflow
- [ ] Write integration tests for realistic decision scenarios
- [ ] Update graph.py to replace placeholder decision node with real implementation