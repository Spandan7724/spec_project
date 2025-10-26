<!-- f67714c1-a54f-4e8d-9617-16955f212afc 50d59013-d5c5-4288-bb03-97eb2a41cf4d -->
# Phase 3.2: Staging Algorithm

## Overview

Implement the staging planner that creates multi-tranche conversion plans. The planner determines how many tranches to use, how to size them based on urgency, when to execute each tranche, and how to avoid high-impact economic events.

## Reference

See `plans/decision-engine.plan.md` for staging logic and tranche patterns (lines 23-29, 259-276).

## Files to Create

### 1. `src/decision/staging_planner.py`

**Purpose**: Generate multi-tranche staged conversion plans.

**Class**: `StagingPlanner`

**Constructor**: Takes `StagingConfig` (from DecisionConfig)

**Main Method**: `create_staged_plan(request: DecisionRequest) -> StagedPlan`

**Inputs**:

- `request.amount`: Total amount to convert
- `request.timeframe_days`: Available time window
- `request.urgency`: urgent/normal/flexible
- `request.intelligence`: Optional intelligence with upcoming events

**Outputs**: `StagedPlan` with:

- `num_tranches`: Number of tranches (2 or 3)
- `tranches`: List of `TrancheSpec` objects
- `spacing_days`: Days between tranches
- `total_extra_cost_bps`: Additional cost vs single conversion
- `benefit`: Explanation of staging benefit

**Core Logic**:

1. **Determine Tranche Count**:

   - ≤5 days: 2 tranches
   - 6-10 days: 3 tranches
   - >10 days: 3 tranches (max)

2. **Calculate Base Spacing**:

   - `spacing_days = timeframe_days / num_tranches`
   - Minimum spacing from config (default 1 day)

3. **Apply Sizing Pattern by Urgency**:

   - Urgent (2-way): [60%, 40%] - front-loaded
   - Urgent (3-way): [50%, 30%, 20%] - front-loaded
   - Normal: Equal split [50%, 50%] or [33%, 33%, 34%]
   - Flexible: Equal split (same as normal)

4. **Calculate Execution Days**:

   - Day 0, Day spacing, Day 2*spacing, etc.
   - Round to whole days

5. **Avoid High-Impact Events** (if intelligence available):

   - Identify high-impact events within timeframe
   - If tranche falls within 0.5 days of event:
     - Shift tranche to day after event (+1 day)
     - OR reduce to fewer tranches if too constrained
   - Never schedule within 12 hours of high-impact event

6. **Generate Rationale** for each tranche:

   - "Initial conversion capturing {pct}%"
   - "Second tranche after {event_name}" (if event-aware)
   - "Final tranche at {days} days"

7. **Calculate Extra Cost**:

   - `extra_cost_bps = (num_tranches - 1) * spread_bps * cost_multiplier`
   - Use staging_cost_multiplier from config

8. **Generate Benefit String**:

   - "Reduces risk by {pct}% through diversification"
   - "Avoids concentration risk around {event_name}"
   - "Balances execution with {urgency} timeline"

**Internal Methods**:

- `_determine_tranche_count(timeframe_days: int) -> int`
  - Returns 2 or 3 based on timeframe

- `_get_sizing_pattern(urgency: str, num_tranches: int) -> List[float]`
  - Returns percentage list based on urgency and count

- `_calculate_execution_schedule(num_tranches: int, timeframe_days: int) -> List[int]`
  - Returns list of execution day offsets

- `_adjust_for_events(execution_days: List[int], events: List[Dict]) -> List[int]`
  - Shifts tranches away from high-impact events
  - Returns adjusted execution days

- `_generate_tranche_specs(execution_days: List[int], percentages: List[float], events: List[Dict]) -> List[TrancheSpec]`
  - Creates TrancheSpec objects with rationale

- `_calculate_extra_cost(num_tranches: int, spread_bps: float) -> float`
  - Calculates additional cost of staging

**Edge Cases**:

- **Multiple events cluster**: Increase spacing between tranches
- **Event on day 2 of 3-day window**: Either shift tranches or reduce to 1 tranche (convert_now)
- **Very short timeframe (<3 days)**: Default to 2 tranches only
- **No intelligence data**: Skip event avoidance, use simple equal spacing

### 2. `src/decision/cost_calculator.py`

**Purpose**: Calculate transaction costs for single and staged conversions.

**Class**: `CostCalculator`

**Constructor**: Takes `CostConfig` (from DecisionConfig)

**Main Method**: `calculate_cost_estimate(request: DecisionRequest, is_staged: bool) -> CostEstimate`

**Inputs**:

- `spread_bps`: From request or config default
- `fee_bps`: From request or config default
- `is_staged`: Boolean indicating if staged conversion
- `num_tranches`: Number of tranches if staged

**Outputs**: `CostEstimate` with:

- `spread_bps`: Spread cost
- `fee_bps`: Fee cost
- `total_bps`: Total per conversion
- `staged_multiplier`: Cost multiplier if staged (>1.0)

**Logic**:

- Get spread (request.spread_bps or config.default_spread_bps)
- Get fee (request.fee_bps or config.default_fee_bps)
- Calculate base total: spread + fee
- If staged: apply staging_cost_multiplier (typically 1.2x)
- Return CostEstimate with all fields

**Note**: Staging multiplier represents:

- Multiple conversion events (more spread hits)
- Slightly higher operational cost
- But benefits from risk reduction

### 3. `tests/decision/test_staging_planner.py`

**Purpose**: Unit tests for StagingPlanner.

**Test Cases**:

- `test_short_timeframe_two_tranches`: ≤5 days → 2 tranches
- `test_long_timeframe_three_tranches`: >5 days → 3 tranches
- `test_urgent_front_loaded`: Urgent urgency → [60%, 40%] or [50%, 30%, 20%]
- `test_normal_equal_split`: Normal urgency → equal percentages
- `test_flexible_equal_split`: Flexible urgency → equal percentages
- `test_avoid_high_impact_event`: Tranche shifts away from event
- `test_multiple_events_increase_spacing`: Clustered events → wider spacing
- `test_very_constrained_reduce_tranches`: Too many events → reduce to fewer tranches
- `test_no_intelligence_simple_spacing`: No intelligence → equal spacing without event avoidance
- `test_extra_cost_calculation`: Verify staging cost calculation
- `test_tranche_rationale_generation`: Verify rationale strings are meaningful

**Fixtures**:

- `staging_config`: StagingConfig from config
- `planner`: StagingPlanner instance

**Mock Data**: Create DecisionRequest objects with various:

- Timeframes (3, 5, 7, 10 days)
- Urgency levels (urgent, normal, flexible)
- Intelligence with/without events
- Events at different proximities

### 4. `tests/decision/test_cost_calculator.py`

**Purpose**: Unit tests for CostCalculator.

**Test Cases**:

- `test_single_conversion_cost`: No staging multiplier
- `test_staged_conversion_cost`: Apply staging multiplier
- `test_use_default_spread`: Fall back to config default
- `test_use_request_spread`: Use provided spread_bps
- `test_zero_fee`: Fee defaults to 0
- `test_custom_fee`: Use provided fee_bps

**Fixture**: `calculator`: CostCalculator instance with config

### 5. `tests/decision/test_staging_integration.py`

**Purpose**: Integration test for staging with events.

**Test Scenario**: Full staging plan generation with realistic data

**Test Case**: `test_full_staging_flow_with_events`

**Setup**:

- Create DecisionRequest with:
  - 7-day timeframe
  - Moderate risk tolerance
  - Normal urgency
  - Intelligence with Fed meeting in 3 days (high impact)
  - Market data with ATR

**Expected Behavior**:

- Should create 3-tranche plan
- Should avoid scheduling tranche on day 3
- Should have reasonable spacing
- Cost estimate should include staging multiplier
- Rationale should mention event avoidance

**Validation**:

- Assert num_tranches == 3
- Assert no tranche on day 3 (±0.5 days)
- Assert tranches sum to 100%
- Assert extra_cost_bps > 0
- Assert benefit mentions event or risk reduction

## Validation

Manual validation script:

```python
from src.decision.staging_planner import StagingPlanner
from src.decision.cost_calculator import CostCalculator
from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest

# Load config
config = DecisionConfig.from_yaml()
planner = StagingPlanner(config.staging)
cost_calc = CostCalculator(config.costs)

# Test scenario: 7-day timeframe with Fed meeting in 3 days
request = DecisionRequest(
    amount=5000,
    risk_tolerance="moderate",
    urgency="normal",
    timeframe="1_week",
    timeframe_days=7,
    market={"indicators": {"atr_14": 0.005}},
    intelligence={
        "upcoming_events": [
            {"importance": "high", "days_until": 3, "event_name": "Fed Meeting"}
        ]
    },
    spread_bps=5.0
)

# Generate staging plan
plan = planner.create_staged_plan(request)
print(f"Tranches: {plan.num_tranches}")
print(f"Spacing: {plan.spacing_days} days")
for tranche in plan.tranches:
    print(f"  Day {tranche.execute_day}: {tranche.percentage}% - {tranche.rationale}")
print(f"Extra cost: {plan.total_extra_cost_bps} bps")
print(f"Benefit: {plan.benefit}")

# Calculate costs
cost_single = cost_calc.calculate_cost_estimate(request, is_staged=False)
cost_staged = cost_calc.calculate_cost_estimate(request, is_staged=True)
print(f"\nSingle conversion: {cost_single.total_bps} bps")
print(f"Staged conversion: {cost_staged.total_bps} bps (multiplier: {cost_staged.staged_multiplier}x)")
```

## Success Criteria

- [ ] Staging planner creates 2 or 3 tranches based on timeframe
- [ ] Urgency parameter influences sizing patterns correctly
- [ ] Tranches avoid high-impact events when intelligence available
- [ ] Edge cases handled (multiple events, very short timeframe)
- [ ] Cost calculator applies staging multiplier correctly
- [ ] All percentages sum to 100%
- [ ] Execution days are valid (within timeframe)
- [ ] Rationale strings are meaningful and informative
- [ ] All unit tests pass with >80% coverage
- [ ] Integration test demonstrates realistic scenario

## Key Design Decisions

1. **Tranche Count**: Simple rule-based (≤5 days: 2, >5 days: 3)
2. **Sizing Patterns**: Urgent front-loads, normal/flexible equal split
3. **Event Avoidance**: 0.5-day buffer around high-impact events
4. **Edge Case**: Too constrained → reduce tranches or recommend convert_now
5. **Cost Model**: Staging adds ~20% cost but reduces risk by ~40%
6. **Spacing**: Equal by default, adjusted for events
7. **Rationale**: Auto-generated explanatory text for each tranche

## Integration Points

- Used by Decision Maker (Phase 3.4) when `staged_conversion` action is selected
- Reads from `DecisionRequest.intelligence` for event data
- Returns `StagedPlan` included in `DecisionResponse`
- Cost calculator used by both utility scorer and decision maker

### To-dos

- [ ] Implement StagingPlanner in src/decision/staging_planner.py with tranche logic and event avoidance
- [ ] Implement CostCalculator in src/decision/cost_calculator.py for spread and fee calculations
- [ ] Add tranche count determination based on timeframe (2 for ≤5 days, 3 for >5 days)
- [ ] Implement urgency-based sizing patterns (front-loaded for urgent, equal split for normal/flexible)
- [ ] Add event avoidance logic with 0.5-day buffer around high-impact events
- [ ] Write comprehensive unit tests for staging planner (>80% coverage)
- [ ] Write unit tests for cost calculator
- [ ] Write integration test for full staging flow with realistic event scenario