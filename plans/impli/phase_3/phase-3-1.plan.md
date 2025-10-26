<!-- f67714c1-a54f-4e8d-9617-16955f212afc 23c8eef0-9d94-4d81-b624-6d13e1e29501 -->
# Phase 3.1: Decision Model Core

## Overview

Build the foundational components of the Decision Engine: data contracts, configuration system, utility scoring model, and risk calculation. This phase creates the core decision-making logic that will evaluate three candidate actions (convert_now, staged_conversion, wait) using a multi-criteria utility model.

## Reference

See `plans/decision-engine.plan.md` for complete architecture details and design decisions.

## Files to Create

### 1. `src/decision/__init__.py`

Empty init file for the decision module.

### 2. `src/decision/models.py`

**Purpose**: Define all data contracts for decision engine inputs and outputs.

**Data Classes to Define**:

- `DecisionRequest`: Input to decision engine
  - User constraints: amount, risk_tolerance, urgency, timeframe, timeframe_days
  - Market snapshot (dict from Market Data Agent)
  - Intelligence report (optional dict from Market Intelligence Agent)
  - Prediction (optional dict from Prediction Agent)
  - Costs: spread_bps, fee_bps
  - System health: components_available, warnings

- `TrancheSpec`: Single tranche in staged conversion
  - tranche_number, percentage, execute_day, rationale

- `StagedPlan`: Complete staged conversion plan
  - num_tranches, tranches list, spacing_days, total_extra_cost_bps, benefit

- `ExpectedOutcome`: Expected outcome of decision
  - expected_rate, range_low, range_high, expected_improvement_bps

- `RiskSummary`: Risk assessment
  - risk_level (low/moderate/high), realized_vol_30d, var_95, event_risk, event_details

- `CostEstimate`: Cost breakdown
  - spread_bps, fee_bps, total_bps, staged_multiplier

- `DecisionResponse`: Output of decision engine
  - action (convert_now/staged_conversion/wait), confidence, timeline
  - staged_plan (optional), expected_outcome, risk_summary, cost_estimate
  - rationale (list of strings), warnings, utility_scores, component_confidences, timestamp

**Notes**: Use dataclasses with proper types and defaults. All should be JSON-serializable.

### 3. `src/decision/config.py`

**Purpose**: Load and manage decision engine configuration from config.yaml.

**Classes to Define**:

- `UtilityWeights`: profit, risk, cost, urgency weights
- `RiskProfile`: weights, min_improvement_bps, event_proximity_threshold_days, volatility_penalty_multiplier
- `DecisionThresholds`: convert_now_min_utility, staged_min_timeframe_days, wait_event_proximity_days, min_model_confidence, max_prediction_age_hours
- `StagingConfig`: max_tranches, min_spacing_days, short_timeframe_tranches, long_timeframe_tranches, urgent/normal/flexible patterns
- `CostConfig`: default_spread_bps, default_fee_bps, staging_cost_multiplier
- `DecisionConfig`: Main config with risk_profiles dict, thresholds, staging, costs
  - `from_yaml()` class method to load from config.yaml

**Logic**: Parse risk profiles (conservative/moderate/aggressive) from YAML, create typed config objects.

### 4. Update `config.yaml`

**Purpose**: Add decision engine configuration section.

**Structure** (as specified in decision-engine.plan.md lines 180-233):

```yaml
decision:
  risk_profiles:
    conservative: {weights: {...}, min_improvement_bps, event_proximity_threshold_days, volatility_penalty_multiplier}
    moderate: {weights: {...}, ...}
    aggressive: {weights: {...}, ...}
  thresholds: {convert_now_min_utility, staged_min_timeframe_days, ...}
  staging: {max_tranches, patterns, ...}
  costs: {default_spread_bps, default_fee_bps, staging_cost_multiplier}
```

### 5. `src/decision/utility_scorer.py`

**Purpose**: Score three candidate actions (convert_now, staged, wait) using utility model.

**Class**: `UtilityScorer`

**Constructor**: Takes `DecisionConfig`

**Main Method**: `score_actions(request: DecisionRequest) -> Dict[str, float]`

- Returns dict with utility scores for each action
- Calls internal scoring methods for each action

**Internal Methods**:

- `_score_convert_now()`: Calculate utility for immediate conversion
  - Expected improvement = 0 (no waiting)
  - Apply risk penalty, cost, urgency fit

- `_score_staged()`: Calculate utility for staged conversion
  - Expected improvement = average (50% of full improvement)
  - Risk penalty reduced by 40% (staging reduces risk)
  - Cost increased by staging multiplier

- `_score_wait()`: Calculate utility for waiting
  - Full expected improvement
  - Full risk penalty
  - Standard cost

- `_get_expected_improvement()`: Extract from prediction → intelligence → technical → neutral
  - Priority 1: Prediction agent (mean_change_pct for timeframe)
  - Priority 2: Intelligence bias (heuristic conversion)
  - Priority 3: Technical indicators (RSI-based)
  - Fallback: 0.0

- `_calculate_risk_penalty()`: Volatility + event proximity
  - Volatility component: ATR * 100 * profile.volatility_penalty_multiplier
  - Event component: Exponential penalty as high-impact event approaches

- `_get_transaction_cost()`: spread_bps + fee_bps (convert to decimal)

- `_get_urgency_fit()`: Action-urgency alignment bonus/penalty
  - Urgent: favors convert_now (+0.5), penalizes wait (-0.3)
  - Normal: balanced (all ~0.2-0.3)
  - Flexible: favors wait (+0.4), penalizes convert_now (-0.1)

**Utility Formula**: `profit*improvement - risk*penalty - cost*transaction_cost + urgency*urgency_fit`

### 6. `src/decision/risk_calculator.py`

**Purpose**: Calculate risk metrics and summaries.

**Class**: `RiskCalculator`

**Main Method**: `calculate_risk_summary(market, intelligence) -> RiskSummary`

**Logic**:

- Extract volatility from market indicators (ATR)
- Annualize to realized_vol_30d (ATR * 100 * 16)
- Calculate VaR 95% (1.65 * volatility)
- Assess event risk from intelligence:
  - High: event within 1 day
  - Moderate: event within 3 days
  - Low: event beyond 3 days
  - None: no high-impact events
- Determine overall risk level:
  - High: event_risk high/moderate OR vol > 15%
  - Moderate: vol > 10%
  - Low: otherwise

**Output**: RiskSummary with all metrics populated

### 7. `tests/decision/__init__.py`

Empty init file for test module.

### 8. `tests/decision/test_utility_scorer.py`

**Purpose**: Unit tests for UtilityScorer.

**Test Cases**:

- `test_score_actions_basic`: Verify all three actions are scored
- `test_urgent_favors_convert_now`: Urgent urgency should score convert_now higher
- `test_flexible_favors_wait`: Flexible urgency should score wait higher
- `test_expected_improvement_from_prediction`: Extract improvement from prediction correctly
- `test_expected_improvement_fallback`: Fall back to intelligence then technical
- `test_risk_penalty_with_event`: Risk penalty increases near high-impact event
- `test_risk_penalty_by_profile`: Conservative has higher penalty multiplier
- `test_staging_reduces_risk`: Staged action has lower risk penalty

**Fixtures**: config (from yaml), scorer (UtilityScorer instance)

### 9. `tests/decision/test_risk_calculator.py`

**Purpose**: Unit tests for RiskCalculator.

**Test Cases**:

- `test_calculate_risk_summary_low_risk`: Low volatility, no events → low risk
- `test_calculate_risk_summary_high_risk_event`: High-impact event → high risk
- `test_calculate_risk_summary_high_volatility`: High ATR → high risk
- `test_event_risk_classification`: Verify event_risk levels by days_until
- `test_var_calculation`: Verify VaR 95% calculation

**Fixture**: calculator (RiskCalculator instance)

## Validation

Run manual validation script:

```python
from src.decision.config import DecisionConfig
from src.decision.utility_scorer import UtilityScorer
from src.decision.risk_calculator import RiskCalculator
from src.decision.models import DecisionRequest

# Load config and verify risk profiles loaded
config = DecisionConfig.from_yaml()
print(f"Risk profiles: {list(config.risk_profiles.keys())}")

# Create test request
request = DecisionRequest(
    amount=5000,
    risk_tolerance="moderate",
    urgency="normal",
    timeframe="1_week",
    timeframe_days=7,
    market={"indicators": {"rsi_14": 55, "atr_14": 0.005}},
    intelligence={"overall_bias": 5.0, "upcoming_events": []},
    prediction={"status": "success", "predictions": {"7": {"mean_change_pct": 0.3}}}
)

# Score actions and verify sensible results
scorer = UtilityScorer(config)
scores = scorer.score_actions(request)
print(f"Utility scores: {scores}")
print(f"Best action: {max(scores, key=scores.get)}")

# Calculate risk
risk_calc = RiskCalculator()
risk_summary = risk_calc.calculate_risk_summary(request.market, request.intelligence)
print(f"Risk level: {risk_summary.risk_level}")
```

## Success Criteria

- [ ] All data contracts defined with proper types
- [ ] Configuration loads from config.yaml successfully
- [ ] Utility scorer produces scores for all three actions
- [ ] Urgency parameter influences action scores correctly
- [ ] Risk calculator assesses volatility and event risk
- [ ] Event proximity increases risk penalty exponentially
- [ ] All unit tests pass with >80% coverage
- [ ] Manual validation shows sensible decision logic

## Key Design Decisions

1. **Utility Model**: Multi-criteria with configurable weights per risk profile
2. **Fallback Hierarchy**: Prediction → Intelligence → Technical → Neutral
3. **Event Gating**: Soft penalties via risk calculation (hard blocks come in Phase 3.4)
4. **Cost Model**: Basis points converted to decimals
5. **Urgency Fit**: Explicit bonus/penalty map for action-urgency alignment
6. **Configuration-Driven**: All thresholds and weights in config.yaml for easy tuning

### To-dos

- [ ] Create data contracts in src/decision/models.py for DecisionRequest, TrancheSpec, StagedPlan, ExpectedOutcome, RiskSummary, CostEstimate, DecisionResponse
- [ ] Implement DecisionConfig in src/decision/config.py with risk profile configuration loading from YAML
- [ ] Add decision engine section to config.yaml with risk profiles, thresholds, staging, and costs
- [ ] Implement UtilityScorer in src/decision/utility_scorer.py with action scoring logic (convert_now, staged, wait)
- [ ] Implement RiskCalculator in src/decision/risk_calculator.py for volatility and event risk assessment
- [ ] Write comprehensive unit tests for utility scorer (>80% coverage)
- [ ] Write comprehensive unit tests for risk calculator (>80% coverage)
- [ ] Manually validate utility scoring with realistic scenarios