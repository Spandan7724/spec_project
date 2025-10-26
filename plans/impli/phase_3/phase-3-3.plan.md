<!-- f67714c1-a54f-4e8d-9617-16955f212afc 5b8e8f9e-2f19-4ba9-9251-14587a92666a -->
# Phase 3.3: Heuristic Fallbacks

## Overview

Implement heuristic-based fallback decision logic that operates when ML prediction models are unavailable, unreliable, or lack sufficient confidence. These fallbacks use technical indicators, event proximity, and momentum-based rules to generate reasonable recommendations.

## Reference

See `plans/decision-engine.plan.md` for fallback strategy (lines 7, 36).

## Purpose

Ensure the Decision Engine can always provide recommendations even when:

- Prediction agent fails or times out
- Model confidence is below threshold
- Insufficient historical data for predictions
- Market conditions are outside model training range

## Fallback Hierarchy

1. **Prediction Agent** (preferred) - ML-based forecasts
2. **Intelligence-Based** - Event proximity and policy bias
3. **Technical Indicators** - RSI, MACD, trend signals
4. **Neutral Default** - Conservative recommendation when all else fails

## Files to Create

### 1. `src/decision/heuristics.py`

**Purpose**: Provide rule-based decision logic as fallback when predictions unavailable.

**Class**: `HeuristicDecisionMaker`

**Constructor**: Takes `DecisionConfig`

**Main Method**: `make_heuristic_decision(request: DecisionRequest) -> Dict[str, Any]`

**Inputs**:

- `request`: Full DecisionRequest with market, intelligence (optional), user params

**Outputs**: Dictionary with:

- `action`: convert_now | staged_conversion | wait
- `confidence`: 0.0-1.0 (lower than model-based, typically 0.3-0.6)
- `rationale`: List of reasoning points
- `source`: "heuristic_fallback"
- `method`: Which heuristic was used

**Core Logic**:

**Priority 1: Event-Gating Rules** (if intelligence available)

- High-impact event within threshold days:
  - Conservative: WAIT (hard block)
  - Moderate: STAGED if timeframe allows, else WAIT
  - Aggressive: STAGED or CONVERT_NOW

- No high-impact events:
  - Proceed to momentum-based heuristics

**Priority 2: Momentum-Based Heuristics** (using technical indicators)

- **RSI-Based**:
  - RSI < 30 (oversold): Bullish → WAIT for reversion up
  - RSI > 70 (overbought): Bearish → CONVERT_NOW before drop
  - RSI 40-60 (neutral): Use MACD or trend

- **MACD-Based**:
  - Bullish crossover (MACD > signal): WAIT for upward momentum
  - Bearish crossover (MACD < signal): CONVERT_NOW before decline
  - No clear crossover: Use trend

- **Trend-Based** (from regime classifier):
  - Uptrend + bullish bias: WAIT for better rate
  - Downtrend + bearish bias: CONVERT_NOW to avoid worse rate
  - Sideways: STAGED to average out volatility

**Priority 3: Urgency Override**

- If urgency is "urgent":
  - Unless very negative signals (RSI > 75, bearish event tomorrow)
  - Default to CONVERT_NOW or STAGED (front-loaded)

**Priority 4: Neutral Default**

- When indicators are mixed or unavailable:
  - Conservative: WAIT (safer to delay)
  - Moderate: STAGED (diversify risk)
  - Aggressive: CONVERT_NOW (seize opportunity)

**Internal Methods**:

- `_check_event_gating(request: DecisionRequest) -> Optional[str]`
  - Returns action if event forces a decision, else None

- `_apply_rsi_heuristic(rsi: float, risk_tolerance: str) -> str`
  - Returns action based on RSI levels

- `_apply_macd_heuristic(macd: Dict, risk_tolerance: str) -> str`
  - Returns action based on MACD crossover

- `_apply_trend_heuristic(trend: str, bias: float, risk_tolerance: str) -> str`
  - Returns action based on trend direction and bias

- `_apply_urgency_override(action: str, urgency: str, signals: Dict) -> str`
  - Overrides action if urgency demands immediate action

- `_get_neutral_default(risk_tolerance: str) -> str`
  - Returns default action for risk profile

- `_calculate_heuristic_confidence(method: str, signal_strength: float) -> float`
  - Returns confidence based on heuristic method used
  - Event-based: 0.5-0.6
  - Technical: 0.4-0.5
  - Neutral: 0.3

- `_generate_heuristic_rationale(action: str, method: str, signals: Dict) -> List[str]`
  - Generates explanation for heuristic decision

**Confidence Calculation**:

Heuristic confidence is intentionally lower than model-based:

- Strong technical signals: 0.5
- Event-based: 0.55
- Mixed signals: 0.4
- Neutral default: 0.3

### 2. `src/decision/confidence_aggregator.py`

**Purpose**: Aggregate confidence from multiple sources (prediction, intelligence, market).

**Class**: `ConfidenceAggregator`

**Main Method**: `aggregate_confidence(request: DecisionRequest, utility_scores: Dict[str, float]) -> Dict[str, Any]`

**Inputs**:

- `request`: Decision request with all data
- `utility_scores`: Scores from utility model
- `is_heuristic`: Boolean indicating if fallback was used

**Outputs**: Dictionary with:

- `overall_confidence`: 0.0-1.0 float
- `component_confidences`: Dict with source-level confidences
- `penalties_applied`: List of penalty descriptions

**Component Confidences**:

1. **Market Data Confidence** (base 0.5):

   - +0.2 if data quality metrics are good (low dispersion)
   - +0.1 if indicators are coherent (RSI and MACD agree)
   - -0.1 if data is stale (>5 minutes old)

2. **Intelligence Confidence** (if available):

   - Base 0.4
   - +0.2 if high event clarity (importance and impact known)
   - +0.1 if multiple sources confirm bias
   - -0.1 if conflicting signals

3. **Prediction Confidence** (if available):

   - Use model's reported confidence directly
   - -0.1 if prediction is stale (>6 hours old)

4. **Heuristic Penalty**:

   - -0.2 if using heuristic fallback (less reliable than model)

**Aggregation Logic**:

- Weighted average of available components
- Market: 30% weight
- Intelligence: 20% weight
- Prediction: 50% weight (if available)
- If prediction unavailable: redistribute weight to market + intelligence

**Global Penalties**:

- Missing components: -5% per missing component
- Warnings in request: -10% per 2 warnings
- Stale data: -5%
- Low utility score spread: -10% (unclear which action is best)

**Final Confidence**: Clipped to [0.0, 1.0]

**Internal Methods**:

- `_calculate_market_confidence(market: Dict) -> float`
- `_calculate_intelligence_confidence(intelligence: Dict) -> float`
- `_calculate_prediction_confidence(prediction: Dict) -> float`
- `_apply_global_penalties(base_confidence: float, request: DecisionRequest) -> float`
- `_check_utility_spread(utility_scores: Dict) -> float`
  - Returns penalty if top 2 scores are too close (ambiguous)

### 3. `tests/decision/test_heuristics.py`

**Purpose**: Unit tests for HeuristicDecisionMaker.

**Test Cases**:

- `test_event_gating_conservative_blocks`: Conservative + event tomorrow → WAIT
- `test_event_gating_aggressive_allows`: Aggressive + event tomorrow → CONVERT_NOW or STAGED
- `test_rsi_oversold_wait`: RSI < 30 → WAIT for reversion
- `test_rsi_overbought_convert`: RSI > 70 → CONVERT_NOW
- `test_macd_bullish_crossover`: MACD > signal → WAIT
- `test_macd_bearish_crossover`: MACD < signal → CONVERT_NOW
- `test_uptrend_wait`: Uptrend + bullish → WAIT
- `test_downtrend_convert`: Downtrend + bearish → CONVERT_NOW
- `test_urgent_override`: Urgent urgency → CONVERT_NOW unless very negative
- `test_neutral_default_by_profile`: Different defaults for conservative/moderate/aggressive
- `test_heuristic_confidence_lower`: Heuristic confidence < model confidence
- `test_mixed_signals_staged`: Conflicting signals → STAGED

**Fixtures**:

- `config`: DecisionConfig
- `heuristic_maker`: HeuristicDecisionMaker instance

**Mock Data**: DecisionRequest with various:

- Technical indicator values (RSI, MACD)
- Event proximities
- Risk tolerance levels
- Urgency levels

### 4. `tests/decision/test_confidence_aggregator.py`

**Purpose**: Unit tests for ConfidenceAggregator.

**Test Cases**:

- `test_all_components_available`: High confidence when all data present
- `test_missing_prediction_penalty`: Confidence drops without prediction
- `test_stale_data_penalty`: Old data reduces confidence
- `test_low_utility_spread_penalty`: Close utility scores reduce confidence
- `test_warning_penalty`: Multiple warnings reduce confidence
- `test_heuristic_fallback_penalty`: Using heuristics lowers confidence
- `test_component_weights`: Verify weighted average calculation
- `test_confidence_clipping`: Confidence stays in [0, 1]

**Fixture**: `aggregator`: ConfidenceAggregator instance

### 5. `tests/decision/test_heuristic_integration.py`

**Purpose**: Integration test for heuristic decision flow.

**Test Scenario**: Full decision using only heuristics (no prediction)

**Test Case**: `test_heuristic_decision_without_prediction`

**Setup**:

- Create DecisionRequest with:
  - No prediction data (prediction=None)
  - Market data with clear technical signals (RSI=65, MACD bullish)
  - Intelligence with event in 5 days
  - Moderate risk tolerance, normal urgency

**Expected Behavior**:

- Should use heuristic fallback
- Should generate action based on technical indicators
- Confidence should be 0.4-0.5 (lower than model-based)
- Rationale should explain heuristic method used

**Validation**:

- Assert action in ["convert_now", "staged_conversion", "wait"]
- Assert 0.3 <= confidence <= 0.6
- Assert "heuristic" in source or method
- Assert rationale is not empty

## Validation

Manual validation script:

```python
from src.decision.heuristics import HeuristicDecisionMaker
from src.decision.confidence_aggregator import ConfidenceAggregator
from src.decision.config import DecisionConfig
from src.decision.models import DecisionRequest

# Load config
config = DecisionConfig.from_yaml()
heuristic_maker = HeuristicDecisionMaker(config)
confidence_agg = ConfidenceAggregator()

# Test scenario: No prediction, use heuristics
request = DecisionRequest(
    amount=5000,
    risk_tolerance="moderate",
    urgency="normal",
    timeframe="1_week",
    timeframe_days=7,
    market={
        "indicators": {
            "rsi_14": 65,  # Slightly overbought
            "macd": {"value": 0.0005, "signal": 0.0003},  # Bullish
            "atr_14": 0.005
        },
        "regime": {"trend": "uptrend", "bias": "bullish"}
    },
    intelligence={
        "overall_bias": 5.0,
        "upcoming_events": [
            {"importance": "high", "days_until": 5, "event_name": "FOMC"}
        ]
    },
    prediction=None,  # Force heuristic fallback
    components_available={"market": True, "intelligence": True, "prediction": False}
)

# Make heuristic decision
decision = heuristic_maker.make_heuristic_decision(request)
print(f"Action: {decision['action']}")
print(f"Confidence: {decision['confidence']}")
print(f"Method: {decision['method']}")
print(f"Rationale: {decision['rationale']}")

# Aggregate confidence
utility_scores = {"convert_now": 0.3, "staged": 0.45, "wait": 0.5}
confidence_data = confidence_agg.aggregate_confidence(request, utility_scores)
print(f"\nOverall confidence: {confidence_data['overall_confidence']}")
print(f"Component confidences: {confidence_data['component_confidences']}")
print(f"Penalties: {confidence_data['penalties_applied']}")
```

## Success Criteria

- [ ] Event-gating rules block or warn based on risk tolerance
- [ ] RSI-based heuristics provide reasonable recommendations
- [ ] MACD-based heuristics detect momentum correctly
- [ ] Trend-based heuristics align with regime
- [ ] Urgency overrides work appropriately
- [ ] Neutral defaults make sense for each risk profile
- [ ] Heuristic confidence is lower than model-based (0.3-0.6)
- [ ] Confidence aggregator weights components correctly
- [ ] Global penalties reduce confidence appropriately
- [ ] All unit tests pass with >80% coverage
- [ ] Integration test demonstrates full heuristic flow

## Key Design Decisions

1. **Fallback Hierarchy**: Prediction → Intelligence → Technical → Neutral
2. **Event Gating**: Risk-profile dependent (hard block for conservative)
3. **Technical Signals**: RSI + MACD + Trend for comprehensive view
4. **Urgency Override**: Can force CONVERT_NOW unless very negative
5. **Lower Confidence**: Heuristics are 0.3-0.6 vs model's 0.5-0.9
6. **Transparency**: Always indicate when heuristic fallback is used
7. **Conservative Bias**: When in doubt, favor safer action (WAIT or STAGED)

## Integration Points

- Used by Decision Maker (Phase 3.4) when prediction unavailable or unreliable
- Confidence aggregator used for all decisions (model-based and heuristic)
- Heuristic decisions marked with lower confidence to signal uncertainty
- Rationale explains which heuristic method was applied

## Edge Cases Handled

- No prediction data → Use intelligence + technical
- No intelligence data → Use only technical indicators
- Conflicting technical signals → Default to STAGED
- Very urgent + negative signals → STAGED (compromise)
- All data missing → Neutral default based on risk profile

### To-dos

- [ ] Implement HeuristicDecisionMaker in src/decision/heuristics.py with event-gating and momentum-based rules
- [ ] Implement ConfidenceAggregator in src/decision/confidence_aggregator.py for multi-source confidence calculation
- [ ] Add event-gating rules based on risk tolerance (hard block for conservative, soft penalty for moderate)
- [ ] Implement RSI-based heuristics for oversold/overbought signals
- [ ] Implement MACD-based momentum heuristics for bullish/bearish crossovers
- [ ] Implement trend-based heuristics using regime classifier output
- [ ] Add urgency override logic for immediate action requirements
- [ ] Write comprehensive unit tests for heuristic decision logic (>80% coverage)
- [ ] Write unit tests for confidence aggregation and penalty calculations
- [ ] Write integration test for full heuristic decision flow without prediction