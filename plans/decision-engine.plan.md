# Decision Engine Agent Implementation Plan

## Overview

The Decision Engine is the final decision-maker that converts all upstream analysis into a single, actionable recommendation. It balances expected returns, risk, costs, and user constraints to decide whether to convert now, wait, or stage the conversion across multiple tranches.

## Purpose

**Primary Goal**: Turn signals + runtime constraints → actionable decision (convert_now | staged_conversion | wait)

**Key Responsibilities**:

- Synthesize signals from Market, Economic, and Prediction agents
- Apply user's risk tolerance, urgency, and timeframe constraints
- Score candidate actions via utility model
- Generate staging plans when appropriate
- Provide confidence scores and clear rationale
- Degrade gracefully when data is partial/missing

## Design Decisions (Locked In)

1. **Expected Improvement**: Percentage change in bps (0.5% = 50 bps) - consistent with Prediction
2. **Tranche Patterns**: 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - ≤5 days: 2 tranches
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - 6-10 days: 3 tranches  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Urgent: 60/40 (2-way) or 50/30/20 (3-way)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Normal: Equal split
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Flexible: Equal split (allows more time)

3. **Event Gating**: Hybrid - Hard block convert_now for conservative, soft penalty for moderate, allow for aggressive
4. **Costs**: User can provide fee_bps, defaults to 0. Spread defaults to 5 bps
5. **Confidence**: Keep as 0.0-1.0 float in response
6. **State Integration**: Build fresh decision module, ignore old code
7. **Fallback**: Intelligence → Technical → Neutral (with error handling)
8. **Urgency Default**: "normal" if not specified
9. **Edge Cases**: Smart staging (shift around events, reduce if constrained)
10. **Weights**: Config defaults, tune after dry-run

## Architecture

### Core Components

```
src/decision/
├── __init__.py
├── config.py                      # Decision model configuration
├── models.py                      # Data contracts
├── utility_scorer.py              # Utility-based decision scoring
├── staging_planner.py             # Multi-tranche staging logic
├── confidence_aggregator.py       # Confidence composition
├── cost_calculator.py             # Spread + fee calculations
└── decision_maker.py              # Main decision orchestrator

tests/decision/
├── test_utility_scorer.py
├── test_staging_planner.py
├── test_confidence_aggregator.py
└── test_decision_maker.py
```

## Data Contracts

### File: `src/decision/models.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class DecisionRequest:
    """Input to decision engine"""
    # User constraints
    amount: float
    risk_tolerance: str  # conservative | moderate | aggressive
    urgency: str  # urgent | normal | flexible
    timeframe: str  # immediate | 1_day | 1_week | 1_month
    timeframe_days: int
    
    # Market snapshot
    market: Dict[str, Any]  # From MarketAnalysis
    
    # Intelligence (optional)
    intelligence: Optional[Dict[str, Any]] = None  # From EconomicAnalysis
    
    # Prediction (optional)
    prediction: Optional[Dict[str, Any]] = None  # From PredictionAgent
    
    # Costs
    spread_bps: Optional[float] = None
    fee_bps: Optional[float] = 0.0
    
    # System health
    components_available: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class TrancheSpec:
    """Single tranche in staged conversion"""
    tranche_number: int
    percentage: float  # 0-100
    execute_day: int  # Day offset from now
    rationale: str

@dataclass
class StagedPlan:
    """Staged conversion plan"""
    num_tranches: int
    tranches: List[TrancheSpec]
    spacing_days: int
    total_extra_cost_bps: float
    benefit: str

@dataclass
class ExpectedOutcome:
    """Expected outcome of decision"""
    expected_rate: Optional[float] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    expected_improvement_bps: Optional[float] = None

@dataclass
class RiskSummary:
    """Risk assessment"""
    risk_level: str  # low | moderate | high
    realized_vol_30d: Optional[float] = None
    var_95: Optional[float] = None
    event_risk: str  # none | low | moderate | high
    event_details: Optional[str] = None

@dataclass
class CostEstimate:
    """Cost breakdown"""
    spread_bps: float
    fee_bps: float
    total_bps: float
    staged_multiplier: float = 1.0  # Cost multiplier if staged

@dataclass
class DecisionResponse:
    """Output of decision engine"""
    # Core decision
    action: str  # convert_now | staged_conversion | wait
    confidence: float  # 0.0-1.0
    timeline: str  # Human-readable timeline
    
    # Plans
    staged_plan: Optional[StagedPlan] = None
    
    # Outcomes
    expected_outcome: ExpectedOutcome = field(default_factory=ExpectedOutcome)
    risk_summary: RiskSummary = field(default_factory=RiskSummary)
    cost_estimate: CostEstimate = field(default_factory=CostEstimate)
    
    # Explanation
    rationale: List[str] = field(default_factory=list)  # Top 3-5 reasons
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    utility_scores: Dict[str, float] = field(default_factory=dict)
    component_confidences: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
```

## Configuration

### File: `src/decision/config.py`

Risk tolerance profiles with utility weights, thresholds, and staging configuration.

Key configs:

- **Utility Weights**: profit, risk, cost, urgency for each risk profile
- **Thresholds**: min improvement, event proximity, quality gates
- **Staging**: tranche counts, spacing rules, sizing patterns

### Update: `config.yaml`

```yaml
decision:
  risk_profiles:
    conservative:
      weights:
        profit: 0.8
        risk: 1.2
        cost: 0.4
        urgency: 0.3
      min_improvement_bps: 10.0
      event_proximity_threshold_days: 2.0
      volatility_penalty_multiplier: 1.5
    
    moderate:
      weights:
        profit: 1.0
        risk: 0.8
        cost: 0.3
        urgency: 0.4
      min_improvement_bps: 5.0
      event_proximity_threshold_days: 1.5
      volatility_penalty_multiplier: 1.0
    
    aggressive:
      weights:
        profit: 1.2
        risk: 0.5
        cost: 0.2
        urgency: 0.5
      min_improvement_bps: 3.0
      event_proximity_threshold_days: 0.5
      volatility_penalty_multiplier: 0.7
  
  thresholds:
    convert_now_min_utility: 0.3
    staged_min_timeframe_days: 3
    wait_event_proximity_days: 1.5
    min_model_confidence: 0.4
    max_prediction_age_hours: 6
  
  staging:
    max_tranches: 3
    min_spacing_days: 1
    short_timeframe_tranches: 2
    long_timeframe_tranches: 3
    urgent_pattern: [0.6, 0.4]
    normal_pattern: [0.5, 0.5]
    flexible_pattern: [0.33, 0.33, 0.34]
  
  costs:
    default_spread_bps: 5.0
    default_fee_bps: 0.0
    staging_cost_multiplier: 1.2
```

## Core Logic Components

### 1. Utility Scorer (`src/decision/utility_scorer.py`)

Scores three candidate actions using utility model:

**Utility Formula:**

```
utility = w_profit × expected_improvement 
    - w_risk × risk_penalty 
    - w_cost × transaction_cost 
    + w_urgency × urgency_fit
```

**Key Methods:**

- `score_actions()`: Score all three actions
- `_score_convert_now()`: Immediate conversion score
- `_score_staged()`: Staged conversion score (reduced risk)
- `_score_wait()`: Wait until timeframe score
- `_get_expected_improvement()`: From Prediction → Intelligence → Technical
- `_calculate_risk_penalty()`: Volatility + event proximity

### 2. Staging Planner (`src/decision/staging_planner.py`)

Generates multi-tranche staging plans:

**Key Logic:**

- Determine tranche count based on timeframe (2 for ≤5 days, 3 for longer)
- Calculate spacing: `timeframe_days / num_tranches`
- Apply sizing pattern by urgency (front-loaded vs equal)
- Avoid high-impact event days (shift tranches if needed)
- Generate rationale for each tranche

**Edge Cases:**

- Event on day 2 of 3-day window → shift after event or reduce tranches
- Multiple events cluster → increase spacing
- Never schedule within 12h of high-impact event

### 3. Confidence Aggregator (`src/decision/confidence_aggregator.py`)

Aggregates confidence from multiple sources:

**Component Confidences:**

- Market: Base 0.5 + data quality + indicator coherence
- Intelligence: Event clarity + source coverage
- Prediction: Model confidence from prediction agent

**Penalties:**

- Missing components: -10% to -5%
- Warnings: -10% per 2 warnings
- Stale data: -5%

**Final**: Weighted average clipped to [0, 1]

### 4. Cost Calculator (`src/decision/cost_calculator.py`)

Calculates transaction costs:

- Spread (default 5 bps or from market data)
- Fee (user-provided or 0)
- Staging multiplier (1.2x for staged, 1.0x for single)

### 5. Decision Maker (`src/decision/decision_maker.py`)

Main orchestrator that:

1. Scores all actions via utility model
2. Applies hard constraints (event blocking for conservative)
3. Selects best action (argmax utility)
4. Generates staging plan if needed
5. Calculates aggregate confidence
6. Builds expected outcome and risk summary
7. Generates top 3-5 rationale points
8. Returns complete DecisionResponse

## Agent Integration

### Update: `src/agentic/nodes/decision.py`

DecisionEngineAgent that:

- Builds DecisionRequest from state
- Calls DecisionMaker
- Updates state.decision with DecisionAnalysis
- Handles errors gracefully

## Example Decision Flow

```
Input:
- Currency: USD/EUR
- Amount: 5000
- Risk: moderate
- Urgency: urgent
- Timeframe: 7 days
- Market: bullish, RSI=65
- Intelligence: Fed meeting in 2 days (high impact)
- Prediction: +0.3% in 7 days (confidence 0.7)

Utility Scores:
- convert_now: 0.25 (blocked due to event in 2 days for moderate)
- staged: 0.45 (best - manages event risk)
- wait: 0.35 (waits for event but less diversification)

Decision:
- Action: staged_conversion
- Tranches: 3 (Day 0: 33%, Day 4: 33%, Day 7: 34%)
- Confidence: 0.68
- Rationale:
 1. Staging manages Fed meeting risk in 2 days
 2. Captures predicted +0.3% upside gradually
 3. Moderate risk profile benefits from diversification
 4. Urgent timeline allows 7-day staging window
```

## Implementation Todos

- [ ] Create data contracts in `src/decision/models.py`
- [ ] Create DecisionConfig in `src/decision/config.py` and update `config.yaml`
- [ ] Implement UtilityScorer in `src/decision/utility_scorer.py`
- [ ] Implement StagingPlanner in `src/decision/staging_planner.py`
- [ ] Implement ConfidenceAggregator in `src/decision/confidence_aggregator.py`
- [ ] Implement CostCalculator in `src/decision/cost_calculator.py`
- [ ] Implement DecisionMaker orchestrator in `src/decision/decision_maker.py`
- [ ] Update DecisionEngineAgent in `src/agentic/nodes/decision.py`
- [ ] Update AgentGraphState to include DecisionAnalysis if missing
- [ ] Write unit tests for utility scorer
- [ ] Write unit tests for staging planner
- [ ] Write unit tests for confidence aggregator
- [ ] Write integration tests for decision maker with scenarios
- [ ] Create example script with mock data
- [ ] Document decision model and tuning guide