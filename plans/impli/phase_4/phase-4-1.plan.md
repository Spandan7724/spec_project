<!-- f67714c1-a54f-4e8d-9617-16955f212afc 6c2a3a47-0845-41da-908d-0184dae85b92 -->
# Phase 4.1: NLU Parameter Extraction

## Overview

Build the Natural Language Understanding (NLU) system that extracts structured parameters from free-form user input. This component uses regex patterns and keyword matching to identify currency pairs, amounts, risk preferences, urgency levels, and timeframes from conversational text.

## Reference

See `plans/supervisor-agent.plan.md` for data contracts and NLU extractor implementation (lines 95-390).

## Files to Create

### 1. `src/supervisor/__init__.py`

Empty init file for supervisor module.

### 2. `src/supervisor/models.py`

**Purpose**: Define data contracts for supervisor agent.

**Enums**:

- `ConversationState`: INITIAL, COLLECTING_CURRENCY_PAIR, COLLECTING_AMOUNT, COLLECTING_RISK, COLLECTING_URGENCY, COLLECTING_TIMEFRAME, CONFIRMING, PROCESSING, COMPLETED, ERROR

**Data Classes**:

- `ExtractedParameters`: Container for extracted parameters
  - Fields: currency_pair, base_currency, quote_currency, amount, risk_tolerance, urgency, timeframe, timeframe_days
  - Methods:
    - `is_complete()`: Returns True if all required parameters set
    - `missing_parameters()`: Returns list of missing parameter names

- `ConversationSession`: Session state for one conversion request
  - Fields: session_id, state (ConversationState), parameters (ExtractedParameters), conversation_history, warnings, created_at, last_updated
  - Methods:
    - `add_message(role, content)`: Add message to history with timestamp

- `SupervisorRequest`: Input to supervisor
  - Fields: user_input, session_id (optional), correlation_id (optional)

- `SupervisorResponse`: Output from supervisor
  - Fields: session_id, state, message, requires_input (bool), parameters (optional), recommendation (optional), warnings, errors

### 3. `src/supervisor/nlu_extractor.py`

**Purpose**: Extract structured parameters from natural language text.

**Class**: `NLUExtractor`

**Class Attributes** (constants):

- `CURRENCY_CODES`: Set of valid ISO currency codes (USD, EUR, GBP, etc.)
- `CURRENCY_NAMES`: Dict mapping currency names to codes ({"dollar": "USD", "euro": "EUR", ...})
- `RISK_KEYWORDS`: Dict mapping risk levels to keyword lists
- `URGENCY_KEYWORDS`: Dict mapping urgency levels to keyword lists
- `TIMEFRAME_KEYWORDS`: Dict mapping timeframes to keyword lists

**Main Method**: `extract(text: str) -> ExtractedParameters`

**Logic**:

1. Convert text to lowercase for matching
2. Create empty ExtractedParameters
3. Extract each parameter using specialized methods
4. Return populated ExtractedParameters

**Parameter Extraction Methods**:

- `_extract_currency_pair(text) -> Tuple[Optional[str], Optional[str]]`
  - Pattern 1: "USD/EUR" or "USD-EUR" or "USDEUR"
  - Pattern 2: "USD to EUR" or "USD into EUR"
  - Pattern 3: Currency names ("dollars to euros")
  - Returns (base_currency, quote_currency)

- `_extract_amount(text) -> Optional[float]`
  - Pattern: Numbers with optional commas and decimals ("5000", "5,000", "5000.50")
  - Returns first valid positive number found
  - Handles comma removal

- `_extract_risk_tolerance(text) -> Optional[str]`
  - Searches for keywords: conservative, moderate, aggressive
  - Returns matched risk level or None

- `_extract_urgency(text) -> Optional[str]`
  - Searches for keywords: urgent, normal, flexible
  - Returns matched urgency level or None

- `_extract_timeframe(text) -> Optional[str]`
  - Searches for keywords: immediate, 1_day, 1_week, 1_month
  - Returns matched timeframe or None

- `_timeframe_to_days(timeframe: str) -> int`
  - Maps timeframe to days: immediate→1, 1_day→1, 1_week→7, 1_month→30

**Validation Method**: `validate_currency_pair(base: str, quote: str) -> Tuple[bool, Optional[str]]`

**Logic**:

- Check base currency is valid
- Check quote currency is valid
- Check base != quote
- Returns (is_valid, error_message)

### 4. `src/supervisor/config.py`

**Purpose**: Configuration for supervisor agent.

**Configuration Items**:

- Session timeout duration (e.g., 30 minutes)
- Maximum conversation history length
- Supported currencies list
- Default values for missing parameters

**Class**: `SupervisorConfig`

**Fields**:

- `session_timeout_minutes`: int
- `max_history_length`: int
- `default_risk_tolerance`: str = "moderate"
- `default_urgency`: str = "normal"
- `default_timeframe`: str = "1_week"

**Method**: `from_yaml()` - Load from config.yaml

### 5. Update `config.yaml`

**Purpose**: Add supervisor configuration section.

```yaml
supervisor:
  session_timeout_minutes: 30
  max_history_length: 50
  defaults:
    risk_tolerance: "moderate"
    urgency: "normal"
    timeframe: "1_week"
```

### 6. `tests/supervisor/__init__.py`

Empty init file for test module.

### 7. `tests/supervisor/test_nlu_extractor.py`

**Purpose**: Unit tests for NLU parameter extraction.

**Test Cases**:

**Currency Pair Extraction**:

- `test_extract_currency_pair_slash_format`: "USD/EUR" → USD, EUR
- `test_extract_currency_pair_to_format`: "USD to EUR" → USD, EUR
- `test_extract_currency_pair_names`: "dollars to euros" → USD, EUR
- `test_extract_currency_pair_invalid`: Invalid currency → None, None
- `test_validate_currency_pair_valid`: USD/EUR → valid
- `test_validate_currency_pair_same`: USD/USD → invalid
- `test_validate_currency_pair_invalid_code`: XXX/EUR → invalid

**Amount Extraction**:

- `test_extract_amount_simple`: "5000" → 5000.0
- `test_extract_amount_with_commas`: "5,000" → 5000.0
- `test_extract_amount_with_decimals`: "5000.50" → 5000.50
- `test_extract_amount_none`: No number → None

**Risk Tolerance Extraction**:

- `test_extract_risk_conservative`: "I'm conservative" → "conservative"
- `test_extract_risk_moderate`: "balanced approach" → "moderate"
- `test_extract_risk_aggressive`: "willing to take risks" → "aggressive"
- `test_extract_risk_none`: No keywords → None

**Urgency Extraction**:

- `test_extract_urgency_urgent`: "I need this urgently" → "urgent"
- `test_extract_urgency_normal`: "normal timeline" → "normal"
- `test_extract_urgency_flexible`: "I can wait" → "flexible"
- `test_extract_urgency_none`: No keywords → None

**Timeframe Extraction**:

- `test_extract_timeframe_immediate`: "right now" → "immediate"
- `test_extract_timeframe_one_day`: "tomorrow" → "1_day"
- `test_extract_timeframe_one_week`: "within a week" → "1_week"
- `test_extract_timeframe_one_month`: "next month" → "1_month"
- `test_extract_timeframe_to_days`: Verify day conversion

**Full Extraction**:

- `test_extract_full_query`: "Convert 5000 USD to EUR urgently, moderate risk" → all params
- `test_extract_partial_query`: "5000 dollars to euros" → only currency and amount
- `test_extract_minimal_query`: "USD to EUR" → only currency pair

**Fixtures**:

- `extractor`: NLUExtractor instance

### 8. `tests/supervisor/test_models.py`

**Purpose**: Unit tests for data models.

**Test Cases**:

- `test_extracted_parameters_is_complete_true`: All params set → is_complete returns True
- `test_extracted_parameters_is_complete_false`: Missing param → is_complete returns False
- `test_extracted_parameters_missing_parameters`: Verify missing list correct
- `test_conversation_session_add_message`: Message added to history with timestamp
- `test_conversation_state_enum`: All enum values defined

## Validation

Manual validation script:

```python
from src.supervisor.nlu_extractor import NLUExtractor
from src.supervisor.models import ExtractedParameters

# Create extractor
extractor = NLUExtractor()

# Test various queries
queries = [
    "I need to convert 5000 USD to EUR urgently",
    "Convert 10,000 dollars to euros, I'm conservative and can wait a week",
    "USD/EUR 3000 moderate risk",
    "5000 from USD into GBP, flexible timeframe",
]

for query in queries:
    print(f"\nQuery: {query}")
    params = extractor.extract(query)
    print(f"  Currency: {params.currency_pair}")
    print(f"  Amount: {params.amount}")
    print(f"  Risk: {params.risk_tolerance}")
    print(f"  Urgency: {params.urgency}")
    print(f"  Timeframe: {params.timeframe}")
    print(f"  Complete: {params.is_complete()}")
    if not params.is_complete():
        print(f"  Missing: {params.missing_parameters()}")

# Test validation
base, quote = "USD", "EUR"
is_valid, error = extractor.validate_currency_pair(base, quote)
print(f"\n{base}/{quote} valid: {is_valid}")

base, quote = "USD", "USD"
is_valid, error = extractor.validate_currency_pair(base, quote)
print(f"{base}/{quote} valid: {is_valid}, error: {error}")
```

## Success Criteria

- [ ] ExtractedParameters dataclass with all required fields
- [ ] ConversationSession tracks state and history
- [ ] NLUExtractor extracts currency pairs from multiple formats
- [ ] NLUExtractor extracts amounts with commas and decimals
- [ ] NLUExtractor recognizes risk keywords correctly
- [ ] NLUExtractor recognizes urgency keywords correctly
- [ ] NLUExtractor recognizes timeframe keywords correctly
- [ ] Currency pair validation works correctly
- [ ] `is_complete()` and `missing_parameters()` methods accurate
- [ ] All unit tests pass with >80% coverage
- [ ] Manual validation shows correct extraction for various queries

## Key Design Decisions

1. **Pattern Matching**: Use regex for structured patterns (currency codes, amounts)
2. **Keyword Matching**: Use dictionaries for semantic concepts (risk, urgency)
3. **Flexible Input**: Support multiple formats for same information
4. **Partial Extraction**: Extract whatever is available, don't require all parameters
5. **No LLM for Basic Extraction**: Simple regex/keywords sufficient for MVP
6. **Validation**: Separate extraction from validation for flexibility
7. **Currency Names**: Support common names (dollar, euro) alongside codes

## Integration Points

- Used by Conversation Manager (Phase 4.2) to parse user input
- ExtractedParameters passed to Agent Orchestrator (Phase 4.3)
- Validation results guide conversation flow

### To-dos

- [ ] Create data contracts in src/supervisor/models.py (ExtractedParameters, ConversationSession, etc.)
- [ ] Implement NLUExtractor in src/supervisor/nlu_extractor.py with pattern matching and keyword recognition
- [ ] Create SupervisorConfig in src/supervisor/config.py and add supervisor section to config.yaml
- [ ] Write comprehensive unit tests for NLU extraction (currency, amount, risk, urgency, timeframe)
- [ ] Write unit tests for data models (is_complete, missing_parameters, add_message)