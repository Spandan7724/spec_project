<!-- f67714c1-a54f-4e8d-9617-16955f212afc dcc9aacc-ecac-414e-aacd-e97ebf96a335 -->
# Phase 4.3: Response Generator & Formatting

## Overview

Build the response formatter that takes the technical recommendation output from the Decision Engine and formats it into clear, professional, user-friendly text. This includes formatting the action, confidence, rationale, staged plans, risk assessments, and warnings in a visually appealing way.

## Reference

See `plans/supervisor-agent.plan.md` for response formatter implementation (lines 1026-1130) and `plans/decision-engine.plan.md` for DecisionResponse structure.

## Files to Create

### 1. `src/supervisor/response_formatter.py`

**Purpose**: Format technical recommendation data into user-friendly text.

**Class**: `ResponseFormatter`

**Main Method**: `format_recommendation(recommendation: Dict[str, Any]) -> str`

**Inputs**:

- `recommendation`: Dict containing decision response (from Decision Engine)
  - Can have status "success" or "error"
  - Contains: action, confidence, timeline, rationale, staged_plan, expected_outcome, risk_summary, cost_estimate, warnings

**Outputs**:

- Formatted string ready for display to user
- Multi-line text with visual separators, emojis, and clear sections

**Logic for Success Response**:

1. **Header Section**:

   - Separator line (━━━)
   - Title: "RECOMMENDATION FOR CURRENCY CONVERSION"
   - Separator line

2. **Action Section**:

   - Display action in uppercase (CONVERT NOW, STAGED CONVERSION, WAIT)
   - Bold or highlighted presentation

3. **Confidence Section**:

   - Show numeric confidence (0.XX format)
   - Translate to level: High (>0.7), Moderate (>0.4), Low (≤0.4)
   - Format: "CONFIDENCE: 0.75 (High)"

4. **Timeline Section**:

   - Display timeline string from decision engine
   - Examples: "Immediate execution recommended", "Execute in 3 tranches over 7 days"

5. **Staged Plan Section** (if applicable):

   - Header: "STAGED CONVERSION PLAN:"
   - List each tranche with:
     - Tranche number
     - Percentage
     - Execution day
     - Rationale (optional)
   - Format: "• Tranche 1: 60% on Day 0"

6. **Expected Outcome Section** (if available):

   - Expected rate
   - Expected improvement in bps
   - Range (low to high)

7. **Rationale Section**:

   - Header: "RATIONALE:"
   - Numbered list of reasons (top 3-5)
   - Format: "1. Prediction shows +0.5% improvement potential"

8. **Risk Assessment Section** (if available):

   - Risk level: Low/Moderate/High
   - Volatility metric (if available)
   - Event risk details (if applicable)

9. **Cost Estimate Section** (if available):

   - Total cost in basis points
   - Spread and fee breakdown (optional)

10. **Warnings Section** (if any):

    - Header: "WARNINGS:"
    - List with warning emoji (⚠️)
    - Format: "⚠️  High volatility detected"

11. **Next Steps Section**:

    - Prompt user for next action:
      - "Execute this recommendation"
      - "Get alternative scenarios"
      - "Start a new analysis"

**Error Handling**: `_format_error(recommendation: Dict[str, Any]) -> str`

**Logic for Error Response**:

1. Header with "ERROR GENERATING RECOMMENDATION"
2. Display error message
3. Show warnings/details if available
4. Suggest retry or contact support

**Internal Helper Methods**:

- `_format_confidence_level(confidence: float) -> str`
  - Converts numeric confidence to text level

- `_format_staged_plan(plan: Dict) -> List[str]`
  - Formats staged plan as text lines

- `_format_rationale(rationale: List[str]) -> List[str]`
  - Formats rationale as numbered list

- `_format_warnings(warnings: List[str]) -> List[str]`
  - Formats warnings with emoji

**Visual Elements**:

- Use Unicode box-drawing characters (━, ─, │)
- Use emoji for warnings (⚠️), checkmarks (✓), bullets (•)
- Use consistent indentation and spacing
- Clear section separation

### 2. `src/supervisor/message_templates.py`

**Purpose**: Store reusable message templates and text snippets.

**Templates**:

- `GREETING_MESSAGE`: Welcome message for new users
- `CONFIRMATION_PROMPT`: Standard confirmation request text
- `RESTART_MESSAGE`: Message when conversation restarts
- `PROCESSING_MESSAGE`: "Analyzing market conditions..." with spinner/progress indication
- `ERROR_MESSAGES`: Dict of error types to user-friendly messages
- `HELP_TEXT`: Instructions for using the system

**Functions**:

- `get_processing_message(stage: str) -> str`
  - Returns stage-specific processing message
  - Stages: "market_data", "market_intelligence", "prediction", "decision"
  - Example: "Fetching real-time market data...", "Analyzing price predictions..."

- `get_parameter_prompt(param_name: str) -> str`
  - Returns formatted prompt for specific parameter
  - Includes examples and options

### 3. `tests/supervisor/test_response_formatter.py`

**Purpose**: Unit tests for response formatter.

**Test Cases**:

**Success Response Formatting**:

- `test_format_convert_now_action`: Action "convert_now" → formatted correctly
- `test_format_staged_action`: Action "staged_conversion" with plan → includes tranche details
- `test_format_wait_action`: Action "wait" → formatted correctly
- `test_format_high_confidence`: Confidence 0.85 → shows "High"
- `test_format_moderate_confidence`: Confidence 0.55 → shows "Moderate"
- `test_format_low_confidence`: Confidence 0.35 → shows "Low"
- `test_format_with_staged_plan`: Staged plan present → formatted with tranches
- `test_format_with_rationale`: Rationale list → numbered correctly
- `test_format_with_warnings`: Warnings present → shown with emoji
- `test_format_with_all_sections`: Complete recommendation → all sections present

**Error Response Formatting**:

- `test_format_error_response`: Error status → error message displayed
- `test_format_error_with_warnings`: Error + warnings → both shown

**Visual Formatting**:

- `test_separator_lines_present`: Check separator lines in output
- `test_sections_properly_spaced`: Verify spacing between sections
- `test_bullet_points_formatted`: Check bullet formatting

**Edge Cases**:

- `test_format_minimal_recommendation`: Only required fields → no crashes
- `test_format_empty_rationale`: Empty rationale list → skip section
- `test_format_no_warnings`: No warnings → skip section

**Fixtures**:

- `formatter`: ResponseFormatter instance
- `sample_recommendation`: Complete recommendation dict
- `sample_error`: Error response dict

### 4. `tests/supervisor/test_message_templates.py`

**Purpose**: Unit tests for message templates.

**Test Cases**:

- `test_greeting_message_exists`: Greeting template defined
- `test_get_processing_message_market_data`: Returns market data message
- `test_get_processing_message_prediction`: Returns prediction message
- `test_get_parameter_prompt_risk`: Returns risk prompt with options
- `test_get_parameter_prompt_urgency`: Returns urgency prompt with options
- `test_error_messages_defined`: All error types have messages

### 5. `tests/supervisor/test_response_integration.py`

**Purpose**: Integration test for complete recommendation formatting.

**Test Scenario**: Format complete decision response end-to-end

**Test Case**: `test_format_complete_recommendation`

**Setup**:

- Create complete DecisionResponse (from Phase 3)
- Convert to dict
- Pass to ResponseFormatter

**Expected Behavior**:

- All sections formatted correctly
- Text is readable and well-structured
- No missing or broken formatting
- Visual elements display properly

**Validation**:

- Assert action appears in output
- Assert confidence appears with level
- Assert rationale items present
- Assert sections are separated
- Assert total length reasonable (not too short/long)

## Validation

Manual validation script:

```python
from src.supervisor.response_formatter import ResponseFormatter

# Create formatter
formatter = ResponseFormatter()

# Test recommendation 1: Convert Now
recommendation_convert = {
    "status": "success",
    "action": "convert_now",
    "confidence": 0.78,
    "timeline": "Immediate execution recommended",
    "rationale": [
        "Urgent timeline requires immediate action",
        "Current rate is favorable",
        "No high-impact events in next 24 hours"
    ],
    "risk_summary": {
        "risk_level": "low",
        "realized_vol_30d": 8.5
    },
    "cost_estimate": {
        "total_bps": 5.0
    },
    "warnings": []
}

print("=== Convert Now Recommendation ===")
print(formatter.format_recommendation(recommendation_convert))

# Test recommendation 2: Staged Conversion
recommendation_staged = {
    "status": "success",
    "action": "staged_conversion",
    "confidence": 0.68,
    "timeline": "Execute in 3 tranches over 7 days",
    "rationale": [
        "Staging manages Fed meeting risk in 3 days",
        "Captures predicted +0.5% upside gradually",
        "Moderate risk profile benefits from diversification"
    ],
    "staged_plan": {
        "num_tranches": 3,
        "tranches": [
            {"number": 1, "percentage": 33, "execute_day": 0, "rationale": "Initial conversion"},
            {"number": 2, "percentage": 33, "execute_day": 4, "rationale": "After Fed meeting"},
            {"number": 3, "percentage": 34, "execute_day": 7, "rationale": "Final tranche"}
        ]
    },
    "risk_summary": {
        "risk_level": "moderate",
        "event_risk": "moderate",
        "event_details": "Fed meeting in 3 days"
    },
    "cost_estimate": {
        "total_bps": 6.0
    },
    "warnings": ["High-impact event approaching"]
}

print("\n\n=== Staged Conversion Recommendation ===")
print(formatter.format_recommendation(recommendation_staged))

# Test recommendation 3: Wait
recommendation_wait = {
    "status": "success",
    "action": "wait",
    "confidence": 0.72,
    "timeline": "Wait until day 7 for better rate",
    "rationale": [
        "Prediction shows +0.8% improvement in 7 days",
        "Strong uptrend in technical indicators",
        "Flexible timeline allows waiting"
    ],
    "expected_outcome": {
        "expected_rate": 0.9245,
        "expected_improvement_bps": 80
    },
    "risk_summary": {
        "risk_level": "low"
    },
    "warnings": []
}

print("\n\n=== Wait Recommendation ===")
print(formatter.format_recommendation(recommendation_wait))

# Test error response
error_response = {
    "status": "error",
    "error": "Prediction service unavailable",
    "warnings": ["Using heuristic fallback", "Confidence reduced"]
}

print("\n\n=== Error Response ===")
print(formatter.format_recommendation(error_response))
```

## Success Criteria

- [ ] ResponseFormatter formats convert_now action correctly
- [ ] ResponseFormatter formats staged_conversion with tranche plan
- [ ] ResponseFormatter formats wait action correctly
- [ ] Confidence shown as both number and level (High/Moderate/Low)
- [ ] Rationale formatted as clear numbered list
- [ ] Staged plan shows all tranches with details
- [ ] Warnings displayed with appropriate emoji
- [ ] Error responses formatted with helpful messages
- [ ] Visual separators and spacing make output readable
- [ ] All sections properly labeled and organized
- [ ] Missing/optional sections handled gracefully
- [ ] All unit tests pass with >80% coverage
- [ ] Manual validation shows professional, clear output

## Key Design Decisions

1. **Clear Structure**: Consistent section ordering and formatting
2. **Visual Elements**: Use Unicode characters and emoji for better UX
3. **Progressive Disclosure**: Show relevant sections only (skip empty ones)
4. **Professional Tone**: Formal but friendly language
5. **Actionable**: Always include next steps for user
6. **Error-Friendly**: Clear error messages with actionable advice
7. **Extensible**: Easy to add new sections or modify formatting
8. **Text-Based**: No HTML/rich formatting (works in TUI and web)

## Integration Points

- Receives recommendation dict from Agent Orchestrator (Decision Engine output)
- Output displayed in TUI (Phase 5.1) and Web UI (Phase 5.2, 5.3)
- Message templates used by Conversation Manager
- Formatting consistent across all user interfaces

## Future Enhancements (Post-MVP)

- Color coding for confidence levels
- Charts/graphs for predictions (web UI)
- Multi-language support
- Customizable formatting preferences
- Export to PDF/email format

### To-dos

- [ ] Implement ResponseFormatter in src/supervisor/response_formatter.py with section formatting
- [ ] Create message templates in src/supervisor/message_templates.py
- [ ] Write comprehensive unit tests for response formatter (all actions, confidence levels, sections)
- [ ] Write unit tests for message templates
- [ ] Write integration test for complete recommendation formatting