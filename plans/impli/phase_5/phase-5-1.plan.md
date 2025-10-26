<!-- f67714c1-a54f-4e8d-9617-16955f212afc 12c48d5c-0a70-4b89-8920-049f9402be91 -->
# Phase 5.1: TUI (Terminal User Interface) Implementation

## Overview

Build an interactive Terminal User Interface using the `rich` library to provide a conversational interface for the Currency Assistant. The TUI will be the first user-facing interface, enabling end-to-end testing of the complete agent system.

## Purpose

- Test Supervisor Agent and all downstream agents end-to-end
- Provide fast, lightweight interface for development validation
- Enable multi-turn conversations with visual feedback
- Display formatted recommendations with tables, panels, and colors

## Architecture

```
src/ui/tui/
├── __init__.py
├── app.py                    # Main TUI application
├── display.py                # Display components (tables, panels)
├── input_handler.py          # User input handling
├── renderer.py               # Rich formatting utilities
└── config.py                 # TUI configuration

tests/ui/tui/
├── test_tui_app.py
└── test_display.py
```

## File Descriptions

### 1. `src/ui/tui/app.py`

**Purpose**: Main TUI application entry point with conversation loop.

**Inputs**:

- User text input via Rich prompts
- Session state from ConversationManager
- Recommendations from AgentOrchestrator

**Outputs**:

- Formatted console output with colors and panels
- Interactive prompts for user input
- Progress indicators during agent execution

**Key Components**:

- `CurrencyAssistantTUI` class: Main application controller
- `run()`: Main entry point with welcome screen
- `conversation_loop()`: Multi-turn conversation handling
- `run_agents()`: Execute agent orchestration with progress display
- `display_recommendation()`: Format and show final recommendation

**Integration**:

- Uses `ConversationManager` for NLU and parameter extraction
- Uses `AgentOrchestrator` to run the complete agent workflow
- Uses `ResponseFormatter` to prepare user-friendly messages

### 2. `src/ui/tui/display.py`

**Purpose**: Rich display components for formatting output.

**Functions**:

- `create_welcome_panel()`: Welcome screen with app description
- `create_parameter_table()`: Display extracted parameters for user confirmation
- `create_recommendation_panel()`: Format recommendation as Rich panel
- `create_staged_plan_table()`: Display staging plan with tranches
- `create_risk_summary_table()`: Show risk metrics
- `format_confidence_gauge()`: Text-based confidence visualization

**Inputs**: Python dictionaries from state/recommendation

**Outputs**: Rich renderable objects (Panel, Table, Text)

### 3. `src/ui/tui/input_handler.py`

**Purpose**: Handle user input and validation.

**Functions**:

- `get_user_input(prompt)`: Get text input with validation
- `confirm_parameters(params)`: Ask user to confirm extracted parameters
- `get_parameter_edit(param_name)`: Allow user to edit specific parameter
- `ask_yes_no(question)`: Yes/no prompt with default

**Inputs**: Prompt strings, validation rules

**Outputs**: Validated user input strings

### 4. `src/ui/tui/renderer.py`

**Purpose**: Rich formatting utilities and style definitions.

**Functions**:

- `format_currency(amount, currency)`: Format currency values
- `format_percentage(value)`: Format percentage values
- `format_confidence(confidence)`: Color-coded confidence display
- `format_action(action)`: Format action with icon and color
- `get_color_for_confidence(confidence)`: Return appropriate color

**Inputs**: Raw values (floats, strings)

**Outputs**: Formatted Rich text with styles

### 5. `src/ui/tui/config.py`

**Purpose**: TUI configuration and color schemes.

**Configuration**:

- Color themes (success, warning, error, info)
- Progress spinner styles
- Table box styles
- Default prompts and messages

## Key Features

### 1. Welcome Screen

- Display app name and description
- Show what the assistant can do
- Provide example queries

### 2. Conversational Flow

- Multi-turn parameter collection
- Smart extraction with confirmation
- Allow parameter editing before analysis
- Clear progress indicators

### 3. Recommendation Display

- Action card with confidence gauge
- Timeline and staging plan (if applicable)
- Rationale bullets with reasoning
- Risk metrics and cost summaries
- Highlighted warnings

### 4. Progress Feedback

- Spinner during agent execution
- Progress messages ("Analyzing market...", "Generating predictions...")
- Visual completion indicators

## Implementation Steps

1. **Create TUI directory structure**
   ```bash
   mkdir -p src/ui/tui tests/ui/tui
   ```

2. **Implement `config.py`**: Define color schemes and styles

3. **Implement `renderer.py`**: Create formatting utilities

4. **Implement `display.py`**: Build Rich components (panels, tables)

5. **Implement `input_handler.py`**: Handle user input with validation

6. **Implement `app.py`**: Main TUI application with conversation loop

7. **Write tests**: Test display components and input handling

8. **Add entry point**: Update `pyproject.toml` with CLI command

## Dependencies

- `rich>=13.0.0`: Console formatting library (already in dependencies)
- Existing supervisor and orchestrator modules

## Entry Point Configuration

**Update `pyproject.toml`**:

```toml
[project.scripts]
currency-assistant-tui = "src.ui.tui.app:main"
```

## Usage

```bash
# Direct execution
python -m src.ui.tui.app

# Installed command (after pip install)
currency-assistant-tui

# With uv
uv run currency-assistant
```

## Example Session Flow

1. Welcome screen displays
2. User enters: "I need to convert 5000 USD to EUR"
3. System extracts parameters and displays table for confirmation
4. User confirms parameters
5. Progress spinner shows agent execution
6. Recommendation displays with:

   - Action (CONVERT NOW / STAGED / WAIT)
   - Confidence gauge
   - Timeline
   - Rationale bullets
   - Risk summary
   - Cost estimate

7. User can start another analysis or exit

## Testing Strategy

- **Unit tests**: Test display components and formatters
- **Mock tests**: Test TUI flow with mocked orchestrator
- **Manual testing**: Run actual TUI and test conversation flows

## Success Criteria

- TUI displays welcome screen correctly
- Multi-turn conversation works smoothly
- Parameters are extracted and displayed for confirmation
- Progress indicators show during agent execution
- Recommendations are formatted clearly with all required information
- User can edit parameters before analysis
- Error messages are displayed appropriately
- User can perform multiple analyses in one session

## Integration Points

- `src.supervisor/conversation_manager.py`: Conversation flow + session management
- `src.supervisor/nlu_extractor.py`: Parameter extraction
- `src.supervisor/agent_orchestrator.py`: Agent workflow execution
- `src/agentic/graph.py`: LangGraph workflow
- `src.supervisor/response_formatter.py`: Response formatting

## Notes

- TUI is the **primary testing interface** before implementing web UI
- Should work completely offline (no external dependencies beyond agents)
- All agent functionality should be testable through TUI
- Focus on clarity and usability over visual complexity
- Use colors strategically (success=green, warning=yellow, error=red)

### To-dos

- [ ] Create main TUI application in src/ui/tui/app.py with conversational interface
- [ ] Implement Rich display components in src/ui/tui/display.py (welcome, parameter table, recommendation panel)
- [ ] Create input handler in src/ui/tui/input_handler.py for user input validation
- [ ] Implement Rich formatting utilities in src/ui/tui/renderer.py
- [ ] Create TUI configuration in src/ui/tui/config.py with color schemes
- [ ] Write comprehensive unit tests for display components and input handling
- [ ] Add CLI entry point in pyproject.toml with currency-assistant command
- [ ] Test end-to-end TUI flow with ConversationManager and AgentOrchestrator
