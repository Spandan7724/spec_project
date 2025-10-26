# TUI (Terminal User Interface) Implementation Plan

## Overview

Build an interactive Terminal User Interface (TUI) using the `rich` library for initial testing and validation of the Currency Assistant system. The TUI will provide a conversational interface with formatted output, progress indicators, and color-coded information.

## Purpose

**Primary Goals:**

- Test Supervisor Agent and all downstream agents end-to-end
- Provide a fast, lightweight interface for development and testing
- Validate conversation flows and decision outputs
- Enable quick iteration without web complexity

## Technology Stack

**Library:** `rich` (already in dependencies)

- Console output with colors and formatting
- Tables for structured data
- Progress bars for agent execution
- Panels for recommendations
- Prompts for user input

**Alternative:** `textual` (for more interactive widgets)

- Not needed for MVP
- Can add later if interactive features required

## Features

### Core Features

1. **Interactive Conversation**

   - Multi-turn parameter collection
   - Smart extraction with confirmation
   - Parameter change support
   - Clear progress indicators

2. **Formatted Output**

   - Colored text (success/warning/error)
   - Tables for parameters and results
   - Panels for recommendations
   - Progress bars during agent execution

3. **Recommendation Display**

   - Action with confidence gauge
   - Timeline and staging plan (if applicable)
   - Rationale bullets
   - Risk and cost summaries
   - Warnings highlighted

## Architecture

```
src/ui/tui/
├── __init__.py
├── app.py                        # Main TUI application
├── formatters.py                 # Rich formatting utilities
├── prompts.py                    # User input prompts
└── display.py                    # Display components

tests/ui/tui/
├── test_tui_formatting.py
└── test_tui_flow.py
```

## Implementation

### 1. Main TUI Application

**File: `src/ui/tui/app.py`**

```python
import asyncio
import uuid
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.agent_orchestrator import AgentOrchestrator
from src.supervisor.response_formatter import ResponseFormatter
from src.supervisor.models import SupervisorRequest, ConversationState

console = Console()

class CurrencyAssistantTUI:
    """Terminal User Interface for Currency Assistant"""
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.orchestrator = AgentOrchestrator()
        self.formatter = ResponseFormatter()
        self.session_id = str(uuid.uuid4())
    
    def run(self):
        """Main entry point"""
        self.display_welcome()
        
        while True:
            try:
                asyncio.run(self.conversation_loop())
                
                # Ask if user wants another analysis
                another = Prompt.ask(
                    "\n[bold cyan]Would you like to make another conversion analysis?[/]",
                    choices=["yes", "no"],
                    default="no"
                )
                
                if another.lower() != "yes":
                    console.print("\n[bold green]Thank you for using Currency Assistant! Goodbye![/]")
                    break
                
                # Reset session for new conversation
                self.session_id = str(uuid.uuid4())
                console.print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Goodbye![/]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error: {e}[/]")
                console.print("[yellow]Please try again or contact support.[/]")
    
    def display_welcome(self):
        """Display welcome screen"""
        welcome_text = """
[bold cyan]Currency Assistant[/bold cyan]
AI-powered currency conversion recommendations

I'll help you decide when and how to convert currencies based on:
• Market analysis and technical indicators
• Economic calendar and major events  
• ML-based price predictions
• Your risk tolerance and timing preferences
        """
        
        panel = Panel(
            welcome_text.strip(),
            title="Welcome",
            border_style="cyan",
            box=box.DOUBLE
        )
        
        console.print(panel)
        console.print()
    
    async def conversation_loop(self):
        """Main conversation loop"""
        
        # Initial prompt
        console.print("[bold]Let's start with your currency conversion request.[/]\n")
        user_input = Prompt.ask("[cyan]You[/]")
        
        while True:
            # Process input through supervisor
            request = SupervisorRequest(
                user_input=user_input,
                session_id=self.session_id,
                correlation_id=str(uuid.uuid4())
            )
            
            response = self.conversation_manager.process_input(request)
            
            # Display response
            console.print(f"\n[bold magenta]Assistant[/]: {response.message}\n")
            
            # Check if we need more input
            if response.requires_input:
                user_input = Prompt.ask("[cyan]You[/]")
                continue
            
            # If processing, run agents
            if response.state == ConversationState.PROCESSING:
                recommendation = await self.run_agents(response.parameters, request.correlation_id)
                self.display_recommendation(recommendation)
                break
            
            # If completed or error
            if response.state in [ConversationState.COMPLETED, ConversationState.ERROR]:
                break
    
    async def run_agents(self, parameters, correlation_id):
        """Run agent orchestration with progress display"""
        
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Layer 1
            task1 = progress.add_task("[cyan]Analyzing market conditions...", total=None)
            await asyncio.sleep(0.5)  # Give visual feedback
            
            # Layer 2
            progress.update(task1, description="[cyan]Fetching economic calendar and news...")
            await asyncio.sleep(0.5)
            
            # Layer 3
            progress.update(task1, description="[cyan]Generating price predictions...")
            await asyncio.sleep(0.5)
            
            # Decision
            progress.update(task1, description="[cyan]Calculating optimal recommendation...")
            
            # Run actual agents
            recommendation = await self.orchestrator.run_analysis(parameters, correlation_id)
            
            progress.update(task1, description="[green]✓ Analysis complete!", completed=True)
        
        console.print()
        return recommendation
    
    def display_recommendation(self, recommendation):
        """Display final recommendation"""
        
        if recommendation.get("status") == "error":
            self.display_error(recommendation)
            return
        
        # Action and confidence
        action = recommendation["action"].replace("_", " ").title()
        confidence = recommendation["confidence"]
        
        # Confidence color
        if confidence > 0.7:
            conf_color = "green"
            conf_label = "High"
        elif confidence > 0.4:
            conf_color = "yellow"
            conf_label = "Moderate"
        else:
            conf_color = "red"
            conf_label = "Low"
        
        # Create summary table
        table = Table(title="Recommendation Summary", box=box.ROUNDED, show_header=False)
        table.add_column("Field", style="cyan bold", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("Action", f"[bold green]{action}[/]")
        table.add_row("Confidence", f"[{conf_color}]{confidence:.2f}[/] ({conf_label})")
        table.add_row("Timeline", recommendation["timeline"])
        
        console.print(table)
        console.print()
        
        # Staged plan (if applicable)
        if "staged_plan" in recommendation:
            self.display_staged_plan(recommendation["staged_plan"])
        
        # Rationale
        if recommendation.get("rationale"):
            console.print("[bold cyan]Rationale:[/]")
            for i, reason in enumerate(recommendation["rationale"], 1):
                console.print(f"  [white]{i}.[/] {reason}")
            console.print()
        
        # Risk and costs
        if "risk_summary" in recommendation:
            risk = recommendation["risk_summary"]["risk_level"].title()
            risk_color = "green" if risk == "Low" else "yellow" if risk == "Moderate" else "red"
            console.print(f"[bold]Risk Level:[/] [{risk_color}]{risk}[/]")
        
        if "cost_estimate" in recommendation:
            cost = recommendation["cost_estimate"]["total_bps"]
            console.print(f"[bold]Estimated Costs:[/] {cost:.1f} basis points")
        
        console.print()
        
        # Warnings
        if recommendation.get("warnings"):
            console.print("[bold yellow]⚠ Warnings:[/]")
            for warning in recommendation["warnings"]:
                console.print(f"  [yellow]•[/] {warning}")
            console.print()
    
    def display_staged_plan(self, plan):
        """Display staging plan"""
        
        table = Table(title="Staged Conversion Plan", box=box.SIMPLE)
        table.add_column("Tranche", style="cyan")
        table.add_column("Percentage", justify="right", style="green")
        table.add_column("Execute Day", justify="center", style="yellow")
        table.add_column("Rationale", style="white")
        
        for tranche in plan["tranches"]:
            table.add_row(
                f"#{tranche['number']}",
                f"{tranche['percentage']:.0f}%",
                f"Day {tranche['execute_day']}",
                tranche.get('rationale', '')
            )
        
        console.print(table)
        console.print()
    
    def display_error(self, recommendation):
        """Display error message"""
        
        error_text = f"[bold red]Error:[/] {recommendation.get('error', 'Unknown error')}"
        
        if recommendation.get("warnings"):
            error_text += "\n\n[bold yellow]Details:[/]"
            for warning in recommendation["warnings"]:
                error_text += f"\n  • {warning}"
        
        panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            box=box.HEAVY
        )
        
        console.print(panel)


def main():
    """Entry point"""
    app = CurrencyAssistantTUI()
    app.run()


if __name__ == "__main__":
    main()
```

### 2. CLI Entry Point

You can expose a dedicated TUI command without conflicting with the existing CLI:

```toml
[project.scripts]
currency-assistant-tui = "src.ui.tui.app:main"
```

Alternatively, add an `interactive` subcommand in `src/cli/main.py` that imports and runs `CurrencyAssistantTUI`.

### 3. Package Entry Point

Use the same entry as above so the installed command is consistent.

```toml
[project.scripts]
currency-assistant-tui = "src.ui.tui.app:main"
```

## Usage

### Running the TUI

```bash
# Option 1: Direct
python -m src.ui.tui.app

# Option 2: Installed command (after pip install)
currency-assistant-tui

# Option 3: Using uv
uv run currency-assistant-tui
```

### Example Session

```

╔══════════════════════════════════════════════════════════╗

║                        Welcome                           ║

║                                                          ║

║              Currency Assistant                          ║

║      AI-powered currency conversion recommendations      ║

║                                                          ║

║  I'll help you decide when and how to convert          ║

║  currencies based on:                                   ║

║  • Market analysis and technical indicators             ║

║  • Economic calendar and major events                   ║

║  • ML-based price predictions                           ║

║  • Your risk tolerance and timing preferences           ║

╚══════════════════════════════════════════════════════════╝

Let's start with your currency conversion request.

You: I need to convert 5000 USD to EUR
