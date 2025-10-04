#!/usr/bin/env python3
"""
Currency Assistant CLI - Interactive currency conversion timing advisor
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.chat import ChatSession
from src.cli.services import CurrencyAdvisorService
from src.cli.display import DisplayManager
from src.cli.config import ConfigManager

# Initialize CLI components
app = typer.Typer(
    name="currency-assistant",
    help="Interactive currency conversion timing advisor",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Resume a previous session"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging")
) -> None:
    """Start an interactive chat session with the currency advisor."""

    console.print(Panel(
        Text("Currency Assistant", style="bold blue"),
        subtitle="Interactive currency conversion timing advisor"
    ))

    try:
        chat_session = ChatSession(session_id=session_id, debug=debug)
        asyncio.run(chat_session.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting chat session: {e}[/red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your currency conversion question"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging")
) -> None:
    """Ask a single question and get an immediate recommendation."""

    console.print(f"[blue]Question:[/blue] {question}")

    try:
        service = CurrencyAdvisorService(debug=debug)
        display = DisplayManager()

        with console.status("[bold green]Analyzing...", spinner="dots"):
            result = asyncio.run(service.ask_question(question))

        display.show_recommendation(result)

    except Exception as e:
        console.print(f"[red]Error processing question: {e}[/red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status")
) -> None:
    """Check system health and configuration status."""

    display = DisplayManager()

    try:
        service = CurrencyAdvisorService()
        status_info = asyncio.run(service.get_system_status())
        display.show_system_status(status_info, verbose=verbose)

    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key (format: key=value)"),
) -> None:
    """Manage CLI configuration."""

    config_manager = ConfigManager()
    display = DisplayManager()

    try:
        if set_key:
            if "=" not in set_key:
                console.print("[red]Invalid format. Use: key=value[/red]")
                raise typer.Exit(1)

            key, value = set_key.split("=", 1)
            config_manager.set(key.strip(), value.strip())
            console.print(f"[green]✅ Set {key.strip()} = {value.strip()}[/green]")

        if show or not set_key:
            config_info = config_manager.get_all()
            display.show_configuration(config_info)

    except Exception as e:
        console.print(f"[red]Error managing configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def demo() -> None:
    """Run a demo showcasing the assistant's capabilities."""

    console.print(Panel(
        Text("Demo Mode", style="bold purple"),
        subtitle="Experience the Currency Assistant capabilities"
    ))

    demo_questions = [
        "I need to convert $1000 USD to EUR this week. What's your recommendation?",
        "Should I wait or convert 5000 GBP to USD right now?",
        "What's the outlook for EUR/JPY over the next 30 days?",
    ]

    service = CurrencyAdvisorService()
    display = DisplayManager()

    for i, question in enumerate(demo_questions, 1):
        console.print(f"\n[bold cyan]Demo {i}:[/bold cyan]")
        console.print(f"[blue]Question:[/blue] {question}")

        try:
            with console.status("[bold green]Analyzing...", spinner="dots"):
                result = asyncio.run(service.ask_question(question))

            display.show_recommendation(result)

            if i < len(demo_questions):
                input("\nPress Enter to continue to next demo...")

        except Exception as e:
            console.print(f"[red]Demo question failed: {e}[/red]")

    console.print("\n[bold green]Demo completed![/bold green]")
    console.print("Try [cyan]currency-assistant chat[/cyan] for your own questions!")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", help="Show version and exit"
    )
) -> None:
    """Currency Assistant - AI-powered currency conversion timing advisor."""

    if version:
        console.print("Currency Assistant v0.1.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()