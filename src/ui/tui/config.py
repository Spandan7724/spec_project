from __future__ import annotations

"""TUI configuration and style constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    primary: str = "cyan"
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "blue"
    neutral: str = "white"


@dataclass(frozen=True)
class BoxStyles:
    welcome: str = "DOUBLE"
    panel: str = "ROUNDED"
    error: str = "HEAVY"


THEME = Theme()
BOX = BoxStyles()

WELCOME_TEXT = (
    """
[bold cyan]Currency Assistant[/bold cyan]
AI-powered currency conversion recommendations

I'll help you decide when and how to convert currencies based on:
• Market analysis and technical indicators
• Economic calendar and major events
• ML-based price predictions
• Your risk tolerance and timing preferences
    """
    .strip()
)

