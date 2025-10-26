from __future__ import annotations

"""Formatting helpers for the TUI."""

from typing import Optional


def format_currency(amount: Optional[float], currency: Optional[str]) -> str:
    if amount is None:
        return "—"
    if not currency:
        return f"{amount:,.2f}"
    return f"{amount:,.2f} {currency}"


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return str(value)


def get_color_for_confidence(confidence: Optional[float]) -> str:
    try:
        c = float(confidence)
    except Exception:
        return "white"
    if c > 0.7:
        return "green"
    if c > 0.4:
        return "yellow"
    return "red"


def format_confidence(confidence: Optional[float]) -> str:
    if confidence is None:
        return "—"
    color = get_color_for_confidence(confidence)
    return f"[{color}]{float(confidence):.2f}[/]"


def format_action(action: Optional[str]) -> str:
    if not action:
        return "—"
    a = str(action).replace("_", " ").title()
    if "Convert" in a:
        return f"[green]{a}[/]"
    if a.lower().startswith("wait"):
        return f"[yellow]{a}[/]"
    return a

