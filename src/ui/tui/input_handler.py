from __future__ import annotations

"""Prompt helpers for the TUI.

Note: The main conversation uses ConversationManager to drive multi-turn flow.
These helpers provide simple yes/no confirmation and input prompts if needed.
"""

from typing import Optional

from rich.prompt import Prompt, Confirm


def get_user_input(prompt: str = "You") -> str:
    return Prompt.ask(f"[cyan]{prompt}[/]")


def ask_yes_no(question: str, default: bool = True) -> bool:
    return Confirm.ask(f"[bold]{question}[/]", default=default)


def confirm_parameters() -> bool:
    return ask_yes_no("Proceed with these parameters?", default=True)


def get_parameter_edit(param_name: str) -> str:
    return Prompt.ask(f"Enter new value for [bold]{param_name}[/]")

