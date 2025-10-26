"""Prompt helpers for Supervisor NLU extraction using LLMs."""
from __future__ import annotations

from typing import Dict, List


def get_system_prompt() -> str:
    return (
        "You are an NLU extraction assistant for a currency conversion advisor. "
        "Extract structured parameters from the user's message. "
        "When tools are provided, always respond by calling the tool with strictly validated JSON arguments. "
        "Never include explanatory text in the tool arguments. "
        "If an item is not present or ambiguous, set it to null."
    )


def get_user_prompt(user_text: str) -> str:
    return (
        "Extract the following parameters from the text: "
        "currency_pair (e.g., USD/EUR), base_currency, quote_currency, amount (number), "
        "risk_tolerance (conservative|moderate|aggressive), urgency (urgent|normal|flexible), "
        "timeframe (immediate|1_day|1_week|1_month), timeframe_days (int).\n\n"
        f"Text: {user_text}"
    )


def get_tools_schema() -> List[Dict]:
    """Return a single tool schema for structured parameter output."""
    return [
        {
            "type": "function",
            "function": {
                "name": "set_parameters",
                "description": "Record extracted parameters for currency conversion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "currency_pair": {"type": ["string", "null"]},
                        "base_currency": {"type": ["string", "null"]},
                        "quote_currency": {"type": ["string", "null"]},
                        "amount": {"type": ["number", "null"]},
                        "risk_tolerance": {
                            "type": ["string", "null"],
                            "enum": ["conservative", "moderate", "aggressive", None],
                        },
                        "urgency": {
                            "type": ["string", "null"],
                            "enum": ["urgent", "normal", "flexible", None],
                        },
                        "timeframe": {
                            "type": ["string", "null"],
                            "enum": ["immediate", "1_day", "1_week", "1_month", None],
                        },
                        "timeframe_days": {"type": ["integer", "null"]},
                    },
                    "required": [
                        "currency_pair",
                        "base_currency",
                        "quote_currency",
                        "amount",
                        "risk_tolerance",
                        "urgency",
                        "timeframe",
                        "timeframe_days",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]

