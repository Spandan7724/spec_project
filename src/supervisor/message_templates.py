"""Reusable message templates for Supervisor interactions."""
from __future__ import annotations

from typing import Dict


GREETING_MESSAGE = (
    "Welcome to the Currency Assistant. I can help you time your currency conversion.\n"
    "Tell me what you want to convert (e.g., 'Convert 5000 USD to EUR today')."
)

CONFIRMATION_PROMPT = (
    "Is this correct?\n- Type 'yes' to proceed\n- Type 'change <parameter>' to modify\n- Type 'restart' to start over"
)

RESTART_MESSAGE = (
    "Conversation restarted. Please describe your conversion (e.g., 'Convert 5000 USD to EUR')."
)

ERROR_MESSAGES: Dict[str, str] = {
    "invalid_pair": "That doesn't look like a valid currency pair. Please use codes like USD/EUR.",
    "invalid_amount": "Please provide a positive amount (e.g., 5000).",
    "unknown_command": "I didn't recognize that command.",
}

HELP_TEXT = (
    "You can say things like:\n"
    "- 'Convert 5000 USD to EUR today'\n"
    "- 'I want to move 10k dollars to euros next week'\n"
    "- 'Change risk to aggressive'\n"
)


def get_processing_message(stage: str) -> str:
    stage_map = {
        "market_data": "Fetching real-time market data...",
        "market_intelligence": "Analyzing market news and events...",
        "prediction": "Evaluating price predictions...",
        "decision": "Synthesizing decision...",
    }
    return stage_map.get(stage, "Processing...")


def get_parameter_prompt(param_name: str) -> str:
    prompts = {
        "currency_pair": "What currency pair? (e.g., USD/EUR)",
        "amount": "What amount? (e.g., 5000)",
        "risk_tolerance": "What is your risk tolerance? (conservative/moderate/aggressive)",
        "urgency": "What is your urgency? (urgent/normal/flexible)",
        "timeframe": "What is your timeframe? (e.g., immediate, 1_day, 1_week, 1_month, 'in 10 days', 'by 2025-11-15', '3-5 days', 'in 12 hours')",
    }
    return prompts.get(param_name, "Could you clarify?")
