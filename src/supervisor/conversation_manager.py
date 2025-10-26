from __future__ import annotations

import re
import uuid
import logging
from typing import Dict, Optional, List

from .models import (
    ConversationSession,
    ConversationState,
    ExtractedParameters,
    SupervisorRequest,
    SupervisorResponse,
)
from .nlu_extractor import NLUExtractor
from .validation import validate_currency_pair, timeframe_to_days
from .message_templates import (
    CONFIRMATION_PROMPT,
    RESTART_MESSAGE,
    get_parameter_prompt,
    ERROR_MESSAGES,
    GREETING_MESSAGE,
)


logger = logging.getLogger(__name__)


class ConversationManager:
    """Manage multi-turn conversation flow and session state."""

    def __init__(self, extractor: Optional[NLUExtractor] = None):
        self.extractor = extractor or NLUExtractor()
        self.sessions: Dict[str, ConversationSession] = {}

    def process_input(self, request: SupervisorRequest) -> SupervisorResponse:
        """Process user input and return appropriate response."""

        # Get or create session
        session = self._get_or_create_session(request.session_id)

        # Add user message to history
        session.add_message("user", request.user_input)

        # Global restart command from any state
        if request.user_input.strip().lower() in {"restart", "start over", "reset"}:
            session.state = ConversationState.INITIAL
            session.parameters = ExtractedParameters()
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message=RESTART_MESSAGE,
                requires_input=True,
                parameters=session.parameters,
            )

        # Route based on state
        if session.state == ConversationState.INITIAL:
            return self._handle_initial(session, request.user_input)

        if session.state == ConversationState.CONFIRMING:
            return self._handle_confirmation(session, request.user_input)

        if session.state in {
            ConversationState.COLLECTING_CURRENCY_PAIR,
            ConversationState.COLLECTING_AMOUNT,
            ConversationState.COLLECTING_RISK,
            ConversationState.COLLECTING_URGENCY,
            ConversationState.COLLECTING_TIMEFRAME,
        }:
            return self._handle_collection(session, request.user_input)

        # Fallback invalid state
        return SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message="Invalid session state. Please type 'restart' to begin again.",
            requires_input=True,
        )

    async def aprocess_input(self, request: SupervisorRequest) -> SupervisorResponse:
        """Async variant to be used when already running in an event loop."""
        session = self._get_or_create_session(request.session_id)
        session.add_message("user", request.user_input)

        if request.user_input.strip().lower() in {"restart", "start over", "reset"}:
            session.state = ConversationState.INITIAL
            session.parameters = ExtractedParameters()
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message=RESTART_MESSAGE,
                requires_input=True,
                parameters=session.parameters,
            )

        if session.state == ConversationState.INITIAL:
            return await self._ahandle_initial(session, request.user_input)

        if session.state == ConversationState.CONFIRMING:
            return self._handle_confirmation(session, request.user_input)

        if session.state in {
            ConversationState.COLLECTING_CURRENCY_PAIR,
            ConversationState.COLLECTING_AMOUNT,
            ConversationState.COLLECTING_RISK,
            ConversationState.COLLECTING_URGENCY,
            ConversationState.COLLECTING_TIMEFRAME,
        }:
            return await self._ahandle_collection(session, request.user_input)

        return SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message="Invalid session state. Please type 'restart' to begin again.",
            requires_input=True,
        )

    # ---- State handlers ----
    def _handle_initial(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Process initial user query."""

        extracted = self.extractor.extract(user_input)
        session.parameters = extracted

        # Build acknowledgement
        parts: List[str] = []
        if extracted.currency_pair:
            parts.append(f"I'll help you convert {extracted.currency_pair}.")
            parts.append("I've noted:")
            if extracted.amount is not None:
                parts.append(
                    f"✓ Amount: {extracted.amount:,.2f} {extracted.base_currency}"
                )
            else:
                parts.append(f"✓ Currency pair: {extracted.currency_pair}")
            if extracted.urgency:
                parts.append(f"✓ Urgency: {extracted.urgency.capitalize()}")
            if extracted.risk_tolerance:
                parts.append(f"✓ Risk tolerance: {extracted.risk_tolerance.capitalize()}")
            if extracted.timeframe:
                parts.append(f"✓ Timeframe: {extracted.timeframe}")
        else:
            parts.append(GREETING_MESSAGE)

        # Determine next step
        missing = extracted.missing_parameters()
        if not missing:
            # All parameters collected → confirmation
            return self._move_to_confirmation(session, prefix="\n".join(parts) + "\n\n")

        # Ask for first missing parameter
        return self._ask_for_parameter(session, missing[0], prefix_parts=parts)

    def _handle_collection(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        """Collect specific parameter based on current state."""

        param_state_map = {
            ConversationState.COLLECTING_CURRENCY_PAIR: "currency_pair",
            ConversationState.COLLECTING_AMOUNT: "amount",
            ConversationState.COLLECTING_RISK: "risk_tolerance",
            ConversationState.COLLECTING_URGENCY: "urgency",
            ConversationState.COLLECTING_TIMEFRAME: "timeframe",
        }
        param_name = param_state_map[session.state]

        # Extract potentially updated value using NLU
        extracted = self.extractor.extract(user_input)
        
        # Update the targeted parameter
        error_msg: Optional[str] = None
        if param_name == "currency_pair":
            base = extracted.base_currency or session.parameters.base_currency
            quote = extracted.quote_currency or session.parameters.quote_currency
            ok, err = validate_currency_pair(base, quote)
            if not ok:
                error_msg = err or ERROR_MESSAGES.get("invalid_pair", "Invalid currency pair.")
            else:
                session.parameters.base_currency = base
                session.parameters.quote_currency = quote
                session.parameters.currency_pair = f"{base}/{quote}"
        elif param_name == "amount":
            amt = extracted.amount
            if amt is None or amt <= 0:
                error_msg = ERROR_MESSAGES.get("invalid_amount", "Please provide a valid positive amount (e.g., 5000).")
            else:
                session.parameters.amount = float(amt)
        elif param_name == "risk_tolerance":
            if extracted.risk_tolerance is None:
                error_msg = "Please specify: conservative, moderate, or aggressive."
            else:
                session.parameters.risk_tolerance = extracted.risk_tolerance
        elif param_name == "urgency":
            if extracted.urgency is None:
                error_msg = "Please specify: urgent, normal, or flexible."
            else:
                session.parameters.urgency = extracted.urgency
        elif param_name == "timeframe":
            if extracted.timeframe is None:
                error_msg = "Please specify: immediate, 1_day, 1_week, or 1_month."
            else:
                session.parameters.timeframe = extracted.timeframe
                session.parameters.timeframe_days = timeframe_to_days(extracted.timeframe)

        if error_msg:
            # Re-ask the same parameter with the error message
            return self._ask_for_parameter(session, param_name, prefix_parts=[error_msg])

        # Confirm set and move on
        confirm_line = self._format_param_line(session, param_name)

        # Check if now complete
        missing = session.parameters.missing_parameters()
        if not missing:
            return self._move_to_confirmation(session, prefix=f"{confirm_line}\n\n")

        # Ask for next missing parameter
        return self._ask_for_parameter(session, missing[0], prefix_parts=[confirm_line])

    def _handle_confirmation(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        low = user_input.strip().lower()
        if low in {"yes", "y", "confirm", "proceed"}:
            session.state = ConversationState.PROCESSING
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message="Confirmed. Proceeding with analysis.",
                requires_input=False,
                parameters=session.parameters,
            )

        if low.startswith("change"):
            return self._handle_parameter_change(session, user_input)

        if low in {"restart", "start over", "reset"}:
            session.state = ConversationState.INITIAL
            session.parameters = ExtractedParameters()
            return SupervisorResponse(
                session_id=session.session_id,
                state=session.state,
                message=RESTART_MESSAGE,
                requires_input=True,
                parameters=session.parameters,
            )

        # Repeat prompt
        return self._move_to_confirmation(session, prefix="I didn't catch that.\n\n")

    # ---- Helpers ----
    def _get_or_create_session(self, session_id: Optional[str]) -> ConversationSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        new_id = session_id or str(uuid.uuid4())
        session = ConversationSession(
            session_id=new_id,
            state=ConversationState.INITIAL,
            parameters=ExtractedParameters(),
        )
        self.sessions[new_id] = session
        return session

    def _move_to_confirmation(self, session: ConversationSession, prefix: str = "") -> SupervisorResponse:
        session.state = ConversationState.CONFIRMING
        lines = []
        if prefix:
            lines.append(prefix.strip())
        lines.append("Please confirm the parameters:")
        lines.append(self._format_param_line(session, "currency_pair"))
        lines.append(self._format_param_line(session, "amount"))
        lines.append(self._format_param_line(session, "risk_tolerance"))
        lines.append(self._format_param_line(session, "urgency"))
        lines.append(self._format_param_line(session, "timeframe"))
        lines.append("")
        lines.append(CONFIRMATION_PROMPT)
        message = "\n".join([l for l in lines if l])
        session.add_message("assistant", message)
        return SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message=message,
            requires_input=True,
            parameters=session.parameters,
        )

    # Async counterparts
    async def _ahandle_initial(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        extracted = await self.extractor.aextract(user_input)
        session.parameters = extracted

        parts: List[str] = []
        if extracted.currency_pair:
            parts.append(f"I'll help you convert {extracted.currency_pair}.")
            parts.append("I've noted:")
            if extracted.amount is not None:
                parts.append(f"✓ Amount: {extracted.amount:,.2f} {extracted.base_currency}")
            else:
                parts.append(f"✓ Currency pair: {extracted.currency_pair}")
            if extracted.urgency:
                parts.append(f"✓ Urgency: {extracted.urgency.capitalize()}")
            if extracted.risk_tolerance:
                parts.append(f"✓ Risk tolerance: {extracted.risk_tolerance.capitalize()}")
            if extracted.timeframe:
                parts.append(f"✓ Timeframe: {extracted.timeframe}")
        else:
            parts.append(GREETING_MESSAGE)

        missing = extracted.missing_parameters()
        if not missing:
            return self._move_to_confirmation(session, prefix="\n".join(parts) + "\n\n")
        return self._ask_for_parameter(session, missing[0], prefix_parts=parts)

    async def _ahandle_collection(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        param_state_map = {
            ConversationState.COLLECTING_CURRENCY_PAIR: "currency_pair",
            ConversationState.COLLECTING_AMOUNT: "amount",
            ConversationState.COLLECTING_RISK: "risk_tolerance",
            ConversationState.COLLECTING_URGENCY: "urgency",
            ConversationState.COLLECTING_TIMEFRAME: "timeframe",
        }
        param_name = param_state_map[session.state]

        extracted = await self.extractor.aextract(user_input)
        error_msg: Optional[str] = None
        if param_name == "currency_pair":
            base = extracted.base_currency or session.parameters.base_currency
            quote = extracted.quote_currency or session.parameters.quote_currency
            ok, err = validate_currency_pair(base, quote)
            if not ok:
                error_msg = err or ERROR_MESSAGES.get("invalid_pair", "Invalid currency pair.")
            else:
                session.parameters.base_currency = base
                session.parameters.quote_currency = quote
                session.parameters.currency_pair = f"{base}/{quote}"
        elif param_name == "amount":
            amt = extracted.amount
            if amt is None or amt <= 0:
                error_msg = ERROR_MESSAGES.get("invalid_amount", "Please provide a valid positive amount (e.g., 5000).")
            else:
                session.parameters.amount = float(amt)
        elif param_name == "risk_tolerance":
            if extracted.risk_tolerance is None:
                error_msg = "Please specify: conservative, moderate, or aggressive."
            else:
                session.parameters.risk_tolerance = extracted.risk_tolerance
        elif param_name == "urgency":
            if extracted.urgency is None:
                error_msg = "Please specify: urgent, normal, or flexible."
            else:
                session.parameters.urgency = extracted.urgency
        elif param_name == "timeframe":
            if extracted.timeframe is None:
                error_msg = "Please specify: immediate, 1_day, 1_week, or 1_month."
            else:
                session.parameters.timeframe = extracted.timeframe
                session.parameters.timeframe_days = timeframe_to_days(extracted.timeframe)

        if error_msg:
            return self._ask_for_parameter(session, param_name, prefix_parts=[error_msg])

        confirm_line = self._format_param_line(session, param_name)
        missing = session.parameters.missing_parameters()
        if not missing:
            return self._move_to_confirmation(session, prefix=f"{confirm_line}\n\n")
        return self._ask_for_parameter(session, missing[0], prefix_parts=[confirm_line])

    def _ask_for_parameter(
        self,
        session: ConversationSession,
        param_name: str,
        prefix_parts: Optional[List[str]] = None,
    ) -> SupervisorResponse:
        state_map = {
            "currency_pair": ConversationState.COLLECTING_CURRENCY_PAIR,
            "amount": ConversationState.COLLECTING_AMOUNT,
            "risk_tolerance": ConversationState.COLLECTING_RISK,
            "urgency": ConversationState.COLLECTING_URGENCY,
            "timeframe": ConversationState.COLLECTING_TIMEFRAME,
        }

        session.state = state_map[param_name]
        parts = []
        if prefix_parts:
            parts.extend(prefix_parts)
        parts.append(get_parameter_prompt(param_name))
        message = "\n".join(parts)
        session.add_message("assistant", message)
        return SupervisorResponse(
            session_id=session.session_id,
            state=session.state,
            message=message,
            requires_input=True,
            parameters=session.parameters,
        )

    def _handle_parameter_change(self, session: ConversationSession, user_input: str) -> SupervisorResponse:
        # Parse pattern: change <param> to <value>
        m = re.match(r"\s*change\s+(\w+)\s*(?:to\s+(.+))?", user_input.strip(), re.IGNORECASE)
        if not m:
            return self._move_to_confirmation(session, prefix="I couldn't parse the change request.\n\n")
        name_raw, value_raw = m.groups()
        name = name_raw.lower()

        # Map possible aliases
        alias = {
            "risk": "risk_tolerance",
            "time": "timeframe",
            "pair": "currency_pair",
        }
        param = alias.get(name, name)
        if param not in {"currency_pair", "amount", "risk_tolerance", "urgency", "timeframe"}:
            return self._move_to_confirmation(session, prefix=f"Unknown parameter '{name}'.\n\n")

        if not value_raw:
            # Ask for the new value for that parameter
            return self._ask_for_parameter(session, param, prefix_parts=[f"Let's update {param}."]) 

        # Extract from provided value
        extracted = self.extractor.extract(value_raw)

        # Reuse collection logic by setting state and delegating
        state_map = {
            "currency_pair": ConversationState.COLLECTING_CURRENCY_PAIR,
            "amount": ConversationState.COLLECTING_AMOUNT,
            "risk_tolerance": ConversationState.COLLECTING_RISK,
            "urgency": ConversationState.COLLECTING_URGENCY,
            "timeframe": ConversationState.COLLECTING_TIMEFRAME,
        }
        session.state = state_map[param]
        return self._handle_collection(session, value_raw)

    def _format_param_line(self, session: ConversationSession, name: str) -> str:
        p = session.parameters
        if name == "currency_pair":
            return f"✓ Currency pair: {p.currency_pair or '—'}"
        if name == "amount":
            return (
                f"✓ Amount: {p.amount:,.2f} {p.base_currency}" if p.amount is not None else "✓ Amount: —"
            )
        if name == "risk_tolerance":
            return f"✓ Risk tolerance: {p.risk_tolerance.capitalize()}" if p.risk_tolerance else "✓ Risk tolerance: —"
        if name == "urgency":
            return f"✓ Urgency: {p.urgency.capitalize()}" if p.urgency else "✓ Urgency: —"
        if name == "timeframe":
            if p.timeframe:
                return f"✓ Timeframe: {p.timeframe}"
            if p.timeframe_days is not None:
                label = "day" if p.timeframe_days == 1 else "days"
                return f"✓ Timeframe: {p.timeframe_days} {label}"
            return "✓ Timeframe: —"
        return ""
