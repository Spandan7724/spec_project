from __future__ import annotations

import re
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List

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
from .response_formatter import ResponseFormatter
from src.config import get_config, load_config
from src.utils.errors import ConfigurationError


logger = logging.getLogger(__name__)


class ConversationManager:
    """Manage multi-turn conversation flow and session state."""

    def __init__(self, extractor: Optional[NLUExtractor] = None):
        self.extractor = extractor or NLUExtractor()
        self.sessions: Dict[str, ConversationSession] = {}
        self.history_limit = self._load_history_limit()

    def _load_history_limit(self) -> Optional[int]:
        """
        Load conversation history limit from configuration.

        Returns:
            Optional[int]: Number of messages to include, or None for unlimited.
        """
        cfg = None
        try:
            cfg = get_config()
        except ConfigurationError:
            try:
                cfg = load_config()
            except Exception:
                cfg = None
        except Exception:
            cfg = None

        default_limit = 8
        if not cfg:
            return default_limit

        history_cfg = cfg.get("chat.history", {})
        if not isinstance(history_cfg, dict):
            return default_limit

        use_max = history_cfg.get("max")
        if isinstance(use_max, str):
            use_max = use_max.lower() in {"true", "1", "yes", "on", "max"}
        if use_max:
            return None

        messages_val = history_cfg.get("messages")
        if isinstance(messages_val, int) and messages_val > 0:
            return messages_val

        return default_limit

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

        # Chat continuity: Check if session has completed analysis
        if session.state == ConversationState.RESULTS_READY and session.analysis_result:
            # This is a follow-up question about analysis
            return self.handle_followup_question(session.session_id, request.user_input)

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

        # Chat continuity: Check if session has completed analysis
        if session.state == ConversationState.RESULTS_READY and session.analysis_result:
            # This is a follow-up question about analysis
            return await self.ahandle_followup_question(session.session_id, request.user_input)

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
            # Accept either categorical timeframe or flexible canonical fields
            if (
                extracted.timeframe is None
                and extracted.timeframe_days is None
                and not extracted.deadline_utc
                and not extracted.window_days
                and extracted.timeframe_hours is None
            ):
                error_msg = (
                    "Please provide a timeframe (e.g., immediate, 1_day, 1_week, 1_month, "
                    "'in 10 days', 'by 2025-11-15', '3-5 days', 'in 12 hours')."
                )
            else:
                if extracted.timeframe:
                    session.parameters.timeframe = extracted.timeframe
                    session.parameters.timeframe_days = timeframe_to_days(extracted.timeframe)
                else:
                    session.parameters.timeframe = None
                    session.parameters.timeframe_days = extracted.timeframe_days
                # Copy flexible fields
                session.parameters.timeframe_mode = extracted.timeframe_mode
                session.parameters.deadline_utc = extracted.deadline_utc
                session.parameters.window_days = extracted.window_days
                session.parameters.time_unit = extracted.time_unit
                session.parameters.timeframe_hours = extracted.timeframe_hours

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
            if (
                extracted.timeframe is None
                and extracted.timeframe_days is None
                and not extracted.deadline_utc
                and not extracted.window_days
                and extracted.timeframe_hours is None
            ):
                error_msg = (
                    "Please provide a timeframe (e.g., immediate, 1_day, 1_week, 1_month, "
                    "'in 10 days', 'by 2025-11-15', '3-5 days', 'in 12 hours')."
                )
            else:
                if extracted.timeframe:
                    session.parameters.timeframe = extracted.timeframe
                    session.parameters.timeframe_days = timeframe_to_days(extracted.timeframe)
                else:
                    session.parameters.timeframe = None
                    session.parameters.timeframe_days = extracted.timeframe_days
                session.parameters.timeframe_mode = extracted.timeframe_mode
                session.parameters.deadline_utc = extracted.deadline_utc
                session.parameters.window_days = extracted.window_days
                session.parameters.time_unit = extracted.time_unit
                session.parameters.timeframe_hours = extracted.timeframe_hours

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
            if p.deadline_utc:
                return f"✓ Timeframe: by {p.deadline_utc}"
            if p.window_days:
                try:
                    s = int(p.window_days.get("start", 0))
                    e = int(p.window_days.get("end", 0))
                    return f"✓ Timeframe: {s}-{e} days"
                except Exception:
                    pass
            if p.timeframe_hours is not None and (p.timeframe_days is None or int(p.timeframe_days) == 0):
                try:
                    h = int(p.timeframe_hours)
                    unit = "hour" if h == 1 else "hours"
                    return f"✓ Timeframe: {h} {unit}"
                except Exception:
                    return f"✓ Timeframe: {p.timeframe_hours} hours"
            if p.timeframe_days is not None:
                label = "day" if p.timeframe_days == 1 else "days"
                return f"✓ Timeframe: {p.timeframe_days} {label}"
            return "✓ Timeframe: —"
        return ""

    def update_with_analysis_result(
        self,
        session_id: str,
        correlation_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Update session with completed analysis results (for chat continuity)."""
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for analysis update")
            return

        session.analysis_correlation_id = correlation_id
        session.analysis_result = result
        session.state = ConversationState.RESULTS_READY
        session.last_updated = datetime.now()

        # Format summary for chat display
        formatter = ResponseFormatter()
        summary = formatter.format_recommendation(result)
        session.result_summary = summary

        # Add to message history
        message = f"""Analysis Complete!

{summary}

You can now:
• Ask me questions about this analysis
• Request clarification on any recommendations
• View the detailed dashboard with charts and evidence

What would you like to know?"""

        session.add_message("assistant", message)

        logger.info(
            f"Updated session {session_id} with analysis results. "
            f"Correlation: {correlation_id}, Action: {result.get('action')}"
        )

    def handle_followup_question(
        self,
        session_id: str,
        question: str
    ) -> SupervisorResponse:
        """Answer questions about completed analysis using LLM with context."""
        session = self.sessions[session_id]
        result = session.analysis_result

        if not result:
            return SupervisorResponse(
                session_id=session_id,
                state=session.state,
                message="I don't have analysis results to discuss. Would you like to start a new conversion request?",
                requires_input=True,
                parameters=None,
                recommendation=None
            )

        # Check for reset/new request keywords
        reset_keywords = ['new request', 'start over', 'new analysis', 'different conversion']
        if any(kw in question.lower() for kw in reset_keywords):
            # Reset session to initial state
            session.state = ConversationState.INITIAL
            session.parameters = ExtractedParameters()
            session.analysis_result = None
            session.analysis_correlation_id = None
            return SupervisorResponse(
                session_id=session_id,
                state=ConversationState.INITIAL,
                message="Let's start fresh! What currency pair would you like to analyze?",
                requires_input=True,
                parameters=None,
                recommendation=None
            )

        # Build context from analysis results and recent conversation
        context = self._build_analysis_context(result, session.parameters)
        history_context = self._build_conversation_history(session.conversation_history)

        # Use LLM to answer question with context
        answer = self._answer_with_llm(context, history_context, question)

        # Add to message history
        session.add_message("assistant", answer)
        session.last_updated = datetime.now()

        return SupervisorResponse(
            session_id=session_id,
            state=ConversationState.RESULTS_READY,
            message=answer,
            requires_input=True,
            parameters=session.parameters,
            recommendation=result
        )

    async def ahandle_followup_question(
        self,
        session_id: str,
        question: str
    ) -> SupervisorResponse:
        """Async version of handle_followup_question."""
        session = self.sessions[session_id]
        result = session.analysis_result

        if not result:
            return SupervisorResponse(
                session_id=session_id,
                state=session.state,
                message="I don't have analysis results to discuss. Would you like to start a new conversion request?",
                requires_input=True,
                parameters=None,
                recommendation=None
            )

        # Check for reset/new request keywords
        reset_keywords = ['new request', 'start over', 'new analysis', 'different conversion']
        if any(kw in question.lower() for kw in reset_keywords):
            # Reset session to initial state
            session.state = ConversationState.INITIAL
            session.parameters = ExtractedParameters()
            session.analysis_result = None
            session.analysis_correlation_id = None
            return SupervisorResponse(
                session_id=session_id,
                state=ConversationState.INITIAL,
                message="Let's start fresh! What currency pair would you like to analyze?",
                requires_input=True,
                parameters=None,
                recommendation=None
            )

        # Build context from analysis results
        context = self._build_analysis_context(result, session.parameters)
        history_context = self._build_conversation_history(session.conversation_history)

        # Use LLM to answer question with context (async)
        answer = await self._aanswer_with_llm(context, history_context, question)

        # Add to message history
        session.add_message("assistant", answer)
        session.last_updated = datetime.now()

        return SupervisorResponse(
            session_id=session_id,
            state=ConversationState.RESULTS_READY,
            message=answer,
            requires_input=True,
            parameters=session.parameters,
            recommendation=result
        )

    def _build_analysis_context(
        self,
        result: Dict[str, Any],
        parameters: Optional[ExtractedParameters] = None
    ) -> str:
        """Build comprehensive context from analysis results and captured parameters."""
        # Extract key information
        action = result.get('action', 'unknown')
        confidence_val = result.get('confidence')
        if not isinstance(confidence_val, (int, float)):
            confidence_val = 0
        timeline = result.get('timeline', 'Not specified')
        rationale = result.get('rationale') or []
        warnings = result.get('warnings') or []

        risk_summary = result.get('risk_summary') or {}
        cost_estimate = result.get('cost_estimate') or {}
        expected_outcome = result.get('expected_outcome') or {}

        metadata = result.get('metadata') or {}

        # Build request summary from parameters/metadata
        currency_pair = None
        base_currency = None
        quote_currency = None
        amount = None
        urgency = None
        risk_tolerance = None
        timeframe = None

        if parameters:
            currency_pair = parameters.currency_pair or currency_pair
            base_currency = parameters.base_currency or base_currency
            quote_currency = parameters.quote_currency or quote_currency
            amount = parameters.amount if parameters.amount is not None else amount
            urgency = parameters.urgency or urgency
            risk_tolerance = parameters.risk_tolerance or risk_tolerance
            timeframe = parameters.timeframe or timeframe

        currency_pair = currency_pair or metadata.get('currency_pair')
        base_currency = base_currency or metadata.get('base_currency')
        quote_currency = quote_currency or metadata.get('quote_currency')
        amount = amount if amount is not None else metadata.get('amount')
        urgency = urgency or metadata.get('urgency')
        risk_tolerance = risk_tolerance or metadata.get('risk_tolerance')
        timeframe = timeframe or metadata.get('timeframe')

        # Format context
        request_lines = []
        if currency_pair:
            request_lines.append(f"Currency Pair: {currency_pair}")
        else:
            if base_currency and quote_currency:
                request_lines.append(f"Currencies: {base_currency} to {quote_currency}")
            elif base_currency:
                request_lines.append(f"Base Currency: {base_currency}")
            elif quote_currency:
                request_lines.append(f"Quote Currency: {quote_currency}")
        if amount:
            try:
                request_lines.append(f"Amount: {float(amount):,.2f}")
            except (TypeError, ValueError):
                request_lines.append(f"Amount: {amount}")
        if urgency:
            request_lines.append(f"Urgency: {urgency}")
        if risk_tolerance:
            request_lines.append(f"Risk tolerance: {risk_tolerance}")
        if timeframe:
            request_lines.append(f"Timeframe: {timeframe}")
        elif expected_outcome.get('timeframe'):
            request_lines.append(f"Timeframe: {expected_outcome['timeframe']}")

        request_summary = "\n".join(request_lines) if request_lines else "Not specified"

        context = f"""Conversation Context:

User Request:
{request_summary}

Analysis Results Summary:

**Recommendation:** {action.replace('_', ' ').title()}
**Confidence Level:** {confidence_val * 100:.1f}%
**Timeline:** {timeline}

**Rationale:**
"""
        for i, point in enumerate(rationale, 1):
            context += f"{i}. {point}\n"

        if warnings:
            context += "\n**Warnings:**\n"
            for warning in warnings:
                context += f"⚠ {warning}\n"

        context += f"""
**Risk Assessment:**
- Risk Level: {risk_summary.get('risk_level', 'Unknown')}
- Volatility (30d): {risk_summary.get('realized_vol_30d', 'N/A')}
- Event Risk: {risk_summary.get('event_risk', 'N/A')}

**Cost Estimate:**
- Spread: {cost_estimate.get('spread_bps', 'N/A')} bps
- Fees: {cost_estimate.get('fee_bps', 'N/A')} bps
- Total: {cost_estimate.get('total_bps', 'N/A')} bps

**Expected Outcome:**
- Expected Rate: {expected_outcome.get('expected_rate', 'N/A')}
- Potential Improvement: {expected_outcome.get('expected_improvement_bps', 'N/A')} bps
"""

        # Add evidence summary if available
        evidence = result.get('evidence') or {}
        if evidence:
            context += "\n**Supporting Evidence:**\n"

        market = evidence.get('market') or {}
        if market:
            context += f"- Current Rate: {market.get('mid_rate', 'N/A')}\n"
            regime = market.get('regime') or {}
            context += f"- Market Regime: {regime.get('regime_name', 'N/A')}\n"

        intel = evidence.get('intelligence') or {}
        if intel:
            context += f"- News Sentiment: {intel.get('pair_bias', 'N/A')}\n"
            next_event = intel.get('next_high_event')
            if next_event:
                context += f"- Next High-Impact Event: {next_event}\n"

        return context

    def _build_conversation_history(
        self,
        history: List[Dict[str, str]],
        max_messages: Optional[int] = None,
        exclude_last: bool = True,
    ) -> str:
        """Format recent conversation history for LLM context."""

        if not history:
            return "No prior conversation."

        trimmed_history = history[:-1] if exclude_last else history
        if not trimmed_history:
            return "No prior conversation."

        limit = max_messages if max_messages is not None else self.history_limit
        recent_messages = trimmed_history if limit is None else trimmed_history[-limit:]
        lines: List[str] = []
        for message in recent_messages:
            role = message.get("role", "assistant")
            speaker = "User" if role == "user" else "Assistant"
            content = message.get("content", "")
            lines.append(f"{speaker}: {content}")

        return "\n".join(lines)

    def _answer_with_llm(self, context: str, history: str, question: str) -> str:
        """Use LLM to generate contextual answer."""
        import asyncio

        system_prompt = """You are a helpful currency trading assistant.
You have just completed an analysis for the user and they are asking follow-up questions.

Use both the prior conversation history and the analysis results to answer clearly and concisely.
- Reference the analysis data wherever relevant
- Use the exact numbers and facts from the results
- If the question is about something not in the conversation or analysis, say so politely
- Keep answers conversational but professional
- Format numbers clearly (e.g., "85% confidence", "15 basis points")
"""

        history_section = history or "No prior conversation."

        user_prompt = f"""Here is the recent conversation history:

{history_section}

Here are the analysis results:

{context}

User Question: {question}

Answer the question using the conversation history and analysis results above. Be specific and helpful."""

        try:
            # Use async LLM helper (run synchronously for backward compatibility)
            from src.llm.agent_helpers import chat_with_model_for_task
            from src.llm.manager import LLMManager

            llm_manager = LLMManager()

            # Run async function synchronously
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context - create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    response = loop.run_in_executor(
                        pool,
                        lambda: asyncio.run(chat_with_model_for_task(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            task_type="conversation",
                            llm_manager=llm_manager
                        ))
                    )
                    response = asyncio.create_task(response)
                    # This won't work - fall back to async version
                    raise RuntimeError("Cannot run sync in async context")
            else:
                # Not in async context - run directly
                response = asyncio.run(chat_with_model_for_task(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    task_type="conversation",
                    llm_manager=llm_manager
                ))

            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM follow-up answer failed: {e}")
            return f"I understand you're asking about the analysis, but I'm having trouble generating a response right now. Could you rephrase your question?"

    async def _aanswer_with_llm(self, context: str, history: str, question: str) -> str:
        """Async version of _answer_with_llm."""
        from src.llm.agent_helpers import chat_with_model_for_task

        system_prompt = """You are a helpful currency trading assistant.
You have just completed an analysis for the user and they are asking follow-up questions.

Use both the prior conversation history and the analysis results to answer clearly and concisely.
- Reference the analysis data wherever relevant
- Use the exact numbers and facts from the results
- If the question is about something not in the conversation or analysis, say so politely
- Keep answers conversational but professional
- Format numbers clearly (e.g., "85% confidence", "15 basis points")
"""

        history_section = history or "No prior conversation."

        user_prompt = f"""Here is the recent conversation history:

{history_section}

Here are the analysis results:

{context}

User Question: {question}

Answer the question using the conversation history and analysis results above. Be specific and helpful."""

        try:
            # Use async LLM helper with proper task type
            from src.llm.manager import LLMManager
            llm_manager = LLMManager()

            response = await chat_with_model_for_task(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                task_type="conversation",
                llm_manager=llm_manager
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM async follow-up answer failed: {e}")
            return f"I understand you're asking about the analysis, but I'm having trouble generating a response right now. Could you rephrase your question?"
