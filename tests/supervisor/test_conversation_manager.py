import os
import re

from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.models import SupervisorRequest, ConversationState
from src.supervisor.nlu_extractor import NLUExtractor


def make_manager_offline() -> ConversationManager:
    # Ensure offline (no LLM) for deterministic tests
    os.environ["OFFLINE_DEMO"] = "true"
    extractor = NLUExtractor(use_llm=False)
    return ConversationManager(extractor=extractor)


def test_initial_full_query_goes_to_confirmation():
    mgr = make_manager_offline()
    req = SupervisorRequest(user_input="I need to convert 5000 USD to EUR today, moderate risk")
    resp = mgr.process_input(req)

    assert resp.state == ConversationState.CONFIRMING
    assert resp.requires_input is True
    assert "Currency pair" in resp.message
    assert "Amount" in resp.message
    assert "Risk tolerance" in resp.message
    assert "Urgency" in resp.message or "Timeframe" in resp.message


def test_missing_amount_triggers_collection_then_confirmation():
    mgr = make_manager_offline()

    # First input without amount
    resp1 = mgr.process_input(SupervisorRequest(user_input="Convert USD to EUR"))
    assert resp1.state == ConversationState.COLLECTING_AMOUNT
    assert resp1.requires_input is True
    assert "What amount" in resp1.message

    # Provide amount
    resp2 = mgr.process_input(SupervisorRequest(user_input="5000", session_id=resp1.session_id))
    assert resp2.state in {ConversationState.COLLECTING_RISK, ConversationState.COLLECTING_URGENCY, ConversationState.COLLECTING_TIMEFRAME, ConversationState.CONFIRMING}
    assert resp2.requires_input is True


def test_confirmation_yes_transitions_to_processing():
    mgr = make_manager_offline()
    # Fill everything at once
    resp1 = mgr.process_input(SupervisorRequest(user_input="Convert 5000 USD to EUR today, urgent, moderate risk"))
    # Confirm
    resp2 = mgr.process_input(SupervisorRequest(user_input="yes", session_id=resp1.session_id))
    assert resp2.state == ConversationState.PROCESSING
    assert resp2.requires_input is False

