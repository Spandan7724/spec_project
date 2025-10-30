from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.models.requests import ConversationInput
from backend.models.responses import ConversationOutput
from backend.dependencies import get_conversation_manager
from src.supervisor.models import SupervisorRequest


router = APIRouter()


@router.post("/message", response_model=ConversationOutput)
def process_message(input: ConversationInput, cm=Depends(get_conversation_manager)):
    try:
        req = SupervisorRequest(user_input=input.user_input, session_id=input.session_id)
        resp = cm.process_input(req)
        return ConversationOutput(
            session_id=resp.session_id,
            state=resp.state.value,
            message=resp.message,
            requires_input=resp.requires_input,
            parameters=(resp.parameters.__dict__ if resp.parameters else None),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reset/{session_id}")
def reset_conversation(session_id: str, cm=Depends(get_conversation_manager)):
    try:
        # Recreate a blank session by removing it if present
        if session_id in cm.sessions:
            del cm.sessions[session_id]
        return {"status": "reset", "session_id": session_id}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/session/{session_id}")
def get_session(session_id: str, cm=Depends(get_conversation_manager)):
    try:
        session = cm.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "parameters": session.parameters.__dict__ if session.parameters else None,
            "history": session.conversation_history,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/sessions/active")
def get_active_sessions(cm=Depends(get_conversation_manager)):
    """Get all active conversation sessions."""
    try:
        sessions_list = []
        for session_id, session in cm.sessions.items():
            sessions_list.append({
                "session_id": session.session_id,
                "state": session.state.value,
                "has_parameters": bool(session.parameters and not session.parameters.missing_parameters()),
                "currency_pair": getattr(session.parameters, "currency_pair", None) if session.parameters else None,
                "message_count": len(session.conversation_history),
            })

        return {
            "total": len(sessions_list),
            "sessions": sessions_list,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

