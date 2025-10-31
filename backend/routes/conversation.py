from __future__ import annotations

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

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
            # Chat continuity: analysis results if available
            "correlation_id": session.analysis_correlation_id,
            "has_results": session.analysis_result is not None,
            "result_summary": session.result_summary,
            "recommendation": session.analysis_result,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.last_updated.isoformat() if session.last_updated else None,
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


@router.post("/message/stream")
async def process_message_stream(input: ConversationInput, cm=Depends(get_conversation_manager)):
    """
    Stream conversation responses to the client using Server-Sent Events (SSE).

    Note: Currently streams the response word-by-word for immediate feedback.
    Future enhancement: Stream directly from LLM providers for true real-time responses.
    """
    async def generate():
        try:
            # Process request (async)
            req = SupervisorRequest(user_input=input.user_input, session_id=input.session_id)
            resp = await cm.aprocess_input(req)

            # Prepare response data
            response_data = {
                "session_id": resp.session_id,
                "state": resp.state.value,
                "message": resp.message,
                "requires_input": resp.requires_input,
                "parameters": (resp.parameters.__dict__ if resp.parameters else None),
            }

            # Stream the message word by word for better UX
            words = resp.message.split()
            accumulated_message = ""

            for i, word in enumerate(words):
                accumulated_message += word
                if i < len(words) - 1:
                    accumulated_message += " "

                # Send chunk with accumulated message
                chunk_data = {
                    **response_data,
                    "message": accumulated_message,
                    "is_complete": False
                }

                yield f"data: {json.dumps(chunk_data)}\n\n"

                # Small delay for smoother streaming effect
                import asyncio
                await asyncio.sleep(0.05)

            # Send final complete message
            final_data = {
                **response_data,
                "is_complete": True
            }
            yield f"data: {json.dumps(final_data)}\n\n"

            # Send done signal
            yield "data: [DONE]\n\n"

        except Exception as e:  # noqa: BLE001
            error_data = {
                "error": str(e),
                "message": "Sorry, I encountered an error. Please try again.",
                "is_complete": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

