from __future__ import annotations

import uuid
from typing import Optional

from .conversation_manager import ConversationManager
from .agent_orchestrator import AgentOrchestrator
from .response_formatter import ResponseFormatter
from .models import SupervisorRequest, SupervisorResponse, ConversationState


class Supervisor:
    """Main entry for orchestrated supervisor flow (non-graph mode)."""

    def __init__(self):
        self.conversation = ConversationManager()
        self.orchestrator = AgentOrchestrator()
        self.formatter = ResponseFormatter()

    async def aprocess_request(self, request: SupervisorRequest) -> SupervisorResponse:
        # 1) Conversation pass (async to avoid nested loop issues)
        conv_resp = await self.conversation.aprocess_input(request)
        if conv_resp.requires_input:
            return conv_resp

        # 2) If just confirmed, run analysis
        if conv_resp.state == ConversationState.PROCESSING:
            correlation_id = request.correlation_id or str(uuid.uuid4())
            params = conv_resp.parameters
            assert params is not None
            # Orchestrate agents
            rec = await self.orchestrator.run_analysis(params, correlation_id)
            # Format
            message = self.formatter.format_recommendation(rec)
            # Return final
            return SupervisorResponse(
                session_id=conv_resp.session_id,
                state=ConversationState.COMPLETED,
                message=message,
                requires_input=False,
                parameters=params,
                recommendation=rec if rec.get("status") == "success" else None,
                warnings=rec.get("warnings", []),
                errors=([rec.get("error")] if rec.get("status") == "error" else []),
            )

        # Otherwise return conversation response
        return conv_resp

    def process_request(self, request: SupervisorRequest) -> SupervisorResponse:
        import asyncio

        try:
            return asyncio.get_event_loop().run_until_complete(self.aprocess_request(request))
        except RuntimeError:
            return asyncio.run(self.aprocess_request(request))
