from __future__ import annotations

from typing import Optional

from src.supervisor.conversation_manager import ConversationManager
from src.supervisor.agent_orchestrator import AgentOrchestrator


_cm_singleton: Optional[ConversationManager] = None
_orc_singleton: Optional[AgentOrchestrator] = None


def get_conversation_manager() -> ConversationManager:
    global _cm_singleton
    if _cm_singleton is None:
        _cm_singleton = ConversationManager()
    return _cm_singleton


def get_orchestrator() -> AgentOrchestrator:
    global _orc_singleton
    if _orc_singleton is None:
        _orc_singleton = AgentOrchestrator()
    return _orc_singleton

