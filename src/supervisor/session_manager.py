from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

from .models import ConversationSession


class SessionManager:
    """Utility for session lifecycle and cleanup."""

    def __init__(self, timeout_minutes: int = 30) -> None:
        self.timeout = timedelta(minutes=timeout_minutes)

    def cleanup_expired_sessions(self, sessions: Dict[str, ConversationSession]) -> int:
        """Remove sessions older than timeout. Returns number removed."""
        now = datetime.now()
        to_delete = [sid for sid, s in sessions.items() if now - s.last_updated > self.timeout]
        for sid in to_delete:
            del sessions[sid]
        return len(to_delete)

    def get_session_stats(self, sessions: Dict[str, ConversationSession]) -> Dict[str, Any]:
        """Return simple stats about current sessions."""
        if not sessions:
            return {"count": 0, "oldest_minutes": 0, "newest_minutes": 0}
        now = datetime.now()
        ages = [(now - s.last_updated).total_seconds() / 60.0 for s in sessions.values()]
        return {
            "count": len(sessions),
            "oldest_minutes": max(ages),
            "newest_minutes": min(ages),
        }

