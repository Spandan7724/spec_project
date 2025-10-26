"""Simple in-memory cache with TTL."""
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import threading


class SimpleCache:
    """Thread-safe in-memory cache with TTL."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Store value with expiration."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        with self._lock:
            self._cache[key] = {"value": value, "expires_at": expires_at}
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            item = self._cache[key]
            if datetime.utcnow() < item["expires_at"]:
                return item["value"]
            else:
                del self._cache[key]
                return None
    
    def delete(self, key: str) -> None:
        """Delete key."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        with self._lock:
            now = datetime.utcnow()
            expired = [k for k, v in self._cache.items() if now >= v["expires_at"]]
            for key in expired:
                del self._cache[key]
            return len(expired)


# Global cache instance
cache = SimpleCache()

