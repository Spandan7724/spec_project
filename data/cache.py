"""
Simple in-memory cache implementation for MVP.

Provides fast access to current FX rates with TTL expiration.
In production, this would be replaced with Redis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: datetime
    created_at: datetime
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.utcnow() > self.expires_at


class SimpleCache:
    """
    Simple in-memory cache with TTL support.
    
    This is a basic implementation for the MVP. In production,
    we would use Redis or another distributed cache.
    """
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expired_cleanups": 0
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task to clean up expired entries."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
    
    async def _cleanup_expired_entries(self):
        """Background task to periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.expires_at < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1
        
        if expired_keys:
            self._stats["expired_cleanups"] += 1
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Returns None if key doesn't exist or has expired.
        """
        entry = self._cache.get(key)
        
        if entry is None:
            self._stats["misses"] += 1
            return None
        
        if entry.is_expired():
            # Remove expired entry
            del self._cache[key]
            self._stats["misses"] += 1
            self._stats["evictions"] += 1
            return None
        
        self._stats["hits"] += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 30) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to store
            ttl_seconds: Time to live in seconds
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        self._cache[key] = CacheEntry(
            value=value,
            expires_at=expires_at,
            created_at=now
        )
        
        self._stats["sets"] += 1
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Returns True if key existed, False otherwise.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        
        if entry.is_expired():
            del self._cache[key]
            self._stats["evictions"] += 1
            return False
        
        return True
    
    async def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "sets": self._stats["sets"],
            "evictions": self._stats["evictions"],
            "expired_cleanups": self._stats["expired_cleanups"]
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._cache.clear()


class CacheManager:
    """
    Cache manager for FX rates and related data.
    
    Provides typed interfaces for different types of cached data
    with appropriate TTL values.
    """
    
    def __init__(self):
        self.cache = SimpleCache()
        
        # TTL values for different data types
        self.current_rate_ttl = 30      # 30 seconds for current rates
        self.provider_rate_ttl = 14400  # 4 hours for provider rates  
        self.prediction_ttl = 3600      # 1 hour for predictions
    
    # Current FX rates
    async def get_current_rate(self, currency_pair: str) -> Optional[Any]:
        """Get current rate from cache."""
        key = f"rate:{currency_pair}"
        return await self.cache.get(key)
    
    async def set_current_rate(self, currency_pair: str, rate_data: Any) -> None:
        """Cache current rate."""
        key = f"rate:{currency_pair}"
        await self.cache.set(key, rate_data, self.current_rate_ttl)
    
    # Provider rates  
    async def get_provider_rate(self, provider: str, currency_pair: str) -> Optional[Any]:
        """Get provider-specific rate from cache."""
        key = f"provider:{provider}:{currency_pair}"
        return await self.cache.get(key)
    
    async def set_provider_rate(self, provider: str, currency_pair: str, rate_data: Any) -> None:
        """Cache provider-specific rate."""
        key = f"provider:{provider}:{currency_pair}"
        await self.cache.set(key, rate_data, self.provider_rate_ttl)
    
    # Predictions
    async def get_prediction(self, currency_pair: str) -> Optional[Any]:
        """Get ML prediction from cache.""" 
        key = f"prediction:{currency_pair}"
        return await self.cache.get(key)
    
    async def set_prediction(self, currency_pair: str, prediction_data: Any) -> None:
        """Cache ML prediction."""
        key = f"prediction:{currency_pair}"
        await self.cache.set(key, prediction_data, self.prediction_ttl)
    
    # Utility methods
    async def clear_all(self) -> None:
        """Clear all cached data."""
        await self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    async def cleanup(self) -> None:
        """Clean up cache resources."""
        await self.cache.cleanup()