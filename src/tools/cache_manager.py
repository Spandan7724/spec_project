#!/usr/bin/env python3
"""
Intelligent Cache Manager for Web Scraping Tool
Handles TTL-based caching with content-type awareness
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from hashlib import md5

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    url: str
    content: str
    extracted_data: Dict[str, Any]
    timestamp: float
    content_type: str
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.timestamp > self.ttl_seconds
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp

class CacheManager:
    """
    Intelligent cache manager for scraped content
    Different TTL based on content type
    """
    
    # Content type TTL mapping (in seconds)
    TTL_CONFIG = {
        'exchange_rates': 300,      # 5 minutes
        'economic_news': 1800,      # 30 minutes  
        'provider_policies': 86400, # 24 hours
        'regulatory_changes': 21600, # 6 hours
        'company_info': 604800,     # 7 days
        'general': 3600,            # 1 hour default
    }
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "scraping_cache.json"
        self._cache: Dict[str, CacheEntry] = {}
        self._load_cache()
    
    def _url_hash(self, url: str) -> str:
        """Generate hash key for URL"""
        return md5(url.encode()).hexdigest()[:12]
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for key, data in cache_data.items():
                    self._cache[key] = CacheEntry(**data)
            except (json.JSONDecodeError, TypeError):
                # Corrupted cache, start fresh
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_data = {}
        for key, entry in self._cache.items():
            if not entry.is_expired():  # Only save non-expired entries
                cache_data[key] = asdict(entry)
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def get_ttl_for_content_type(self, content_type: str) -> int:
        """Get TTL seconds for content type"""
        return self.TTL_CONFIG.get(content_type, self.TTL_CONFIG['general'])
    
    def should_bypass_cache_for_query(self, query_context: str) -> bool:
        """
        Determine if cache should be bypassed based on query context
        
        Args:
            query_context: User query or conversation context
            
        Returns:
            True if cache should be bypassed for fresh data
        """
        bypass_keywords = [
            'today', 'latest', 'current', 'now', 'just announced',
            'breaking', 'recent', 'this week', 'updated', 'new'
        ]
        
        query_lower = query_context.lower()
        return any(keyword in query_lower for keyword in bypass_keywords)
    
    def get(self, url: str, content_type: str = 'general', 
            query_context: str = "") -> Optional[CacheEntry]:
        """
        Get cached entry if valid and fresh
        
        Args:
            url: URL to check cache for
            content_type: Type of content for TTL determination
            query_context: User query context for bypass decisions
            
        Returns:
            CacheEntry if valid and fresh, None otherwise
        """
        # Check if we should bypass cache based on query
        if self.should_bypass_cache_for_query(query_context):
            return None
        
        key = self._url_hash(url)
        entry = self._cache.get(key)
        
        if entry and not entry.is_expired():
            return entry
        
        return None
    
    def set(self, url: str, content: str, extracted_data: Dict[str, Any],
            content_type: str = 'general') -> None:
        """
        Store content in cache with appropriate TTL
        
        Args:
            url: URL being cached
            content: Scraped content
            extracted_data: Processed/extracted data
            content_type: Content type for TTL determination
        """
        key = self._url_hash(url)
        ttl = self.get_ttl_for_content_type(content_type)
        
        entry = CacheEntry(
            url=url,
            content=content,
            extracted_data=extracted_data,
            timestamp=time.time(),
            content_type=content_type,
            ttl_seconds=ttl
        )
        
        self._cache[key] = entry
        self._save_cache()
    
    def clear_expired(self):
        """Remove expired entries from cache"""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        self._save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        content_type_breakdown = {}
        for entry in self._cache.values():
            if not entry.is_expired():
                content_type_breakdown[entry.content_type] = content_type_breakdown.get(entry.content_type, 0) + 1
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'content_type_breakdown': content_type_breakdown
        }