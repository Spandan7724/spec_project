"""Tests for cache module."""
import time
from src.cache import SimpleCache


def test_cache_set_get():
    """Test basic set and get."""
    cache = SimpleCache()
    cache.set("key1", "value1", ttl_seconds=10)
    assert cache.get("key1") == "value1"


def test_cache_expiration():
    """Test TTL expiration."""
    cache = SimpleCache()
    cache.set("key1", "value1", ttl_seconds=1)
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_cache_delete():
    """Test delete."""
    cache = SimpleCache()
    cache.set("key1", "value1")
    cache.delete("key1")
    assert cache.get("key1") is None


def test_cache_clear():
    """Test clear all."""
    cache = SimpleCache()
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_cleanup_expired():
    """Test cleanup of expired entries."""
    cache = SimpleCache()
    cache.set("key1", "value1", ttl_seconds=10)
    cache.set("key2", "value2", ttl_seconds=1)
    time.sleep(1.1)
    
    count = cache.cleanup_expired()
    assert count == 1
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None

