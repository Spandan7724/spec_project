"""System health check endpoint."""
from typing import Dict, Any
from datetime import datetime
import asyncio
from src.database.connection import get_engine
from src.cache import cache
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus:
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        engine = get_engine()
        # Simple query to check connection
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Database connection OK"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Database error: {str(e)}"
        }


async def check_cache() -> Dict[str, Any]:
    """Check cache functionality."""
    try:
        test_key = "_health_check_test"
        test_value = "test"
        
        cache.set(test_key, test_value, ttl_seconds=5)
        retrieved = cache.get(test_key)
        cache.delete(test_key)
        
        if retrieved == test_value:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Cache working correctly"
            }
        else:
            return {
                "status": HealthStatus.DEGRADED,
                "message": "Cache read/write issue"
            }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Cache error: {str(e)}"
        }


async def check_config() -> Dict[str, Any]:
    """Check configuration loading."""
    try:
        from src.config import load_config
        config = load_config()
        required_fields = ["app_name", "database_path", "cache_ttl"]
        
        for field in required_fields:
            if not hasattr(config, field):
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"Missing config field: {field}"
                }
        
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Configuration loaded"
        }
    except Exception as e:
        logger.error(f"Config health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY,
            "message": f"Config error: {str(e)}"
        }


async def get_health_status() -> Dict[str, Any]:
    """
    Get overall system health status.
    
    Returns:
        Dict containing overall status and component statuses
    """
    checks = await asyncio.gather(
        check_database(),
        check_cache(),
        check_config(),
        return_exceptions=True
    )
    
    database_health, cache_health, config_health = checks
    
    # Determine overall status
    statuses = [
        database_health.get("status") if isinstance(database_health, dict) else HealthStatus.UNHEALTHY,
        cache_health.get("status") if isinstance(cache_health, dict) else HealthStatus.UNHEALTHY,
        config_health.get("status") if isinstance(config_health, dict) else HealthStatus.UNHEALTHY,
    ]
    
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall_status = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall_status = HealthStatus.UNHEALTHY
    else:
        overall_status = HealthStatus.DEGRADED
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": database_health if isinstance(database_health, dict) else {"status": HealthStatus.UNHEALTHY, "message": str(database_health)},
            "cache": cache_health if isinstance(cache_health, dict) else {"status": HealthStatus.UNHEALTHY, "message": str(cache_health)},
            "config": config_health if isinstance(config_health, dict) else {"status": HealthStatus.UNHEALTHY, "message": str(config_health)},
        }
    }
