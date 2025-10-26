"""Tests for health check functionality."""
import pytest
from src.health import (
    check_database,
    check_cache,
    check_config,
    get_health_status,
    HealthStatus
)


@pytest.mark.asyncio
async def test_check_database():
    """Test database health check."""
    result = await check_database()
    assert "status" in result
    assert "message" in result
    assert result["status"] in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]


@pytest.mark.asyncio
async def test_check_cache():
    """Test cache health check."""
    result = await check_cache()
    assert "status" in result
    assert "message" in result
    assert result["status"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_check_config():
    """Test config health check."""
    result = await check_config()
    assert "status" in result
    assert "message" in result
    assert result["status"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_get_health_status():
    """Test overall health status."""
    status = await get_health_status()
    
    assert "status" in status
    assert "timestamp" in status
    assert "components" in status
    
    assert "database" in status["components"]
    assert "cache" in status["components"]
    assert "config" in status["components"]
    
    # Overall status should be one of the valid statuses
    assert status["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
