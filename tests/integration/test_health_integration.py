"""Integration test for health check system."""
import pytest
from src.health import get_health_status, HealthStatus
from src.config import load_config
from src.database.connection import create_tables


@pytest.mark.asyncio
async def test_health_check_full_system():
    """Test health check with full system initialized."""
    # Initialize system
    load_config()
    create_tables()
    
    # Get health status
    status = await get_health_status()
    
    # System should be healthy
    assert status["status"] == HealthStatus.HEALTHY
    assert status["components"]["database"]["status"] == HealthStatus.HEALTHY
    assert status["components"]["cache"]["status"] == HealthStatus.HEALTHY
    assert status["components"]["config"]["status"] == HealthStatus.HEALTHY
