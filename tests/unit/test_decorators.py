"""Tests for decorator utilities."""
import pytest
import asyncio
from src.utils.decorators import retry, timeout, log_execution


@pytest.mark.asyncio
async def test_retry_success():
    """Test retry decorator with successful execution."""
    call_count = 0
    
    @retry(max_attempts=3, delay=0.1)
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary error")
        return "success"
    
    result = await flaky_function()
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_failure():
    """Test retry decorator with persistent failure."""
    @retry(max_attempts=3, delay=0.1)
    async def always_fails():
        raise ValueError("Persistent error")
    
    with pytest.raises(ValueError, match="Persistent error"):
        await always_fails()


@pytest.mark.asyncio
async def test_timeout_success():
    """Test timeout decorator with fast function."""
    @timeout(1.0)
    async def fast_function():
        await asyncio.sleep(0.1)
        return "done"
    
    result = await fast_function()
    assert result == "done"


@pytest.mark.asyncio
async def test_timeout_failure():
    """Test timeout decorator with slow function."""
    @timeout(0.5)
    async def slow_function():
        await asyncio.sleep(2.0)
        return "done"
    
    with pytest.raises(TimeoutError):
        await slow_function()


@pytest.mark.asyncio
async def test_log_execution():
    """Test log_execution decorator."""
    @log_execution(log_args=True, log_result=True)
    async def logged_function(x, y):
        return x + y
    
    result = await logged_function(2, 3)
    assert result == 5


def test_retry_sync():
    """Test retry decorator with synchronous function."""
    call_count = 0
    
    @retry(max_attempts=3, delay=0.1)
    def flaky_sync():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary error")
        return "success"
    
    result = flaky_sync()
    assert result == "success"
    assert call_count == 2
