"""Utility decorators for error handling and resilience."""
import asyncio
import functools
import time
from typing import Callable, Type, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
    
    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(httpx.HTTPError,))
        async def fetch_data():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            extra={"error": str(e), "attempts": attempt}
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{max_attempts}), retrying in {current_delay}s",
                        extra={"error": str(e), "delay": current_delay}
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            extra={"error": str(e), "attempts": attempt}
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{max_attempts}), retrying in {current_delay}s",
                        extra={"error": str(e), "delay": current_delay}
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    
    Example:
        @timeout(10.0)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    f"Function {func.__name__} timed out after {seconds}s",
                    extra={"timeout": seconds}
                )
                raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
        
        return wrapper
    
    return decorator


def log_execution(log_args: bool = True, log_result: bool = False):
    """
    Decorator to log function execution with timing.
    
    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    
    Example:
        @log_execution(log_args=True, log_result=True)
        async def process_data(data):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            extra = {"function": func_name}
            if log_args:
                extra["function_args"] = str(args)[:100]  # Truncate long args
                extra["function_kwargs"] = str(kwargs)[:100]
            
            logger.info(f"Starting {func_name}", extra=extra)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                log_extra = {"function": func_name, "execution_time_ms": round(execution_time, 2)}
                if log_result:
                    log_extra["result"] = str(result)[:100]
                
                logger.info(f"Completed {func_name}", extra=log_extra)
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed {func_name}",
                    extra={"function": func_name, "execution_time_ms": round(execution_time, 2), "error": str(e)}
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            extra = {"function": func_name}
            if log_args:
                extra["function_args"] = str(args)[:100]
                extra["function_kwargs"] = str(kwargs)[:100]
            
            logger.info(f"Starting {func_name}", extra=extra)
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                log_extra = {"function": func_name, "execution_time_ms": round(execution_time, 2)}
                if log_result:
                    log_extra["result"] = str(result)[:100]
                
                logger.info(f"Completed {func_name}", extra=log_extra)
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"Failed {func_name}",
                    extra={"function": func_name, "execution_time_ms": round(execution_time, 2), "error": str(e)}
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

