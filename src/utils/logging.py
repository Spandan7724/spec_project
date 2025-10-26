"""Centralized logging configuration."""
import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "json",
    enabled: bool = True
) -> None:
    """Configure application logging."""
    
    if not enabled:
        # Disable all logging completely
        logging.disable(logging.CRITICAL)
        return
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    if format_type == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override existing configuration
    )


class JsonFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        return json.dumps(log_data)


def get_logger(name: str, correlation_id: Optional[str] = None) -> logging.Logger:
    """Get a logger with optional correlation ID."""
    logger = logging.getLogger(name)
    
    if correlation_id:
        # Add correlation ID to all log records
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.correlation_id = correlation_id
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger

