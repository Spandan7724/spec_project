"""Configuration management for Currency Assistant."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from src.utils.errors import ConfigurationError
from src.utils.logging import setup_logging
import logging

logger = logging.getLogger(__name__)


class Config:
    """Application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from YAML and environment."""
        # Load environment variables from .env
        load_dotenv()
        
        # Load YAML config
        if not self.config_path.exists():
            raise ConfigurationError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        if not self._config:
            raise ConfigurationError(f"Empty configuration file: {self.config_path}")
        
        # Validate required sections
        self._validate()
        
        # Setup logging
        log_config = self._config.get('logging', {})
        setup_logging(
            level=os.getenv('LOG_LEVEL', log_config.get('level', 'INFO')),
            log_file=log_config.get('file'),
            format_type=log_config.get('format', 'json'),
            enabled=log_config.get('enabled', True)
        )
        
        logger.info("Configuration loaded successfully")
    
    def _validate(self) -> None:
        """Validate required configuration sections."""
        required_sections = ['llm', 'app']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required config section: {section}")
        
        # Validate LLM config
        if 'default_provider' not in self._config['llm']:
            raise ConfigurationError("Missing llm.default_provider in config")
        
        if 'providers' not in self._config['llm']:
            raise ConfigurationError("Missing llm.providers in config")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., "llm.default_provider")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return os.getenv(key, default)
    
    def require_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if value is None:
            raise ConfigurationError(f"Required environment variable not set: {key}")
        return value
    
    @property
    def app_name(self) -> str:
        """Get application name."""
        return self.get('app.name', 'Currency Assistant')
    
    @property
    def app_version(self) -> str:
        """Get application version."""
        return self.get('app.version', '0.1.0')
    
    @property
    def debug(self) -> bool:
        """Get debug mode."""
        return self.get('app.debug', False)
    
    @property
    def database_path(self) -> str:
        """Get database path."""
        return self.get('database.path', 'data/currency_assistant.db')
    
    @property
    def cache_ttl(self) -> int:
        """Get default cache TTL."""
        return self.get('cache.default_ttl', 300)


# Global config instance
_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load and return global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def get_config() -> Config:
    """Get global configuration instance."""
    if _config is None:
        raise ConfigurationError("Configuration not loaded. Call load_config() first.")
    return _config

