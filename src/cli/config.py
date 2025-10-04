"""
Configuration management for the CLI
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class CLIConfig:
    """Configuration settings for the CLI"""

    # User preferences
    default_base_currency: str = "USD"
    default_quote_currency: str = "EUR"
    default_amount: float = 1000.0
    default_risk_tolerance: str = "moderate"
    default_timeframe_days: int = 7

    # Display settings
    show_welcome_message: bool = True
    show_processing_time: bool = True
    show_correlation_id: bool = False
    export_format: str = "json"  # json, csv, txt

    # Session settings
    auto_save_sessions: bool = True
    max_session_history: int = 100
    session_timeout_hours: int = 24

    # API settings
    request_timeout_seconds: int = 30
    max_retries: int = 3
    debug_mode: bool = False

    # Advanced settings
    cache_enabled: bool = True
    cache_ttl_hours: int = 1
    parallel_requests: bool = True


class ConfigManager:
    """Manages CLI configuration"""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or (Path.home() / ".currency_assistant_config.json")
        self.config = CLIConfig()
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)

                # Update config with loaded values
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

        except Exception as e:
            # If config loading fails, use defaults
            print(f"Warning: Could not load config file, using defaults: {e}")

    def _save_config(self) -> None:
        """Save configuration to file"""

        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)

        except Exception as e:
            print(f"Error: Could not save config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""

        return getattr(self.config, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""

        if hasattr(self.config, key):
            # Type validation and conversion
            field_type = type(getattr(self.config, key))

            try:
                if field_type == bool:
                    value = str(value).lower() in ['true', '1', 'yes', 'on']
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif field_type == str:
                    value = str(value)

                setattr(self.config, key, value)
                self._save_config()

            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {key}: expected {field_type.__name__}, got {value}")

        else:
            raise ValueError(f"Unknown configuration key: {key}")

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""

        return asdict(self.config)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""

        self.config = CLIConfig()
        self._save_config()

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""

        issues = []

        # Validate currencies
        if len(self.config.default_base_currency) != 3 or not self.config.default_base_currency.isalpha():
            issues.append("default_base_currency must be a 3-letter currency code")

        if len(self.config.default_quote_currency) != 3 or not self.config.default_quote_currency.isalpha():
            issues.append("default_quote_currency must be a 3-letter currency code")

        if self.config.default_base_currency == self.config.default_quote_currency:
            issues.append("default_base_currency and default_quote_currency must be different")

        # Validate numeric values
        if self.config.default_amount <= 0:
            issues.append("default_amount must be positive")

        if self.config.default_timeframe_days < 1:
            issues.append("default_timeframe_days must be at least 1")

        if self.config.max_session_history < 1:
            issues.append("max_session_history must be at least 1")

        if self.config.session_timeout_hours < 1:
            issues.append("session_timeout_hours must be at least 1")

        # Validate risk tolerance
        if self.config.default_risk_tolerance not in ["low", "moderate", "high"]:
            issues.append("default_risk_tolerance must be one of: low, moderate, high")

        # Validate export format
        if self.config.export_format not in ["json", "csv", "txt"]:
            issues.append("export_format must be one of: json, csv, txt")

        return issues

    def get_session_config(self) -> Dict[str, Any]:
        """Get configuration relevant for sessions"""

        return {
            "auto_save": self.config.auto_save_sessions,
            "max_history": self.config.max_session_history,
            "timeout_hours": self.config.session_timeout_hours,
        }

    def get_display_config(self) -> Dict[str, Any]:
        """Get configuration relevant for display"""

        return {
            "show_welcome": self.config.show_welcome_message,
            "show_time": self.config.show_processing_time,
            "show_id": self.config.show_correlation_id,
            "export_format": self.config.export_format,
        }

    def get_api_config(self) -> Dict[str, Any]:
        """Get configuration relevant for API calls"""

        return {
            "timeout": self.config.request_timeout_seconds,
            "max_retries": self.config.max_retries,
            "debug": self.config.debug_mode,
            "cache_enabled": self.config.cache_enabled,
            "cache_ttl_hours": self.config.cache_ttl_hours,
            "parallel": self.config.parallel_requests,
        }

    def set_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Set multiple configuration values from a dictionary"""

        for key, value in config_dict.items():
            try:
                self.set(key, value)
            except ValueError as e:
                print(f"Warning: Could not set {key}: {e}")

    def export_config(self, export_path: Path) -> None:
        """Export configuration to a file"""

        try:
            with open(export_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Could not export configuration: {e}")

    def import_config(self, import_path: Path) -> None:
        """Import configuration from a file"""

        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)

            self.set_from_dict(config_data)

            # Validate after import
            issues = self.validate_config()
            if issues:
                print("Warning: Imported configuration has issues:")
                for issue in issues:
                    print(f"  - {issue}")

        except Exception as e:
            raise RuntimeError(f"Could not import configuration: {e}")

    def get_effective_defaults(self) -> Dict[str, Any]:
        """Get effective default values for request processing"""

        return {
            "currency_pair": f"{self.config.default_base_currency}/{self.config.default_quote_currency}",
            "amount": self.config.default_amount,
            "risk_tolerance": self.config.default_risk_tolerance,
            "timeframe_days": self.config.default_timeframe_days,
        }

    def update_from_environment(self) -> None:
        """Update configuration from environment variables"""

        env_mappings = {
            "CURRENCY_ASSISTANT_BASE_CURRENCY": "default_base_currency",
            "CURRENCY_ASSISTANT_QUOTE_CURRENCY": "default_quote_currency",
            "CURRENCY_ASSISTANT_DEFAULT_AMOUNT": "default_amount",
            "CURRENCY_ASSISTANT_RISK_TOLERANCE": "default_risk_tolerance",
            "CURRENCY_ASSISTANT_TIMEFRAME_DAYS": "default_timeframe_days",
            "CURRENCY_ASSISTANT_DEBUG": "debug_mode",
            "CURRENCY_ASSISTANT_TIMEOUT": "request_timeout_seconds",
            "CURRENCY_ASSISTANT_CACHE_ENABLED": "cache_enabled",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    self.set(config_key, value)
                except ValueError as e:
                    print(f"Warning: Invalid environment variable {env_var}: {e}")