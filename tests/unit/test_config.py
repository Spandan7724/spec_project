"""Tests for configuration module."""
import pytest
from src.config import Config
from src.utils.errors import ConfigurationError


def test_config_load(temp_config_file):
    """Test basic config loading."""
    config = Config(temp_config_file)
    assert config.app_name == 'Test App'
    assert config.app_version == '0.1.0'
    assert config.debug is True


def test_config_get_nested(temp_config_file):
    """Test getting nested config values."""
    config = Config(temp_config_file)
    assert config.get('llm.default_provider') == 'copilot'
    assert config.get('database.type') == 'sqlite'


def test_config_get_default(temp_config_file):
    """Test default values."""
    config = Config(temp_config_file)
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_missing_file():
    """Test error on missing config file."""
    with pytest.raises(ConfigurationError):
        Config('nonexistent.yaml')

