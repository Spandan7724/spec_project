"""Pytest configuration and fixtures."""
import pytest
from pathlib import Path
import tempfile
import yaml
from src.cache import cache


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        'app': {
            'name': 'Test App',
            'version': '0.1.0',
            'debug': True
        },
        'llm': {
            'default_provider': 'copilot',
            'providers': {
                'copilot': {
                    'model': 'gpt-4o',
                    'enabled': True
                }
            }
        },
        'database': {
            'type': 'sqlite',
            'path': ':memory:'
        },
        'cache': {
            'type': 'memory',
            'default_ttl': 300
        },
        'logging': {
            'level': 'DEBUG',
            'format': 'text'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink()


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the global cache before each test."""
    cache.clear()
    yield
    cache.clear()

