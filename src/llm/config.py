"""
Configuration management for LLM providers
"""

import os
import yaml
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .types import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Main configuration class for LLM system"""
    default_provider: str
    providers: Dict[str, ProviderConfig]
    failover_enabled: bool = True
    failover_order: Optional[List[str]] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'LLMConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            llm_config = data.get('llm', {})
            
            # Parse provider configurations
            providers = {}
            provider_configs = llm_config.get('providers', {})
            
            for provider_name, provider_data in provider_configs.items():
                # Map provider names to their environment variable names
                # Support copilot variants (copilot_mini, copilot_claude, etc.)
                if provider_name.startswith('copilot'):
                    api_key_env = 'COPILOT_ACCESS_TOKEN'
                elif provider_name == 'openai':
                    api_key_env = 'OPENAI_API_KEY'
                elif provider_name == 'claude':
                    api_key_env = 'ANTHROPIC_API_KEY'
                else:
                    api_key_env = None
                
                providers[provider_name] = ProviderConfig(
                    name=provider_name,
                    model=provider_data.get('model', 'gpt-4'),
                    enabled=provider_data.get('enabled', True),
                    api_key_env=api_key_env,
                    kwargs=provider_data.get('kwargs', {})
                )
            
            # Get failover configuration
            failover_config = llm_config.get('failover', {})
            failover_enabled = failover_config.get('enabled', True)
            failover_order = failover_config.get('order', ['copilot', 'openai', 'claude'])
            
            return cls(
                default_provider=llm_config.get('default_provider', 'copilot'),
                providers=providers,
                failover_enabled=failover_enabled,
                failover_order=failover_order
            )
            
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return cls.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls.get_default_config()
    
    @classmethod
    def get_default_config(cls) -> 'LLMConfig':
        """Get default configuration"""
        providers = {
            'copilot': ProviderConfig(
                name='copilot',
                model='gpt-4o',
                enabled=True,
                api_key_env='COPILOT_ACCESS_TOKEN'
            ),
            'openai': ProviderConfig(
                name='openai', 
                model='gpt-4',
                enabled=True,
                api_key_env='OPENAI_API_KEY'
            ),
            'claude': ProviderConfig(
                name='claude',
                model='claude-3-5-sonnet-20241022',
                enabled=True,
                api_key_env='ANTHROPIC_API_KEY'
            )
        }
        
        return cls(
            default_provider='copilot',
            providers=providers,
            failover_enabled=True,
            failover_order=['copilot', 'openai', 'claude']
        )
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names"""
        return [name for name, config in self.providers.items() if config.enabled]
    
    def validate_environment_variables(self) -> Dict[str, bool]:
        """Check which providers have their required environment variables set"""
        validation_results = {}
        
        for provider_name, config in self.providers.items():
            if config.enabled and config.api_key_env:
                env_var_exists = os.getenv(config.api_key_env) is not None
                validation_results[provider_name] = env_var_exists
                
                if not env_var_exists:
                    logger.warning(f"Environment variable {config.api_key_env} not set for provider {provider_name}")
        
        return validation_results
    
    def create_sample_config(self, output_path: str):
        """Create a sample configuration file"""
        sample_config = {
            'llm': {
                'default_provider': 'copilot',
                'providers': {
                    'copilot': {
                        'model': 'gpt-4o',
                        'enabled': True,
                        'kwargs': {
                            'temperature': 0.7,
                            'max_tokens': 200000
                        }
                    },
                    'openai': {
                        'model': 'gpt-4',
                        'enabled': True,
                        'kwargs': {
                            'temperature': 0.7,
                            'max_tokens': 200000
                        }
                    },
                    'claude': {
                        'model': 'claude-3-5-sonnet-20241022',
                        'enabled': True,
                        'kwargs': {
                            'temperature': 0.7,
                            'max_tokens': 200000
                        }
                    }
                },
                'failover': {
                    'enabled': True,
                    'order': ['copilot', 'openai', 'claude']
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Sample configuration created at {output_path}")


def load_config(config_path: Optional[str] = None) -> LLMConfig:
    """Load LLM configuration from file or use defaults"""
    if config_path is None:
        # Look for config.yaml in current directory or currency_assistant root
        possible_paths = [
            'config.yaml',
            '/home/spandan/projects/spec_project_2/currency_assistant/config.yaml',
            '/home/spandan/projects/spec_project_2/currency_assistant/src/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        return LLMConfig.from_yaml(config_path)
    else:
        logger.info("No config file found, using default configuration")
        return LLMConfig.get_default_config()