"""
Configuration Management for LLM Providers and Agents.

Handles loading, validation, and management of configuration files
with environment variable overrides and runtime updates.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_base: str
    models: List[str]
    default_model: str
    auth: Dict[str, Any]
    features: Dict[str, bool]
    rate_limits: Dict[str, int]


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    system_prompt: str = ""


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    parallel_execution: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    fallback_provider: str = "openai"
    enable_caching: bool = True
    cache_ttl_minutes: int = 5


@dataclass
class LLMConfig:
    """Complete LLM system configuration."""
    default_provider: str = "copilot"
    default_model: str = "gpt-4o-2024-11-20"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    fallback_order: List[str] = field(default_factory=lambda: ["copilot", "openai", "anthropic"])
    
    # Runtime settings
    performance: Dict[str, Any] = field(default_factory=dict)
    development: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Loads and manages LLM configuration from YAML files and environment variables.
    
    Features:
    - YAML configuration file loading
    - Environment variable overrides
    - Configuration validation
    - Runtime configuration updates
    - Default fallbacks
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            # Default to config file in the same directory
            config_dir = Path(__file__).parent
            config_path = config_dir / "llm_config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[LLMConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
        
        logger.debug(f"Initialized ConfigLoader with path: {self.config_path}")
    
    def load_config(self) -> LLMConfig:
        """
        Load configuration from YAML file with environment overrides.
        
        Returns:
            Complete LLM configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
            ValueError: If configuration validation fails
        """
        try:
            # Load YAML configuration
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._raw_config = yaml.safe_load(file)
            
            logger.info(f"Loaded configuration from: {self.config_path}")
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Parse and validate configuration
            self._config = self._parse_config(self._raw_config)
            
            # Validate configuration
            self._validate_config(self._config)
            
            logger.info(f"Successfully loaded configuration with default provider: {self._config.default_provider}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise e
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        if not self._raw_config or 'env_overrides' not in self._raw_config:
            return
        
        env_overrides = self._raw_config['env_overrides']
        overrides_applied = []
        
        # Apply each defined override
        for config_key, env_var in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if config_key == 'provider':
                    self._raw_config['default_provider'] = env_value
                elif config_key == 'model':
                    self._raw_config['default_model'] = env_value
                elif config_key == 'temperature':
                    # Apply temperature override to all agents
                    temp_value = float(env_value)
                    for agent_name in self._raw_config.get('agents', {}):
                        self._raw_config['agents'][agent_name]['temperature'] = temp_value
                elif config_key == 'max_tokens':
                    # Apply max_tokens override to all agents
                    tokens_value = int(env_value)
                    for agent_name in self._raw_config.get('agents', {}):
                        self._raw_config['agents'][agent_name]['max_tokens'] = tokens_value
                elif config_key == 'log_level':
                    self._raw_config['performance']['log_level'] = env_value
                
                overrides_applied.append(f"{config_key}={env_value}")
        
        if overrides_applied:
            logger.info(f"Applied environment overrides: {', '.join(overrides_applied)}")
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> LLMConfig:
        """Parse raw configuration dictionary into structured config objects."""
        
        # Parse provider configurations
        providers = {}
        for provider_name, provider_data in raw_config.get('providers', {}).items():
            providers[provider_name] = ProviderConfig(
                name=provider_name,
                api_base=provider_data.get('api_base', ''),
                models=provider_data.get('models', []),
                default_model=provider_data.get('default_model', ''),
                auth=provider_data.get('auth', {}),
                features=provider_data.get('features', {}),
                rate_limits=provider_data.get('rate_limits', {})
            )
        
        # Parse agent configurations
        agents = {}
        for agent_name, agent_data in raw_config.get('agents', {}).items():
            agents[agent_name] = AgentConfig(
                name=agent_name,
                model_override=agent_data.get('model_override'),
                provider_override=agent_data.get('provider_override'),
                temperature=agent_data.get('temperature', 0.3),
                max_tokens=agent_data.get('max_tokens', 2000),
                system_prompt=agent_data.get('system_prompt', '')
            )
        
        # Parse workflow configuration
        workflow_data = raw_config.get('workflow', {})
        workflow = WorkflowConfig(
            parallel_execution=workflow_data.get('parallel_execution', True),
            timeout_seconds=workflow_data.get('timeout_seconds', 30),
            retry_attempts=workflow_data.get('retry_attempts', 3),
            fallback_provider=workflow_data.get('fallback_provider', 'openai'),
            enable_caching=workflow_data.get('enable_caching', True),
            cache_ttl_minutes=workflow_data.get('cache_ttl_minutes', 5)
        )
        
        # Create main configuration
        config = LLMConfig(
            default_provider=raw_config.get('default_provider', 'copilot'),
            default_model=raw_config.get('default_model', 'gpt-4o-2024-11-20'),
            providers=providers,
            agents=agents,
            workflow=workflow,
            fallback_order=raw_config.get('fallback_order', ['copilot', 'openai', 'anthropic']),
            performance=raw_config.get('performance', {}),
            development=raw_config.get('development', {})
        )
        
        return config
    
    def _validate_config(self, config: LLMConfig) -> None:
        """
        Validate configuration for consistency and completeness.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check if default provider exists
        if config.default_provider not in config.providers:
            raise ValueError(f"Default provider '{config.default_provider}' not found in providers")
        
        # Check if default model is available for default provider
        default_provider = config.providers[config.default_provider]
        if config.default_model not in default_provider.models:
            raise ValueError(
                f"Default model '{config.default_model}' not available for provider '{config.default_provider}'"
            )
        
        # Validate fallback order
        for provider_name in config.fallback_order:
            if provider_name not in config.providers:
                logger.warning(f"Fallback provider '{provider_name}' not found in providers")
        
        # Validate agent configurations
        for agent_name, agent_config in config.agents.items():
            if agent_config.provider_override and agent_config.provider_override not in config.providers:
                logger.warning(f"Agent '{agent_name}' specifies unknown provider '{agent_config.provider_override}'")
        
        # Check required environment variables for primary provider
        primary_provider = config.providers[config.default_provider]
        if primary_provider.auth.get('required', False):
            token_env = primary_provider.auth.get('token_env')
            if token_env and not os.getenv(token_env):
                logger.warning(
                    f"Required environment variable '{token_env}' for provider '{config.default_provider}' not set"
                )
        
        logger.debug("Configuration validation completed successfully")
    
    def get_config(self) -> LLMConfig:
        """
        Get the loaded configuration.
        
        Returns:
            LLM configuration
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider configuration
            
        Raises:
            KeyError: If provider not found
        """
        config = self.get_config()
        if provider_name not in config.providers:
            raise KeyError(f"Provider '{provider_name}' not found")
        return config.providers[provider_name]
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration
            
        Raises:
            KeyError: If agent not found
        """
        config = self.get_config()
        if agent_name not in config.agents:
            raise KeyError(f"Agent '{agent_name}' not found")
        return config.agents[agent_name]
    
    def update_provider_config(self, provider_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific provider at runtime.
        
        Args:
            provider_name: Name of the provider to update
            updates: Dictionary of updates to apply
        """
        config = self.get_config()
        if provider_name not in config.providers:
            raise KeyError(f"Provider '{provider_name}' not found")
        
        provider_config = config.providers[provider_name]
        for key, value in updates.items():
            if hasattr(provider_config, key):
                setattr(provider_config, key, value)
                logger.info(f"Updated provider '{provider_name}' {key} to {value}")
            else:
                logger.warning(f"Unknown provider config key: {key}")
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific agent at runtime.
        
        Args:
            agent_name: Name of the agent to update
            updates: Dictionary of updates to apply
        """
        config = self.get_config()
        if agent_name not in config.agents:
            raise KeyError(f"Agent '{agent_name}' not found")
        
        agent_config = config.agents[agent_name]
        for key, value in updates.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)
                logger.info(f"Updated agent '{agent_name}' {key} to {value}")
            else:
                logger.warning(f"Unknown agent config key: {key}")
    
    def reload_config(self) -> LLMConfig:
        """
        Reload configuration from file.
        
        Returns:
            Reloaded configuration
        """
        logger.info("Reloading configuration...")
        return self.load_config()
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List of provider names
        """
        config = self.get_config()
        return list(config.providers.keys())
    
    def get_available_models(self, provider_name: str) -> List[str]:
        """
        Get list of available models for a provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            List of model names
            
        Raises:
            KeyError: If provider not found
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.models
    
    def is_development_mode(self) -> bool:
        """Check if development mode is enabled."""
        config = self.get_config()
        return config.development.get('test_mode', False)
    
    def __str__(self) -> str:
        if self._config:
            return f"ConfigLoader(provider={self._config.default_provider}, model={self._config.default_model})"
        return "ConfigLoader(not loaded)"


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Global ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def load_config(config_path: Optional[str] = None) -> LLMConfig:
    """
    Load LLM configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded LLM configuration
    """
    loader = get_config_loader(config_path)
    return loader.load_config()