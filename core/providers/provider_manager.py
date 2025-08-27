"""
Provider Manager for Multi-Agent Currency Assistant.

Manages LLM provider selection, switching, fallback mechanisms,
and health monitoring across Copilot, OpenAI, and Anthropic providers.
"""

import logging
from typing import Dict, Any, List, Optional, Type
from enum import Enum

from .base_provider import BaseLLMProvider, ChatResponse, ProviderCapabilities
from .copilot_provider import CopilotProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Enumeration of available provider types."""
    COPILOT = "copilot"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderManager:
    """
    Manages multiple LLM providers with automatic failover and switching.
    
    Features:
    - Provider registration and selection
    - Automatic fallback when primary provider fails
    - Health monitoring and recovery
    - Configuration-based provider switching
    - Load balancing (future)
    """
    
    def __init__(self):
        """Initialize the provider manager."""
        # Registry of available provider classes
        self._provider_classes: Dict[ProviderType, Type[BaseLLMProvider]] = {
            ProviderType.COPILOT: CopilotProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.ANTHROPIC: AnthropicProvider
        }
        
        # Active provider instances
        self._providers: Dict[ProviderType, BaseLLMProvider] = {}
        
        # Current primary provider
        self._primary_provider: Optional[BaseLLMProvider] = None
        
        # Fallback order
        self._fallback_order: List[ProviderType] = [
            ProviderType.COPILOT,
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC
        ]
        
        # Health status tracking
        self._provider_health: Dict[ProviderType, bool] = {}
        
        logger.debug("Initialized ProviderManager")
    
    def register_provider(self, 
                         provider_type: ProviderType, 
                         provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a new provider class.
        
        Args:
            provider_type: Type identifier for the provider
            provider_class: Provider class to register
        """
        self._provider_classes[provider_type] = provider_class
        logger.debug(f"Registered provider: {provider_type}")
    
    async def initialize_provider(self, 
                                provider_type: ProviderType, 
                                model: str, 
                                **kwargs) -> BaseLLMProvider:
        """
        Initialize a specific provider.
        
        Args:
            provider_type: Type of provider to initialize
            model: Model to use with the provider
            **kwargs: Provider-specific configuration
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If provider type is not supported
            Exception: If provider initialization fails
        """
        if provider_type not in self._provider_classes:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        try:
            provider_class = self._provider_classes[provider_type]
            provider = provider_class(model=model, **kwargs)
            
            # Test the provider health
            is_healthy = await provider.health_check()
            
            self._providers[provider_type] = provider
            self._provider_health[provider_type] = is_healthy
            
            if is_healthy:
                logger.info(f"Successfully initialized {provider_type} provider with model {model}")
            else:
                logger.warning(f"Initialized {provider_type} provider but health check failed")
            
            return provider
            
        except Exception as e:
            logger.error(f"Failed to initialize {provider_type} provider: {e}")
            self._provider_health[provider_type] = False
            raise e
    
    async def set_primary_provider(self, 
                                 provider_type: ProviderType, 
                                 model: str, 
                                 **kwargs) -> None:
        """
        Set the primary provider for the system.
        
        Args:
            provider_type: Type of provider to set as primary
            model: Model to use
            **kwargs: Provider-specific configuration
        """
        try:
            provider = await self.initialize_provider(provider_type, model, **kwargs)
            self._primary_provider = provider
            logger.info(f"Set primary provider: {provider_type} with model {model}")
            
        except Exception as e:
            logger.error(f"Failed to set primary provider {provider_type}: {e}")
            # Try to set a fallback provider
            await self._try_fallback_providers(model, **kwargs)
    
    async def get_provider(self, 
                         provider_type: Optional[ProviderType] = None) -> BaseLLMProvider:
        """
        Get a provider instance.
        
        Args:
            provider_type: Specific provider type to get. If None, returns primary.
            
        Returns:
            Provider instance
            
        Raises:
            Exception: If no provider is available
        """
        if provider_type is None:
            # Return primary provider
            if self._primary_provider is None:
                raise Exception("No primary provider configured")
            
            # Check if primary provider is healthy
            primary_type = ProviderType(self._primary_provider.get_provider_name())
            if not self._provider_health.get(primary_type, False):
                logger.warning("Primary provider is unhealthy, attempting fallback")
                return await self._get_fallback_provider()
            
            return self._primary_provider
        
        else:
            # Return specific provider
            if provider_type not in self._providers:
                raise Exception(f"Provider {provider_type} not initialized")
            
            return self._providers[provider_type]
    
    async def _get_fallback_provider(self) -> BaseLLMProvider:
        """
        Get a fallback provider when primary fails.
        
        Returns:
            Healthy fallback provider
            
        Raises:
            Exception: If no healthy providers are available
        """
        for provider_type in self._fallback_order:
            if provider_type in self._providers:
                provider = self._providers[provider_type]
                
                # Test provider health
                is_healthy = await provider.health_check()
                self._provider_health[provider_type] = is_healthy
                
                if is_healthy:
                    logger.info(f"Using fallback provider: {provider_type}")
                    return provider
        
        raise Exception("No healthy providers available")
    
    async def _try_fallback_providers(self, model: str, **kwargs) -> None:
        """Try to initialize fallback providers when primary fails."""
        for provider_type in self._fallback_order:
            try:
                provider = await self.initialize_provider(provider_type, model, **kwargs)
                if self._provider_health.get(provider_type, False):
                    self._primary_provider = provider
                    logger.info(f"Set fallback provider as primary: {provider_type}")
                    return
            except Exception as e:
                logger.warning(f"Fallback provider {provider_type} failed: {e}")
                continue
        
        logger.error("All fallback providers failed")
    
    async def chat(self, 
                   messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None,
                   provider_type: Optional[ProviderType] = None,
                   **kwargs) -> ChatResponse:
        """
        Send a chat request using the specified or primary provider.
        
        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            provider_type: Specific provider to use (None for primary)
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse from the provider
        """
        try:
            provider = await self.get_provider(provider_type)
            return await provider.chat(messages, tools, **kwargs)
            
        except Exception as e:
            logger.error(f"Chat request failed with provider {provider_type}: {e}")
            
            # If this was the primary provider, try fallback
            if provider_type is None:
                try:
                    fallback_provider = await self._get_fallback_provider()
                    return await fallback_provider.chat(messages, tools, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback chat request also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e
    
    async def stream_chat(self, 
                         messages: List[Dict[str, Any]], 
                         tools: Optional[List[Dict[str, Any]]] = None,
                         provider_type: Optional[ProviderType] = None,
                         **kwargs):
        """
        Stream a chat request using the specified or primary provider.
        
        Args:
            messages: Chat messages
            tools: Optional tools for function calling
            provider_type: Specific provider to use (None for primary)
            **kwargs: Additional parameters
            
        Yields:
            Chat response chunks
        """
        try:
            provider = await self.get_provider(provider_type)
            async for chunk in provider.stream_chat(messages, tools, **kwargs):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming chat request failed with provider {provider_type}: {e}")
            
            # If this was the primary provider, try fallback
            if provider_type is None:
                try:
                    fallback_provider = await self._get_fallback_provider()
                    async for chunk in fallback_provider.stream_chat(messages, tools, **kwargs):
                        yield chunk
                except Exception as fallback_error:
                    logger.error(f"Fallback streaming chat request also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e
    
    def get_provider_capabilities(self, 
                                provider_type: Optional[ProviderType] = None) -> ProviderCapabilities:
        """
        Get capabilities of a provider.
        
        Args:
            provider_type: Provider type to check (None for primary)
            
        Returns:
            Provider capabilities
        """
        if provider_type is None:
            if self._primary_provider is None:
                raise Exception("No primary provider configured")
            return self._primary_provider.get_supported_features()
        else:
            if provider_type not in self._providers:
                raise Exception(f"Provider {provider_type} not initialized")
            return self._providers[provider_type].get_supported_features()
    
    def get_provider_health_status(self) -> Dict[str, bool]:
        """
        Get health status of all providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        return {provider_type.value: status for provider_type, status in self._provider_health.items()}
    
    async def run_health_checks(self) -> Dict[str, bool]:
        """
        Run health checks on all initialized providers.
        
        Returns:
            Updated health status dictionary
        """
        for provider_type, provider in self._providers.items():
            try:
                is_healthy = await provider.health_check()
                self._provider_health[provider_type] = is_healthy
                logger.debug(f"Health check {provider_type}: {'✓' if is_healthy else '✗'}")
            except Exception as e:
                logger.warning(f"Health check failed for {provider_type}: {e}")
                self._provider_health[provider_type] = False
        
        return self.get_provider_health_status()
    
    def set_fallback_order(self, order: List[ProviderType]) -> None:
        """
        Set the fallback order for providers.
        
        Args:
            order: List of provider types in fallback priority order
        """
        self._fallback_order = order
        logger.info(f"Set fallback order: {[p.value for p in order]}")
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider types.
        
        Returns:
            List of provider type names
        """
        return [provider_type.value for provider_type in self._provider_classes.keys()]
    
    def __str__(self) -> str:
        primary_name = self._primary_provider.get_provider_name() if self._primary_provider else "None"
        return f"ProviderManager(primary={primary_name}, providers={len(self._providers)})"