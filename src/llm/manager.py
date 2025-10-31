"""
LLM Provider Manager with failover capabilities
"""

import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta

from .config import LLMConfig, load_config
from .types import BaseLLMProvider, ChatResponse
from .providers import CopilotProvider, OpenAIProvider, ClaudeProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Central manager for all LLM providers with failover capabilities
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_config()
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = None
        self.health_check_interval = timedelta(minutes=5)
        
        # Initialize providers
        self._initialize_providers()
        
        logger.info(f"LLMManager initialized with {len(self.providers)} providers")
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                logger.info(f"Skipping disabled provider: {provider_name}")
                continue
            
            # Determine provider class based on name
            # Support copilot variants (copilot_mini, copilot_claude, etc.)
            if provider_name.startswith('copilot'):
                provider_class = CopilotProvider
            elif provider_name == 'openai':
                provider_class = OpenAIProvider
            elif provider_name == 'claude':
                provider_class = ClaudeProvider
            else:
                logger.warning(f"Unknown provider: {provider_name}")
                continue
            
            try:
                provider = provider_class(provider_config)
                self.providers[provider_name] = provider
                
                # Initialize health status
                self.provider_health[provider_name] = {
                    'healthy': True,
                    'last_check': None,
                    'error_count': 0,
                    'last_error': None
                }
                
                logger.info(f"Initialized provider: {provider_name} ({provider.get_model_name()})")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
                # Still track it but mark as unhealthy
                self.provider_health[provider_name] = {
                    'healthy': False,
                    'last_check': datetime.now(),
                    'error_count': 1,
                    'last_error': str(e)
                }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_results[provider_name] = is_healthy
                
                # Update health status
                self.provider_health[provider_name].update({
                    'healthy': is_healthy,
                    'last_check': datetime.now()
                })
                
                if is_healthy:
                    # Reset error count on successful health check
                    self.provider_health[provider_name]['error_count'] = 0
                    self.provider_health[provider_name]['last_error'] = None
                
                logger.debug(f"Health check for {provider_name}: {'✓' if is_healthy else '✗'}")
                
            except Exception as e:
                health_results[provider_name] = False
                self.provider_health[provider_name].update({
                    'healthy': False,
                    'last_check': datetime.now(),
                    'error_count': self.provider_health[provider_name]['error_count'] + 1,
                    'last_error': str(e)
                })
                logger.error(f"Health check failed for {provider_name}: {e}")
        
        self.last_health_check = datetime.now()
        return health_results
    
    async def _should_check_health(self) -> bool:
        """Check if it's time for a health check"""
        if self.last_health_check is None:
            return True
        return datetime.now() - self.last_health_check > self.health_check_interval
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of currently healthy providers"""
        healthy = []
        for provider_name, health_info in self.provider_health.items():
            if health_info.get('healthy', False) and provider_name in self.providers:
                healthy.append(provider_name)
        return healthy
    
    def get_failover_order(self) -> List[str]:
        """Get the provider failover order, filtered by availability"""
        if not self.config.failover_enabled:
            # If failover is disabled, only return default provider
            default = self.config.default_provider
            return [default] if default in self.providers else []
        
        # Use configured failover order, filtered by available providers
        failover_order = self.config.failover_order or ['copilot', 'openai', 'claude']
        available_providers = list(self.providers.keys())
        
        return [provider for provider in failover_order if provider in available_providers]
    
    async def chat(self, messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None,
                   provider_name: Optional[str] = None) -> ChatResponse:
        """
        Send a chat request with automatic failover
        """
        # Periodic health check
        if await self._should_check_health():
            await self.health_check_all()
        
        # Determine provider order
        if provider_name:
            # Use specific provider if requested
            providers_to_try = [provider_name] if provider_name in self.providers else []
        else:
            # Use failover order
            providers_to_try = self.get_failover_order()
        
        if not providers_to_try:
            raise Exception("No available providers configured")
        
        last_error = None
        
        for provider_name in providers_to_try:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            # Skip unhealthy providers (but still try if it's the only one)
            provider_health_info = self.provider_health.get(provider_name, {})
            is_healthy = provider_health_info.get('healthy', True)
            
            if not is_healthy and len(providers_to_try) > 1:
                logger.warning(f"Skipping unhealthy provider: {provider_name}")
                continue
            
            try:
                logger.info(f"Attempting chat request with provider: {provider_name}")
                response = await provider.chat(messages, tools)
                
                # Mark provider as healthy on successful request
                self.provider_health[provider_name]['healthy'] = True
                self.provider_health[provider_name]['error_count'] = 0
                self.provider_health[provider_name]['last_error'] = None
                
                logger.info(f"Successful response from {provider_name} "
                           f"(model: {response.model}, tokens: {response.usage.get('total_tokens', 'unknown') if response.usage else 'unknown'})")
                
                return response
                
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                last_error = e
                
                # Update provider health
                health_info = self.provider_health[provider_name]
                health_info['error_count'] += 1
                health_info['last_error'] = str(e)
                
                # Mark as unhealthy if too many errors
                if health_info['error_count'] >= 3:
                    health_info['healthy'] = False
                    logger.warning(f"Marking provider {provider_name} as unhealthy due to repeated failures")
        
        # If we get here, all providers failed
        error_msg = f"All providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    async def stream_chat(self, messages: List[Dict[str, Any]],
                          tools: Optional[List[Dict[str, Any]]] = None,
                          provider_name: Optional[str] = None) -> AsyncGenerator[dict, None]:
        """
        Stream a chat request with automatic failover.

        Yields dictionaries with 'content', 'model', 'finish_reason', 'provider' keys.
        """
        # Periodic health check
        if await self._should_check_health():
            await self.health_check_all()

        # Determine provider order
        if provider_name:
            # Use specific provider if requested
            providers_to_try = [provider_name] if provider_name in self.providers else []
        else:
            # Use failover order
            providers_to_try = self.get_failover_order()

        if not providers_to_try:
            raise Exception("No available providers configured")

        last_error = None

        for provider_name in providers_to_try:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            # Check if provider supports streaming
            features = provider.get_supported_features()
            if not features.get("streaming", False):
                logger.warning(f"Provider {provider_name} does not support streaming, skipping")
                continue

            # Skip unhealthy providers (but still try if it's the only one)
            provider_health_info = self.provider_health.get(provider_name, {})
            is_healthy = provider_health_info.get('healthy', True)

            if not is_healthy and len(providers_to_try) > 1:
                logger.warning(f"Skipping unhealthy provider: {provider_name}")
                continue

            try:
                logger.info(f"Attempting streaming request with provider: {provider_name}")

                # Stream from provider
                async for chunk in provider.stream_chat(messages, tools):
                    yield chunk

                # Mark provider as healthy on successful request
                self.provider_health[provider_name]['healthy'] = True
                self.provider_health[provider_name]['error_count'] = 0
                self.provider_health[provider_name]['last_error'] = None

                logger.info(f"Successful streaming from {provider_name}")

                # If we got here, streaming was successful, don't try other providers
                return

            except Exception as e:
                logger.error(f"Provider {provider_name} streaming failed: {e}")
                last_error = e

                # Update provider health
                health_info = self.provider_health[provider_name]
                health_info['error_count'] += 1
                health_info['last_error'] = str(e)

                # Mark as unhealthy if too many errors
                if health_info['error_count'] >= 3:
                    health_info['healthy'] = False
                    logger.warning(f"Marking provider {provider_name} as unhealthy due to repeated failures")

        # If we get here, all providers failed
        error_msg = f"All providers failed streaming. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def get_default_provider(self) -> Optional[BaseLLMProvider]:
        """Get the default provider instance"""
        provider_name = self.config.default_provider
        return self.providers.get(provider_name)
    
    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Get a specific provider instance"""
        return self.providers.get(provider_name)
    
    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all providers with their status"""
        provider_info = {}
        
        for provider_name, provider in self.providers.items():
            health_info = self.provider_health.get(provider_name, {})
            
            provider_info[provider_name] = {
                'model': provider.get_model_name(),
                'healthy': health_info.get('healthy', False),
                'error_count': health_info.get('error_count', 0),
                'last_error': health_info.get('last_error'),
                'last_check': health_info.get('last_check'),
                'features': provider.get_supported_features(),
                'is_default': provider_name == self.config.default_provider
            }
        
        return provider_info
    
    async def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available models from all providers"""
        all_models = {}
        
        for provider_name, provider in self.providers.items():
            try:
                models = await provider.get_models()
                all_models[provider_name] = models
            except Exception as e:
                logger.error(f"Failed to get models from {provider_name}: {e}")
                all_models[provider_name] = []
        
        return all_models
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'default_provider': self.config.default_provider,
            'failover_enabled': self.config.failover_enabled,
            'failover_order': self.config.failover_order,
            'enabled_providers': list(self.providers.keys()),
            'total_providers': len(self.config.providers)
        }