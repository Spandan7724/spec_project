"""
Base LLM Provider Interface for Multi-Agent Currency Assistant.

Defines the contract that all LLM providers must implement to ensure
consistent behavior across different providers (Copilot, OpenAI, Anthropic).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Unified response format for all LLM providers."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ProviderCapabilities:
    """Defines what features a provider supports."""
    function_calling: bool = False
    streaming: bool = False
    usage_tracking: bool = False
    temperature_control: bool = False
    max_tokens_control: bool = False


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Provides a unified interface for different LLM providers while allowing
    provider-specific implementations of authentication, API calls, and features.
    """
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the provider with a specific model.
        
        Args:
            model: The model identifier to use
            **kwargs: Provider-specific configuration options
        """
        self.model = model
        self.kwargs = kwargs
        self._capabilities = None
        
        # Validate authentication during initialization
        self._validate_authentication()
        
        logger.debug(f"Initialized {self.get_provider_name()} provider with model: {model}")
    
    @abstractmethod
    def _validate_authentication(self) -> None:
        """
        Validate that the provider has proper authentication.
        Should raise ValueError if authentication is invalid.
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider (e.g., 'copilot', 'openai', 'anthropic')."""
        pass
    
    def get_model_name(self) -> str:
        """Return the current model name."""
        return self.model
    
    @abstractmethod
    def get_supported_features(self) -> ProviderCapabilities:
        """Return the capabilities supported by this provider."""
        pass
    
    @abstractmethod
    async def chat(self, 
                   messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None,
                   **kwargs) -> ChatResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ChatResponse with unified format
            
        Raises:
            Exception: If the request fails
        """
        pass
    
    @abstractmethod
    async def stream_chat(self, 
                         messages: List[Dict[str, Any]], 
                         tools: Optional[List[Dict[str, Any]]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a chat completion request.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Dictionary with content chunks as they arrive
            
        Raises:
            Exception: If the request fails
        """
        pass
    
    def format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for this provider's specific format.
        Default implementation assumes OpenAI-compatible format.
        
        Args:
            tools: Tools in standard format
            
        Returns:
            Tools formatted for this provider
        """
        return tools
    
    def parse_response_to_unified(self, response: Any) -> ChatResponse:
        """
        Parse provider-specific response to unified format.
        Default implementation assumes response is already in ChatResponse format.
        
        Args:
            response: Provider-specific response
            
        Returns:
            Unified ChatResponse format
        """
        if isinstance(response, ChatResponse):
            return response
        # Subclasses should override this for provider-specific parsing
        raise NotImplementedError("Provider must implement response parsing")
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        Default implementation returns empty list.
        
        Returns:
            List of model identifiers
        """
        return []
    
    def validate_model_compatibility(self, model: str) -> bool:
        """
        Check if a model is compatible with this provider.
        
        Args:
            model: Model identifier to check
            
        Returns:
            True if model is supported
        """
        # Default implementation - subclasses should override
        return True
    
    def get_rate_limits(self) -> Dict[str, int]:
        """
        Get rate limit information for this provider.
        
        Returns:
            Dictionary with rate limit details (requests_per_minute, tokens_per_minute, etc.)
        """
        return {
            "requests_per_minute": 60,  # Default fallback
            "tokens_per_minute": 10000
        }
    
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and responsive.
        
        Returns:
            True if provider is healthy
        """
        try:
            # Simple test message to verify connectivity
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat(test_messages)
            return response is not None and len(response.content) > 0
        except Exception as e:
            logger.warning(f"Health check failed for {self.get_provider_name()}: {e}")
            return False
    
    def __str__(self) -> str:
        return f"{self.get_provider_name()}({self.model})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model='{self.model}'>"