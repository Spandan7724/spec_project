"""
Common types and schemas for LLM providers
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


@dataclass
class ChatResponse:
    """Unified response format across all LLM providers"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider"""
    name: str
    model: str
    enabled: bool = True
    api_key_env: Optional[str] = None
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model = config.model
        self.kwargs = config.kwargs
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod  
    def get_model_name(self) -> str:
        """Return the current model name"""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None) -> ChatResponse:
        """Send a chat completion request"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available and healthy"""
        pass
    
    def format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for this specific provider (override if needed)"""
        return tools
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Return supported features for this provider"""
        return {
            "function_calling": False,
            "streaming": False,
            "usage_tracking": False,
            "temperature_control": False,
            "max_tokens_control": False
        }