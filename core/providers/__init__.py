"""
LLM Provider System for Multi-Agent Currency Assistant.

Supports multiple LLM providers including GitHub Copilot (default), 
OpenAI, and Anthropic with unified interface and automatic failover.
"""

from .base_provider import BaseLLMProvider, ChatResponse, ProviderCapabilities
from .copilot_provider import CopilotProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .provider_manager import ProviderManager, ProviderType

__all__ = [
    "BaseLLMProvider", 
    "ChatResponse", 
    "ProviderCapabilities",
    "CopilotProvider",
    "OpenAIProvider", 
    "AnthropicProvider",
    "ProviderManager", 
    "ProviderType"
]