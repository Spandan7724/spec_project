"""
Anthropic Provider for Multi-Agent Currency Assistant.

Implements Anthropic Claude API integration as an alternative LLM provider
with support for Claude models, tool calling, and streaming.
"""

import os
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from .base_provider import BaseLLMProvider, ChatResponse, ProviderCapabilities

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider implementation using LangChain integration.
    
    Supports Claude models, tool calling, streaming responses,
    and integrates with LangChain ecosystem.
    """
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
            **kwargs: Additional configuration options
        """
        super().__init__(model, **kwargs)
        
        # Initialize LangChain ChatAnthropic client
        self.client = ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            **self.kwargs
        )
        
        logger.debug(f"Initialized Anthropic provider with model: {model}")
    
    def _validate_authentication(self) -> None:
        """Validate Anthropic API key."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
    
    def get_supported_features(self) -> ProviderCapabilities:
        """Return Anthropic capabilities."""
        return ProviderCapabilities(
            function_calling=True,
            streaming=True,
            usage_tracking=True,
            temperature_control=True,
            max_tokens_control=True
        )
    
    def _convert_messages_to_langchain(self, messages: List[Dict[str, Any]]) -> List:
        """Convert OpenAI format messages to LangChain format."""
        langchain_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "tool":
                langchain_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
            else:
                # Default to human message
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Anthropic uses OpenAI-compatible tool format with some adaptations."""
        # Anthropic tool format is very similar to OpenAI
        return tools
    
    async def chat(self, 
                   messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None,
                   **kwargs) -> ChatResponse:
        """
        Send a chat completion request to Anthropic API.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse with the model's response
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(messages)
            
            # Create a new client with updated parameters if needed
            client_kwargs = {**self.kwargs, **kwargs}
            if client_kwargs != self.kwargs:
                client = ChatAnthropic(
                    model=self.model,
                    api_key=self.api_key,
                    **client_kwargs
                )
            else:
                client = self.client
            
            # Bind tools if provided
            if tools:
                client = client.bind_tools(tools)
                logger.debug(f"Added {len(tools)} tools to Anthropic request")
            
            # Make the request
            response = await client.ainvoke(langchain_messages)
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = []
                for tool_call in response.tool_calls:
                    tool_calls.append({
                        "id": tool_call.get("id"),
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": str(tool_call.get("args", {}))
                        }
                    })
            
            # Extract usage information if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0)
                }
            
            return ChatResponse(
                content=response.content,
                model=self.model,
                usage=usage,
                tool_calls=tool_calls,
                finish_reason=getattr(response, 'finish_reason', None)
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API request failed: {e}")
    
    async def stream_chat(self, 
                         messages: List[Dict[str, Any]], 
                         tools: Optional[List[Dict[str, Any]]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a chat completion request to Anthropic API.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            tools: Optional list of tools/functions the model can call
            **kwargs: Additional parameters
            
        Yields:
            Dictionary with content chunks as they arrive
            
        Raises:
            Exception: If the streaming request fails
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(messages)
            
            # Create a streaming client
            client_kwargs = {**self.kwargs, **kwargs, "streaming": True}
            client = ChatAnthropic(
                model=self.model,
                api_key=self.api_key,
                **client_kwargs
            )
            
            # Bind tools if provided
            if tools:
                client = client.bind_tools(tools)
                logger.debug(f"Added {len(tools)} tools to Anthropic streaming request")
            
            # Stream the response
            async for chunk in client.astream(langchain_messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield {
                        "content": chunk.content,
                        "model": self.model,
                        "finish_reason": getattr(chunk, 'finish_reason', None)
                    }
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise Exception(f"Anthropic streaming failed: {e}")
    
    async def get_available_models(self) -> List[str]:
        """
        Get available Anthropic models.
        
        Returns:
            List of Claude model identifiers
        """
        # Anthropic Claude models available as of implementation
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    def validate_model_compatibility(self, model: str) -> bool:
        """Check if a model is supported by Anthropic."""
        available_models = self.get_available_models()
        return model in available_models
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Get Anthropic rate limits (tier-dependent)."""
        return {
            "requests_per_minute": 1000,  # Tier 1 default
            "tokens_per_minute": 40000,
            "requests_per_day": 50000
        }