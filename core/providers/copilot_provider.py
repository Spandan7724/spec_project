"""
GitHub Copilot Provider for Multi-Agent Currency Assistant.

Implements GitHub Copilot API integration as the default LLM provider
with support for chat completions, tool calling, and streaming.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx

from .base_provider import BaseLLMProvider, ChatResponse, ProviderCapabilities

logger = logging.getLogger(__name__)


class CopilotProvider(BaseLLMProvider):
    """
    GitHub Copilot API provider implementation.
    
    Supports chat completions, tool/function calling, streaming responses,
    and model management via the GitHub Copilot API.
    """
    
    def __init__(self, model: str = "gpt-4o-2024-11-20", **kwargs):
        """
        Initialize GitHub Copilot provider.
        
        Args:
            model: Model to use (default: gpt-4o-2024-11-20)
            **kwargs: Additional configuration options
        """
        self.api_base = "https://api.githubcopilot.com"
        super().__init__(model, **kwargs)
        
        # Set up required headers for GitHub Copilot
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
            "editor-version": "CurrencyAssistant/1.0",
            "User-Agent": "CurrencyAssistant/1.0"
        }
        
        logger.debug(f"Initialized GitHub Copilot provider with model: {model}")
    
    def _validate_authentication(self) -> None:
        """Validate GitHub Copilot authentication token."""
        self.api_token = os.getenv("COPILOT_ACCESS_TOKEN")
        if not self.api_token:
            raise ValueError(
                "COPILOT_ACCESS_TOKEN environment variable is required for GitHub Copilot provider"
            )
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "copilot"
    
    def get_supported_features(self) -> ProviderCapabilities:
        """Return GitHub Copilot capabilities."""
        return ProviderCapabilities(
            function_calling=True,
            streaming=True,
            usage_tracking=True,
            temperature_control=True,
            max_tokens_control=True
        )
    
    def format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """GitHub Copilot uses OpenAI-compatible tool format."""
        return tools
    
    def parse_response_to_unified(self, response: Any) -> ChatResponse:
        """GitHub Copilot responses are already in the expected format."""
        return response
    
    async def chat(self, 
                   messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None,
                   **kwargs) -> ChatResponse:
        """
        Send a chat completion request to GitHub Copilot API.
        
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
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                **self.kwargs,
                **kwargs  # Allow override of default kwargs
            }
            
            # Add tools if provided
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                logger.debug(f"Added {len(tools)} tools to Copilot request")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"GitHub Copilot API error {response.status_code}: {error_text}")
                    
                    # Handle common error cases
                    if response.status_code == 401:
                        raise Exception("GitHub Copilot authentication failed. Check your COPILOT_ACCESS_TOKEN.")
                    elif response.status_code == 403:
                        raise Exception("GitHub Copilot access forbidden. Verify your Copilot subscription.")
                    elif response.status_code == 429:
                        raise Exception("GitHub Copilot rate limit exceeded. Please wait and try again.")
                    else:
                        raise Exception(f"GitHub Copilot API error {response.status_code}: {error_text}")
                
                response_data = response.json()
                
                # Extract the response content
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                # Extract tool calls if present
                tool_calls = None
                if "tool_calls" in message:
                    tool_calls = []
                    for tool_call in message["tool_calls"]:
                        tool_calls.append({
                            "id": tool_call.get("id"),
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"]
                            }
                        })
                
                # Extract usage information
                usage = None
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0)
                    }
                
                return ChatResponse(
                    content=content,
                    model=response_data.get("model", self.model),
                    usage=usage,
                    tool_calls=tool_calls,
                    finish_reason=choice.get("finish_reason")
                )
                
        except httpx.TimeoutException:
            logger.error("GitHub Copilot API request timed out")
            raise Exception("GitHub Copilot API request timed out")
        except httpx.RequestError as e:
            logger.error(f"GitHub Copilot API request error: {e}")
            raise Exception(f"GitHub Copilot API request failed: {e}")
        except Exception as e:
            logger.error(f"GitHub Copilot chat error: {e}")
            raise e
    
    async def stream_chat(self, 
                         messages: List[Dict[str, Any]], 
                         tools: Optional[List[Dict[str, Any]]] = None,
                         **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a chat completion request to GitHub Copilot API.
        
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
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **self.kwargs,
                **kwargs
            }
            
            # Add tools if provided
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                logger.debug(f"Added {len(tools)} tools to Copilot streaming request")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"GitHub Copilot streaming error {response.status_code}: {error_text}")
                        raise Exception(f"GitHub Copilot streaming failed: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                
                                # Extract delta content
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    if content:
                                        yield {
                                            "content": content,
                                            "model": chunk.get("model", self.model),
                                            "finish_reason": choices[0].get("finish_reason")
                                        }
                                        
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
                                
        except httpx.TimeoutException:
            logger.error("GitHub Copilot streaming request timed out")
            raise Exception("GitHub Copilot streaming timed out")
        except httpx.RequestError as e:
            logger.error(f"GitHub Copilot streaming error: {e}")
            raise Exception(f"GitHub Copilot streaming failed: {e}")
        except Exception as e:
            logger.error(f"GitHub Copilot streaming error: {e}")
            raise e
    
    def get_available_models(self) -> List[str]:
        """
        Get available models from GitHub Copilot.
        This is used as a fallback when API discovery fails.
        """
        return [
            "gpt-4o-2024-11-20",
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4",
            "gpt-4-turbo",
            "claude-3.5-sonnet",
            "claude-3.5-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "o1-preview",
            "o1-mini"
        ]
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get detailed model information from GitHub Copilot API.
        
        Returns:
            List of model dictionaries with details
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base}/models",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    logger.debug(f"GitHub Copilot API returned {len(models)} models")
                    return models
                else:
                    error_text = response.text
                    logger.warning(f"Failed to fetch models from Copilot API: {response.status_code} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Error fetching models from Copilot API: {e}")
            return []
    
    def validate_model_compatibility(self, model: str) -> bool:
        """Check if a model is supported by GitHub Copilot."""
        available_models = self.get_available_models()
        return model in available_models
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Get GitHub Copilot rate limits."""
        return {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,  # Generous limit for Copilot
            "requests_per_day": 10000
        }