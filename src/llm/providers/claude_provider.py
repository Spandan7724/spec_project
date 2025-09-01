"""
Anthropic Claude API provider for Currency Assistant.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import httpx

from ..types import BaseLLMProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = "https://api.anthropic.com/v1"
        
        # Validate authentication
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        # Set up headers
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "CurrencyAssistant/1.0"
        }
        
        logger.debug(f"Initialized Claude provider with model: {self.model}")
    
    def get_provider_name(self) -> str:
        return "claude"
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_supported_features(self) -> Dict[str, bool]:
        return {
            "function_calling": True,
            "streaming": False,  # Claude streaming has different format
            "usage_tracking": True,
            "temperature_control": True,
            "max_tokens_control": True
        }
    
    async def health_check(self) -> bool:
        """Check if Claude API is accessible"""
        try:
            # Claude doesn't have a simple models endpoint, so we'll do a minimal chat test
            test_payload = {
                "model": self.model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.api_base}/messages",
                    headers=self.headers,
                    json=test_payload
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            return False
    
    def _convert_messages_to_claude_format(self, messages: List[Dict[str, Any]]) -> tuple:
        """Convert OpenAI-style messages to Claude format"""
        system_message = None
        claude_messages = []
        
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                # Claude handles system messages separately
                if system_message is None:
                    system_message = content
                else:
                    system_message += "\n\n" + content
            elif role in ["user", "assistant"]:
                claude_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool":
                # Handle tool results - convert to user message with context
                tool_content = f"Tool result: {content}"
                claude_messages.append({
                    "role": "user", 
                    "content": tool_content
                })
        
        return system_message, claude_messages
    
    def _convert_tools_to_claude_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Claude format"""
        claude_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                claude_tool = {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {})
                }
                claude_tools.append(claude_tool)
        
        return claude_tools
    
    async def chat(self, messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None) -> ChatResponse:
        """
        Send a chat completion request to Claude API.
        """
        try:
            # Convert messages to Claude format
            system_message, claude_messages = self._convert_messages_to_claude_format(messages)
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": claude_messages,
                "max_tokens": self.kwargs.get("max_tokens", 4000)
            }
            
            # Add system message if present
            if system_message:
                payload["system"] = system_message
            
            # Add other kwargs (temperature, etc.)
            for key, value in self.kwargs.items():
                if key not in ["max_tokens"] and value is not None:
                    payload[key] = value
            
            # Add tools if provided
            if tools:
                claude_tools = self._convert_tools_to_claude_format(tools)
                payload["tools"] = claude_tools
                logger.debug(f"Added {len(claude_tools)} tools to Claude request")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/messages",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Claude API error {response.status_code}: {error_text}")
                    
                    # Handle common error cases
                    if response.status_code == 401:
                        raise Exception("Claude authentication failed. Check your ANTHROPIC_API_KEY.")
                    elif response.status_code == 403:
                        raise Exception("Claude access forbidden. Check your API permissions.")
                    elif response.status_code == 429:
                        raise Exception("Claude rate limit exceeded. Please wait and try again.")
                    elif response.status_code == 400:
                        raise Exception(f"Claude API bad request: {error_text}")
                    else:
                        raise Exception(f"Claude API error {response.status_code}: {error_text}")
                
                response_data = response.json()
                
                # Extract content from Claude response
                content_blocks = response_data.get("content", [])
                content = ""
                tool_calls = []
                
                for block in content_blocks:
                    if block.get("type") == "text":
                        content += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        # Convert Claude tool use to OpenAI format
                        tool_calls.append({
                            "id": block.get("id"),
                            "function": {
                                "name": block.get("name"),
                                "arguments": str(block.get("input", {}))
                            }
                        })
                
                # Extract usage information
                usage = None
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    usage = {
                        "prompt_tokens": usage_data.get("input_tokens", 0),
                        "completion_tokens": usage_data.get("output_tokens", 0),
                        "total_tokens": usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
                    }
                
                return ChatResponse(
                    content=content,
                    model=response_data.get("model", self.model),
                    usage=usage,
                    tool_calls=tool_calls if tool_calls else None,
                    finish_reason=response_data.get("stop_reason"),
                    provider="claude"
                )
                
        except httpx.TimeoutException:
            logger.error("Claude API request timed out")
            raise Exception("Claude API request timed out")
        except httpx.RequestError as e:
            logger.error(f"Claude API request error: {e}")
            raise Exception(f"Claude API request failed: {e}")
        except Exception as e:
            logger.error(f"Claude chat error: {e}")
            raise e
    
    def get_available_models(self) -> List[str]:
        """
        Get available Claude models.
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get available models - Claude API doesn't provide a models endpoint,
        so we return the known models list.
        """
        models = []
        for model_id in self.get_available_models():
            models.append({
                "id": model_id,
                "object": "model",
                "provider": "claude"
            })
        
        logger.debug(f"Returning {len(models)} Claude models")
        return models