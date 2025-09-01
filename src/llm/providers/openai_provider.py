"""
OpenAI API provider for Currency Assistant.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import httpx

from ..types import BaseLLMProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = "https://api.openai.com/v1"
        
        # Validate authentication
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CurrencyAssistant/1.0"
        }
        
        logger.debug(f"Initialized OpenAI provider with model: {self.model}")
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def get_model_name(self) -> str:
        return self.model
    
    def get_supported_features(self) -> Dict[str, bool]:
        return {
            "function_calling": True,
            "streaming": True,
            "usage_tracking": True,
            "temperature_control": True,
            "max_tokens_control": True
        }
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.api_base}/models",
                    headers=self.headers
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    async def chat(self, messages: List[Dict[str, Any]], 
                   tools: Optional[List[Dict[str, Any]]] = None) -> ChatResponse:
        """
        Send a chat completion request to OpenAI API.
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                **self.kwargs
            }
            
            # Add tools if provided
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                logger.debug(f"Added {len(tools)} tools to OpenAI request")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenAI API error {response.status_code}: {error_text}")
                    
                    # Handle common error cases
                    if response.status_code == 401:
                        raise Exception("OpenAI authentication failed. Check your OPENAI_API_KEY.")
                    elif response.status_code == 403:
                        raise Exception("OpenAI access forbidden. Check your API permissions.")
                    elif response.status_code == 429:
                        raise Exception("OpenAI rate limit exceeded. Please wait and try again.")
                    elif response.status_code == 400:
                        raise Exception(f"OpenAI API bad request: {error_text}")
                    else:
                        raise Exception(f"OpenAI API error {response.status_code}: {error_text}")
                
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
                    finish_reason=choice.get("finish_reason"),
                    provider="openai"
                )
                
        except httpx.TimeoutException:
            logger.error("OpenAI API request timed out")
            raise Exception("OpenAI API request timed out")
        except httpx.RequestError as e:
            logger.error(f"OpenAI API request error: {e}")
            raise Exception(f"OpenAI API request failed: {e}")
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise e
    
    def get_available_models(self) -> List[str]:
        """
        Get available OpenAI models.
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125"
        ]
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get detailed model information from OpenAI API.
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
                    # Filter to only chat completion models
                    chat_models = [
                        model for model in models 
                        if model.get("id", "").startswith(("gpt-", "text-davinci"))
                    ]
                    logger.debug(f"OpenAI API returned {len(chat_models)} chat models")
                    return chat_models
                else:
                    error_text = response.text
                    logger.warning(f"Failed to fetch models from OpenAI API: {response.status_code} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Error fetching models from OpenAI API: {e}")
            return []