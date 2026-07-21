"""OpenAI Responses API provider for Currency Assistant."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from ..types import BaseLLMProvider, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider backed by the recommended Responses API."""

    _CLIENT_KWARGS = {"base_url", "max_retries", "organization", "project", "timeout"}

    def __init__(self, config: ProviderConfig, client: Optional[AsyncOpenAI] = None):
        super().__init__(config)

        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env)
        if client is None and not api_key:
            raise ValueError(f"{api_key_env} environment variable is required")

        raw_kwargs = dict(self.kwargs)
        client_kwargs = {
            key: raw_kwargs.pop(key)
            for key in list(raw_kwargs)
            if key in self._CLIENT_KWARGS
        }
        self.request_kwargs = self._normalize_request_kwargs(raw_kwargs)
        self.client = client or AsyncOpenAI(api_key=api_key, **client_kwargs)

        logger.debug("Initialized OpenAI Responses provider with model: %s", self.model)

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
            "max_tokens_control": True,
            "responses_api": True,
            "reasoning": True,
        }

    async def health_check(self) -> bool:
        """Verify authentication and access to the configured model."""
        try:
            await self.client.models.retrieve(self.model)
            return True
        except Exception as exc:
            logger.warning("OpenAI health check failed: %s", exc)
            return False

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Create a non-streaming response using OpenAI's Responses API."""
        payload = self._build_payload(messages, tools)

        try:
            response = await self.client.responses.create(**payload)
        except Exception as exc:
            self._raise_provider_error(exc)

        output = self._get(response, "output", []) or []
        tool_calls = []
        for item in output:
            if self._get(item, "type") == "function_call":
                tool_calls.append(
                    {
                        "id": self._get(item, "call_id") or self._get(item, "id"),
                        "function": {
                            "name": self._get(item, "name", ""),
                            "arguments": self._get(item, "arguments", "{}"),
                        },
                    }
                )

        usage = self._usage_dict(self._get(response, "usage"))
        status = self._get(response, "status")
        finish_reason = (
            "tool_calls" if tool_calls else "stop" if status == "completed" else status
        )

        return ChatResponse(
            content=self._get(response, "output_text", "") or "",
            model=self._get(response, "model", self.model),
            usage=usage,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            provider="openai",
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream text delta events from OpenAI's Responses API."""
        payload = self._build_payload(messages, tools)
        payload["stream"] = True

        try:
            stream = await self.client.responses.create(**payload)
            async for event in stream:
                event_type = self._get(event, "type")
                if event_type == "response.output_text.delta":
                    delta = self._get(event, "delta", "")
                    if delta:
                        yield {
                            "content": delta,
                            "model": self.model,
                            "finish_reason": None,
                            "provider": "openai",
                        }
                elif event_type == "response.completed":
                    completed = self._get(event, "response")
                    yield {
                        "content": "",
                        "model": self._get(completed, "model", self.model),
                        "finish_reason": "stop",
                        "provider": "openai",
                    }
        except Exception as exc:
            self._raise_provider_error(exc, streaming=True)

    def format_tools_for_provider(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Chat Completions function definitions to Responses tools."""
        formatted = []
        for tool in tools:
            if tool.get("type") == "function" and isinstance(
                tool.get("function"), dict
            ):
                function = tool["function"]
                formatted_tool = {
                    "type": "function",
                    "name": function["name"],
                    "parameters": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                    # Responses defaults differ from Chat Completions. Preserve the
                    # existing provider contract unless strict mode is requested.
                    "strict": function.get("strict", False),
                }
                if function.get("description"):
                    formatted_tool["description"] = function["description"]
                formatted.append(formatted_tool)
            else:
                # Native Responses tools (web search, file search, MCP, etc.)
                # already use the correct shape.
                formatted.append(dict(tool))
        return formatted

    def get_available_models(self) -> List[str]:
        """Return the documented GPT-5.6 model family used by this project."""
        return ["gpt-5.6", "gpt-5.6-terra", "gpt-5.6-luna"]

    async def get_models(self) -> List[Dict[str, Any]]:
        """List GPT models visible to the configured OpenAI project."""
        try:
            page = await self.client.models.list()
            models = []
            for model in self._get(page, "data", []) or []:
                model_id = self._get(model, "id", "")
                if not model_id.startswith("gpt-"):
                    continue
                if hasattr(model, "model_dump"):
                    models.append(model.model_dump())
                elif isinstance(model, dict):
                    models.append(dict(model))
                else:
                    models.append({"id": model_id})
            return models
        except Exception as exc:
            logger.warning("Error fetching models from OpenAI API: %s", exc)
            return []

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": self._format_input(messages),
            **self.request_kwargs,
        }
        if tools:
            payload["tools"] = self.format_tools_for_provider(tools)
            payload["tool_choice"] = "auto"
        return payload

    @classmethod
    def _format_input(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map legacy chat transcripts to Responses messages/items."""
        items: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")

            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id"),
                        "output": cls._string_content(message.get("content", "")),
                    }
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                if message.get("content"):
                    items.append({"role": "assistant", "content": message["content"]})
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
                continue

            items.append(dict(message))
        return items

    @staticmethod
    def _string_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _normalize_request_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(kwargs)
        if "max_tokens" in normalized and "max_output_tokens" not in normalized:
            normalized["max_output_tokens"] = normalized.pop("max_tokens")
        if "response_format" in normalized and "text" not in normalized:
            response_format = normalized.pop("response_format")
            if (
                response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                normalized["text"] = {
                    "format": {"type": "json_schema", **response_format["json_schema"]}
                }
            else:
                normalized["text"] = {"format": response_format}
        return normalized

    @classmethod
    def _usage_dict(cls, usage: Any) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        input_tokens = cls._get(usage, "input_tokens", 0)
        output_tokens = cls._get(usage, "output_tokens", 0)
        return {
            # Keep the cross-provider ChatResponse vocabulary stable.
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": cls._get(
                usage, "total_tokens", input_tokens + output_tokens
            ),
        }

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _raise_provider_error(exc: Exception, streaming: bool = False) -> None:
        operation = "streaming request" if streaming else "request"
        if isinstance(exc, AuthenticationError):
            message = "OpenAI authentication failed. Check your OPENAI_API_KEY."
        elif isinstance(exc, PermissionDeniedError):
            message = (
                "OpenAI access forbidden. Check your project and model permissions."
            )
        elif isinstance(exc, RateLimitError):
            message = "OpenAI rate limit exceeded. Please wait and try again."
        elif isinstance(exc, BadRequestError):
            message = f"OpenAI rejected the request: {exc}"
        elif isinstance(exc, APITimeoutError):
            message = f"OpenAI {operation} timed out"
        elif isinstance(exc, APIConnectionError):
            message = f"OpenAI {operation} failed to connect: {exc}"
        elif isinstance(exc, APIStatusError):
            message = f"OpenAI API error {exc.status_code}: {exc}"
        else:
            message = f"OpenAI {operation} failed: {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc
