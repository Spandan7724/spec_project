from types import SimpleNamespace

import pytest

from src.llm.config import LLMConfig
from src.llm.manager import LLMManager
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.types import ProviderConfig


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeModels:
    async def retrieve(self, model):
        return SimpleNamespace(id=model)

    async def list(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id="gpt-5.6", model_dump=lambda: {"id": "gpt-5.6"})]
        )


class FakeClient:
    def __init__(self, result):
        self.responses = FakeResponses(result)
        self.models = FakeModels()


def make_provider(client, **kwargs):
    return OpenAIProvider(
        ProviderConfig(
            name="openai_main",
            model="gpt-5.6",
            api_key_env="OPENAI_API_KEY",
            kwargs=kwargs,
        ),
        client=client,
    )


@pytest.mark.asyncio
async def test_chat_uses_responses_api_and_normalizes_tools_and_usage():
    response = SimpleNamespace(
        output_text="",
        model="gpt-5.6-2026-07-01",
        status="completed",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="set_parameters",
                arguments='{"currency_pair":"USD/EUR"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20),
    )
    client = FakeClient(response)
    provider = make_provider(
        client,
        max_tokens=512,
        reasoning={"effort": "medium"},
        store=False,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_parameters",
                "description": "Return parsed parameters",
                "parameters": {
                    "type": "object",
                    "properties": {"currency_pair": {"type": "string"}},
                },
            },
        }
    ]

    result = await provider.chat(
        [
            {"role": "system", "content": "Extract parameters."},
            {"role": "user", "content": "Convert USD to EUR."},
        ],
        tools=tools,
    )

    payload = client.responses.calls[0]
    assert payload["model"] == "gpt-5.6"
    assert payload["input"][1]["role"] == "user"
    assert payload["max_output_tokens"] == 512
    assert "max_tokens" not in payload
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "set_parameters",
            "description": "Return parsed parameters",
            "parameters": {
                "type": "object",
                "properties": {"currency_pair": {"type": "string"}},
            },
            "strict": False,
        }
    ]
    assert result.tool_calls[0]["id"] == "call_123"
    assert result.finish_reason == "tool_calls"
    assert result.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_chat_tool_transcript_is_mapped_to_responses_items():
    items = OpenAIProvider._format_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "lookup_rate",
                            "arguments": '{"pair":"USD/EUR"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "1.08"},
        ]
    )

    assert items == [
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup_rate",
            "arguments": '{"pair":"USD/EUR"}',
        },
        {"type": "function_call_output", "call_id": "call_123", "output": "1.08"},
    ]


@pytest.mark.asyncio
async def test_stream_chat_emits_responses_text_deltas_and_completion():
    async def events():
        yield SimpleNamespace(type="response.output_text.delta", delta="Hello")
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(model="gpt-5.6-2026-07-01"),
        )

    client = FakeClient(events())
    provider = make_provider(client)

    chunks = [
        chunk
        async for chunk in provider.stream_chat([{"role": "user", "content": "Hi"}])
    ]

    assert chunks[0]["content"] == "Hello"
    assert chunks[1]["finish_reason"] == "stop"
    assert client.responses.calls[0]["stream"] is True


def test_variant_config_and_base_name_routing(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm:
  default_provider: openai
  providers:
    openai_main:
      model: gpt-5.6
    openai_fast:
      model: gpt-5.6-luna
  failover:
    enabled: false
""".strip()
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = LLMConfig.from_yaml(str(config_file))
    manager = LLMManager(config)

    assert config.providers["openai_main"].api_key_env == "OPENAI_API_KEY"
    assert set(manager.providers) == {"openai_main", "openai_fast"}
    assert manager.get_provider("openai") is manager.providers["openai_main"]
    assert manager.get_default_provider() is manager.providers["openai_main"]
    assert manager.get_failover_order() == ["openai_main"]
