# LLM Model Selection Guide

This guide explains how to use different LLM models for different agents in the Currency Assistant project.

## Available Models via GitHub Copilot

Through your GitHub Copilot subscription, you have access to:

### OpenAI Models
- `gpt-4.1` / `gpt-4.1-2025-04-14`
- `gpt-5` / `gpt-5-mini`
- `gpt-4o` / `gpt-4o-2024-11-20` / `gpt-4o-2024-08-06` / `gpt-4o-2024-05-13`
- `gpt-4o-mini` / `gpt-4o-mini-2024-07-18`
- `gpt-4` / `gpt-4-0613` / `gpt-4-0125-preview` / `gpt-4-o-preview`
- `gpt-3.5-turbo` / `gpt-3.5-turbo-0613`
- `o3-mini` / `o3-mini-2025-01-31` / `o3-mini-paygo`
- `o4-mini` / `o4-mini-2025-04-16`

### Claude Models (via Copilot)
- `claude-3.5-sonnet`
- `claude-3.7-sonnet` / `claude-3.7-sonnet-thought`
- `claude-sonnet-4` / `claude-sonnet-4.5`

### Gemini Models (via Copilot)
- `gemini-2.0-flash-001`
- `gemini-2.5-pro`

### Embedding Models
- `text-embedding-ada-002`
- `text-embedding-3-small` / `text-embedding-3-small-inference`

## Configuration

### 1. Basic Setup in `config.yaml`

You can define multiple "providers" that all use the GitHub Copilot API but with different models:

```yaml
llm:
  default_provider: "copilot"
  providers:
    copilot:
      model: "gpt-4o-2024-11-20"
      enabled: true
      kwargs:
        temperature: 0.7
        max_tokens: 128000
    
    copilot_mini:
      model: "gpt-5-mini"
      enabled: true
      kwargs:
        temperature: 0.7
        max_tokens: 128000
    
    copilot_claude:
      model: "claude-3.7-sonnet"
      enabled: true
      kwargs:
        temperature: 0.7
        max_tokens: 128000
    
    copilot_fast:
      model: "gpt-4o-mini"
      enabled: true
      kwargs:
        temperature: 0.5
        max_tokens: 64000
  
  failover:
    enabled: true
    order: ["copilot", "copilot_mini", "openai", "claude"]
```

All providers starting with `copilot` will use the `COPILOT_ACCESS_TOKEN` environment variable.

## Usage Methods

### Method 1: Specify Provider at Call Time

```python
from src.llm.manager import LLMManager

llm_manager = LLMManager()

# Use default provider (gpt-4o)
response = await llm_manager.chat(messages)

# Use gpt-5-mini specifically
response = await llm_manager.chat(messages, provider_name="copilot_mini")

# Use Claude via Copilot
response = await llm_manager.chat(messages, provider_name="copilot_claude")
```

### Method 2: Create Agent-Specific LLM Managers

```python
from src.llm.agent_helpers import create_agent_llm

# Create LLM manager for decision agent using gpt-5-mini
decision_llm = create_agent_llm(model_name="gpt-5-mini")

# Or use provider name directly
decision_llm = create_agent_llm(provider_name="copilot_mini")

# Use with agent
from src.agentic.nodes.decision import DecisionCoordinatorAgent
decision_agent = DecisionCoordinatorAgent(llm_manager=decision_llm)
```

### Method 3: Use AgentLLMFactory (Recommended)

The `AgentLLMFactory` provides a centralized way to manage which models are used for which agents:

```python
from src.llm.agent_helpers import AgentLLMFactory

# Define your agent-model mappings (already configured in agent_helpers.py)
# decision -> gpt-5-mini (fast, cheap)
# economic -> claude-3.7-sonnet (reasoning)
# market -> gpt-4o (default, balanced)
# risk -> gpt-5-mini (fast)

# Create agents with their designated models
decision_llm = AgentLLMFactory.get_llm_for_agent("decision")
decision_agent = DecisionCoordinatorAgent(llm_manager=decision_llm)

# View current configuration
summary = AgentLLMFactory.get_agent_models_summary()
print(summary)
# Output:
# {
#   "decision": {"provider": "copilot_mini", "model": "gpt-5-mini"},
#   "economic": {"provider": "copilot_claude", "model": "claude-3.7-sonnet"},
#   ...
# }

# Change model for an agent type at runtime
AgentLLMFactory.update_agent_model("decision", "copilot_claude")
```

### Method 4: Modify Agent Initialization

For a more permanent solution, modify the agent's `__init__` method:

```python
# In src/agentic/nodes/decision.py
from src.llm.agent_helpers import AgentLLMFactory

class DecisionCoordinatorAgent:
    def __init__(self, llm_manager: Optional[LLMManager] = None) -> None:
        if llm_manager is not None:
            self.llm_manager = llm_manager
        else:
            try:
                # Use gpt-5-mini by default for decision agent
                self.llm_manager = AgentLLMFactory.get_llm_for_agent("decision")
            except Exception as exc:
                logger.warning("LLMManager unavailable: %s", exc)
                self.llm_manager = None
```

## Model Selection Strategy

Here's a recommended strategy for different agent types:

| Agent Type | Recommended Model | Reason |
|------------|------------------|---------|
| **Decision** | `gpt-5-mini` or `gpt-4o-mini` | Fast, cost-effective for final synthesis |
| **Market Analysis** | `gpt-4o` | Balanced performance and accuracy |
| **Economic Analysis** | `claude-3.7-sonnet` | Excellent reasoning for economic events |
| **Risk Assessment** | `gpt-5-mini` | Quick calculations, pattern recognition |
| **Complex Reasoning** | `claude-sonnet-4.5` | Best for deep analysis |
| **Fast Iterations** | `gpt-4o-mini` | Quick responses for testing |

## Cost Optimization Tips

1. **Use cheaper models for simple tasks**: `gpt-5-mini` and `gpt-4o-mini` are great for straightforward tasks
2. **Use powerful models sparingly**: Save `gpt-4o`, `claude-sonnet-4.5` for complex reasoning
3. **Adjust max_tokens**: Reduce `max_tokens` for responses that don't need to be long
4. **Lower temperature for consistency**: Use `temperature: 0.5` or lower for more deterministic outputs

## Testing Different Models

Use the example script to test different models:

```bash
# Activate your virtual environment
source venv/bin/activate  # or your venv path

# Run the example
python example_agent_specific_models.py
```

## Checking Available Models

To see all models available through your Copilot subscription:

```bash
curl -s https://api.githubcopilot.com/models \
  -H "Authorization: Bearer $COPILOT_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Copilot-Integration-Id: vscode-chat" | jq -r '.data[].id'
```

Or programmatically:

```python
from src.llm.providers.copilot_provider import CopilotProvider
from src.llm.types import ProviderConfig

config = ProviderConfig(
    name="copilot",
    model="gpt-4o",
    api_key_env="COPILOT_ACCESS_TOKEN"
)

provider = CopilotProvider(config)

# List all available models
models = await provider.get_models()
for model in models:
    print(f"- {model['id']}")
```

## Environment Variables

Make sure you have the appropriate API keys set:

```bash
# Required for Copilot providers
export COPILOT_ACCESS_TOKEN="your_token_here"

# Only needed if using direct OpenAI/Claude providers
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

## Troubleshooting

### Provider not initializing

Check that:
1. The provider is `enabled: true` in `config.yaml`
2. The environment variable is set (e.g., `COPILOT_ACCESS_TOKEN`)
3. The model name is spelled correctly

### Model not found

The model might not be available through your Copilot subscription. Run the models check command above to verify.

### Failover not working

Check the `failover.order` in `config.yaml` and ensure the providers are enabled.

## Advanced: Dynamic Model Selection

For dynamic model selection based on request complexity:

```python
class SmartLLMManager:
    def __init__(self):
        self.llm_manager = LLMManager()
    
    async def chat_with_auto_model(self, messages, complexity="medium"):
        """Select model based on complexity"""
        model_map = {
            "simple": "copilot_mini",      # gpt-5-mini
            "medium": "copilot",            # gpt-4o
            "complex": "copilot_claude",    # claude-3.7-sonnet
        }
        provider = model_map.get(complexity, "copilot")
        return await self.llm_manager.chat(messages, provider_name=provider)

# Usage
smart_llm = SmartLLMManager()
response = await smart_llm.chat_with_auto_model(messages, complexity="simple")
```

## References

- [LLM Manager Source](src/llm/manager.py)
- [Agent Helpers Source](src/llm/agent_helpers.py)
- [Example Script](example_agent_specific_models.py)
- [Configuration File](config.yaml)

