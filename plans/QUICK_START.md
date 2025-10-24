# Quick Start: Using Different Models for Different Tasks

## The Simple Way 🚀

Use `chat_with_model()` to pick which model to use for each task:

```python
from src.llm.agent_helpers import chat_with_model

# Fast sentiment analysis with gpt-5-mini
messages = [{"role": "user", "content": "Analyze sentiment of this news..."}]
response = await chat_with_model(messages, "gpt-5-mini")

# Complex reasoning with Claude
messages = [{"role": "user", "content": "Explain economic impact of..."}]
response = await chat_with_model(messages, "claude-3.5-sonnet")

# Balanced task with gpt-4o
messages = [{"role": "user", "content": "Summarize this data..."}]
response = await chat_with_model(messages, "gpt-4o-2024-11-20")
```

## Inside Your Agent

```python
from src.llm.manager import LLMManager
from src.llm.agent_helpers import chat_with_model

class MyAgent:
    def __init__(self):
        self.llm_manager = LLMManager()
    
    async def analyze_sentiment(self, text: str):
        """Use gpt-5-mini for fast sentiment analysis"""
        messages = [
            {"role": "system", "content": "Analyze sentiment"},
            {"role": "user", "content": text}
        ]
        return await chat_with_model(messages, "gpt-5-mini", self.llm_manager)
    
    async def deep_analysis(self, data: str):
        """Use Claude for complex reasoning"""
        messages = [
            {"role": "system", "content": "Provide deep analysis"},
            {"role": "user", "content": data}
        ]
        return await chat_with_model(messages, "claude-3.5-sonnet", self.llm_manager)
```

## Available Models

All through your `COPILOT_ACCESS_TOKEN`:

**Fast & Cheap:**
- `gpt-5-mini` - Best for: sentiment, classification, simple extraction
- `gpt-4o-mini` - Alternative fast option

**Balanced:**
- `gpt-4o-2024-11-20` - Best for: general tasks, summaries
- `gpt-4o` - Same, shorter name

**Best Reasoning:**
- `claude-3.5-sonnet` - Best for: complex analysis, reasoning
- `claude-sonnet-4` - Even better (if available)

**Other Options:**
- `gpt-5` - Latest GPT
- `gemini-2.5-pro` - Google's latest

## Task Recommendations

```python
from src.llm.agent_helpers import get_recommended_model_for_task

# Get recommended model for a task type
model = get_recommended_model_for_task("sentiment_analysis")  # Returns "gpt-5-mini"
model = get_recommended_model_for_task("reasoning")            # Returns "claude-3.5-sonnet"

response = await chat_with_model(messages, model)
```

**Available task types:**
- `sentiment_analysis` → `gpt-5-mini`
- `classification` → `gpt-5-mini`
- `data_extraction` → `gpt-5-mini`
- `summarization` → `gpt-4o-2024-11-20`
- `reasoning` → `claude-3.5-sonnet`
- `analysis` → `claude-3.5-sonnet`

## Running Examples

```bash
# Test the setup
uv run python test_model_selection.py

# See practical examples
uv run python example_news_sentiment.py
```

## Configuration

Your `config.yaml` has these providers set up:

```yaml
llm:
  providers:
    copilot:          # gpt-4o-2024-11-20 (default)
    copilot_mini:     # gpt-5-mini (fast)
    copilot_claude:   # claude-3.5-sonnet (reasoning)
```

All use your `COPILOT_ACCESS_TOKEN`, so you only need one API key!

## Cost Optimization

**Strategy:**
1. Use `gpt-5-mini` for simple/fast tasks (sentiment, classification) - cheapest
2. Use `claude-3.5-sonnet` for complex reasoning only - expensive
3. Use `gpt-4o` for everything else - balanced

**Example in your economic agent:**
- News sentiment: `gpt-5-mini` ✓ Fast, cheap
- Event classification: `gpt-5-mini` ✓ Fast, cheap
- Economic impact analysis: `claude-3.5-sonnet` ✓ Best reasoning
- Summary generation: `gpt-4o-2024-11-20` ✓ Balanced

This can reduce your costs by **50-70%** compared to using GPT-4o for everything!

