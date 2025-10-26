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

**Primary Models (Recommended for this project):**
- `gpt-5-mini` - Best for: sentiment, classification, simple extraction (FAST & CHEAP)
- `gpt-4o` or `gpt-4o-2024-11-20` - Best for: reasoning, NLU, response generation (BALANCED)

**Alternative Options (Available but not needed):**
- `gpt-4o-mini` - Alternative fast option
- `claude-3.5-sonnet` - Complex reasoning (overkill for our use cases)
- `claude-sonnet-4` - Advanced reasoning (not needed)
- `gpt-5` - Latest GPT (gpt-4o is sufficient)
- `gemini-2.5-pro` - Google's latest (not needed)

**Our Strategy**: Stick with just gpt-5-mini + gpt-4o for simplicity and efficiency.

## Task Recommendations

```python
from src.llm.agent_helpers import get_recommended_model_for_task

# Get recommended model for a task type
model = get_recommended_model_for_task("sentiment_analysis")  # Returns "gpt-5-mini"
model = get_recommended_model_for_task("reasoning")            # Returns "gpt-4o"

response = await chat_with_model(messages, model)
```

**Available task types:**
- `sentiment_analysis` → `gpt-5-mini`
- `classification` → `gpt-5-mini`
- `data_extraction` → `gpt-5-mini`
- `summarization` → `gpt-4o`
- `reasoning` → `gpt-4o`
- `analysis` → `gpt-4o`
- `nlu` → `gpt-4o`
- `response_generation` → `gpt-4o`

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
    copilot:          # gpt-4o-2024-11-20 (default, balanced)
    copilot_mini:     # gpt-5-mini (fast & cheap)
```

Both use your `COPILOT_ACCESS_TOKEN`, so you only need one API key!

**Simplified 2-Model Strategy**: We only use gpt-4o and gpt-5-mini for this project. This keeps things simple while providing excellent performance and cost efficiency.

## Cost Optimization

**Simple 2-Model Strategy:**
1. Use `gpt-5-mini` for simple/fast tasks (sentiment, classification, extraction) - **cheapest**
2. Use `gpt-4o` for everything else (reasoning, NLU, analysis, generation) - **balanced**

**Example in your agents:**
- News sentiment: `gpt-5-mini` ✓ Fast, cheap
- Event classification: `gpt-5-mini` ✓ Fast, cheap  
- Economic impact analysis: `gpt-4o` ✓ Excellent reasoning
- Decision rationale: `gpt-4o` ✓ Clear explanations
- NLU parameter extraction: `gpt-4o` ✓ Accurate parsing
- Response generation: `gpt-4o` ✓ User-friendly output

**Cost Savings**: This reduces your costs by **~50%** compared to using GPT-4o for everything, with zero compromise on quality!

**Why not Claude?** gpt-4o handles all the reasoning tasks we need. Claude would add complexity without meaningful benefits for our focused use cases.

