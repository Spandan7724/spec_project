"""
Helper utilities for using different LLM models with different agents and tasks.

Provider-Agnostic Model Selection Strategy
===========================================
This codebase uses a flexible provider system where each provider has two variants:
- **{provider}_main**: For complex tasks requiring reasoning (e.g., copilot_main, openai_main)
- **{provider}_fast**: For simple tasks like classification (e.g., copilot_fast, openai_fast)

Benefits:
- Switch providers by changing one config value (llm.default_provider)
- No code changes needed when switching from Copilot → OpenAI → Claude
- Cost optimization: ~50% savings by using fast models for simple tasks

Provider Examples:
==================
Copilot: copilot_main (gpt-4o) / copilot_fast (gpt-5-mini)
OpenAI:  openai_main (gpt-4o) / openai_fast (gpt-4o-mini)
Claude:  claude_main (claude-sonnet-4) / claude_fast (claude-3.5-sonnet)

Usage Patterns
==============
1. **Get provider for a task**: Use get_provider_for_task("classification") → "{provider}_fast"
2. **Explicitly request main/fast**: Use get_main_provider() or get_fast_provider()
3. **Chat with task-appropriate model**: Use chat_with_model_for_task(messages, "nlu", llm)

Examples
========
    # Simple task - news classification (uses fast model)
    provider = get_provider_for_task("classification")  # → "copilot_fast"
    response = await chat_with_model(messages, provider, llm_manager)

    # Complex task - NLU extraction (uses main model)
    provider = get_provider_for_task("nlu")  # → "copilot_main"
    response = await chat_with_model(messages, provider, llm_manager, tools=tools)

    # Or use the convenience function
    response = await chat_with_model_for_task(messages, "classification", llm_manager)
"""

import logging
from typing import Optional, List, Dict, Any
from .manager import LLMManager
from .config import load_config

logger = logging.getLogger(__name__)


def get_provider_for_model(model_name: str) -> Optional[str]:
    """
    Find the provider name for a given model.

    Args:
        model_name: Model identifier (e.g., "gpt-5-mini", "claude-3.5-sonnet")

    Returns:
        Provider name if found, None otherwise
    """
    config = load_config()
    for prov_name, prov_config in config.providers.items():
        if prov_config.model == model_name and prov_config.enabled:
            return prov_name
    return None


async def chat_with_model(
    messages: List[Dict[str, Any]],
    provider_or_model: str,
    llm_manager: Optional[LLMManager] = None,
    tools: Optional[List[Dict[str, Any]]] = None
):
    """
    Execute a chat request with a specific provider or model.

    NOTE: Now accepts provider names (e.g., "copilot_fast") OR model names (e.g., "gpt-5-mini")
    for backward compatibility. Prefer using provider names with the new system.

    Args:
        messages: List of chat messages
        provider_or_model: Provider name (e.g., "copilot_fast", "openai_main")
                          OR model name (e.g., "gpt-5-mini", "claude-3.5-sonnet")
        llm_manager: Optional LLMManager instance (will create one if not provided)
        tools: Optional function calling tools

    Returns:
        ChatResponse from the specified provider/model

    Examples:
        # NEW: Using provider name (recommended)
        provider = get_provider_for_task("classification")  # → "copilot_fast"
        response = await chat_with_model(messages, provider, llm_manager)

        # Legacy: Using model name (still works)
        response = await chat_with_model(messages, "gpt-5-mini", llm_manager)
    """
    if llm_manager is None:
        llm_manager = LLMManager()

    # Check if it's already a valid provider name
    config = load_config()
    if provider_or_model in config.providers:
        provider_name = provider_or_model
        logger.debug(f"Using provider '{provider_name}'")
    else:
        # Try to find provider for this model name (backward compatibility)
        provider_name = get_provider_for_model(provider_or_model)

        if provider_name is None:
            logger.warning(f"No provider found for '{provider_or_model}', using default provider")
            return await llm_manager.chat(messages, tools=tools)

        logger.debug(f"Using provider '{provider_name}' for model '{provider_or_model}'")

    return await llm_manager.chat(messages, tools=tools, provider_name=provider_name)


def create_agent_llm(model_name: Optional[str] = None, provider_name: Optional[str] = None) -> LLMManager:
    """
    Create an LLMManager configured for a specific agent.
    
    Note: For task-specific model selection within an agent, use chat_with_model() instead.
    
    Args:
        model_name: Specific model to use (e.g., "gpt-5-mini", "claude-3.5-sonnet")
                   If provided, will find the matching provider from config
        provider_name: Specific provider to use as default (e.g., "copilot_mini", "copilot_claude")
                      Takes precedence over model_name
    
    Returns:
        LLMManager instance configured with the specified model/provider as default
    """
    config = load_config()
    
    # If provider_name is specified, use it directly
    if provider_name:
        if provider_name in config.providers:
            config.default_provider = provider_name
            logger.info(f"Using provider '{provider_name}' with model '{config.providers[provider_name].model}'")
        else:
            logger.warning(f"Provider '{provider_name}' not found in config, using default")
    
    # If model_name is specified, find matching provider
    elif model_name:
        matching_provider = get_provider_for_model(model_name)
        
        if matching_provider:
            config.default_provider = matching_provider
            logger.info(f"Found provider '{matching_provider}' for model '{model_name}'")
        else:
            logger.warning(f"No enabled provider found for model '{model_name}', using default")
    
    return LLMManager(config)


# Task complexity mapping: determines whether to use _main or _fast provider
# Strategy: Use _fast for simple/quick tasks, _main for complex reasoning
# This provides ~50% cost savings vs using _main everywhere
TASK_COMPLEXITY = {
    # Simple tasks → use {provider}_fast
    "sentiment_analysis": "fast",     # News sentiment classification
    "classification": "fast",         # Article categorization
    "data_extraction": "fast",        # Structured data extraction from text

    # Complex tasks → use {provider}_main
    "summarization": "main",          # Narrative generation, summaries
    "reasoning": "main",              # Complex reasoning tasks
    "nlu": "main",                    # Natural language understanding with function calling
    "conversation": "main",           # User-facing Q&A
    "creative_writing": "main",       # Response generation
    "code_generation": "main",        # Code tasks
    "analysis": "main",               # Deep analysis, decision synthesis
}


def get_main_provider() -> str:
    """
    Get the main (complex task) provider name for the current default provider.

    Returns:
        Provider name with _main suffix (e.g., "copilot_main", "openai_main")

    Example:
        provider = get_main_provider()  # If default is "copilot" → "copilot_main"
    """
    config = load_config()
    base_provider = config.default_provider
    return f"{base_provider}_main"


def get_fast_provider() -> str:
    """
    Get the fast (simple task) provider name for the current default provider.

    Returns:
        Provider name with _fast suffix (e.g., "copilot_fast", "openai_fast")

    Example:
        provider = get_fast_provider()  # If default is "copilot" → "copilot_fast"
    """
    config = load_config()
    base_provider = config.default_provider
    return f"{base_provider}_fast"


def get_provider_for_task(task_type: str) -> str:
    """
    Get the appropriate provider (main or fast) for a specific task type.

    This is the recommended way to get a provider for task-specific LLM calls.

    Args:
        task_type: Type of task (e.g., "classification", "nlu", "reasoning")

    Returns:
        Provider name (e.g., "copilot_fast" for simple tasks, "copilot_main" for complex)

    Example:
        provider = get_provider_for_task("classification")  # → "copilot_fast"
        response = await chat_with_model(messages, provider, llm_manager)
    """
    complexity = TASK_COMPLEXITY.get(task_type, "main")  # Default to main if unknown

    if complexity == "fast":
        return get_fast_provider()
    else:
        return get_main_provider()


async def chat_with_model_for_task(
    messages: List[Dict[str, Any]],
    task_type: str,
    llm_manager: Optional[LLMManager] = None,
    tools: Optional[List[Dict[str, Any]]] = None
):
    """
    Convenience function: chat with the appropriate model for a given task type.

    Automatically selects the right provider variant (_main or _fast) based on task complexity.

    Args:
        messages: List of chat messages
        task_type: Type of task (e.g., "classification", "nlu", "summarization")
        llm_manager: Optional LLMManager instance
        tools: Optional function calling tools

    Returns:
        ChatResponse from the appropriate model

    Example:
        # Automatically uses copilot_fast for classification
        response = await chat_with_model_for_task(messages, "classification", llm_manager)

        # Automatically uses copilot_main for NLU
        response = await chat_with_model_for_task(messages, "nlu", llm_manager, tools=tools)
    """
    provider = get_provider_for_task(task_type)
    if llm_manager is None:
        llm_manager = LLMManager()

    logger.debug(f"Using provider '{provider}' for task type '{task_type}'")
    return await llm_manager.chat(messages, tools=tools, provider_name=provider)


# Legacy function for backward compatibility
def get_recommended_model_for_task(task_type: str) -> str:
    """
    DEPRECATED: Use get_provider_for_task() instead.

    Legacy function that returns provider name instead of model name.
    Kept for backward compatibility but will delegate to new system.

    Args:
        task_type: Type of task

    Returns:
        Provider name (e.g., "copilot_fast", "copilot_main")
    """
    logger.warning("get_recommended_model_for_task() is deprecated. Use get_provider_for_task() instead.")
    return get_provider_for_task(task_type)

