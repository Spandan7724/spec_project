"""
Helper utilities for using different LLM models with different agents and tasks
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
    model_name: str,
    llm_manager: Optional[LLMManager] = None,
    tools: Optional[List[Dict[str, Any]]] = None
):
    """
    Execute a chat request with a specific model, regardless of default configuration.
    
    This is the recommended way to use different models for different tasks within an agent.
    
    Args:
        messages: List of chat messages
        model_name: Specific model to use (e.g., "gpt-5-mini", "claude-3.5-sonnet", "gpt-4o-2024-11-20")
        llm_manager: Optional LLMManager instance (will create one if not provided)
        tools: Optional function calling tools
    
    Returns:
        ChatResponse from the specified model
    
    Examples:
        # In your agent, for news sentiment analysis with gpt-5-mini
        from src.llm.agent_helpers import chat_with_model
        
        sentiment_messages = [
            {"role": "system", "content": "Analyze news sentiment."},
            {"role": "user", "content": f"News: {news_text}"}
        ]
        response = await chat_with_model(sentiment_messages, "gpt-5-mini", self.llm_manager)
        
        # For complex reasoning with Claude
        analysis_messages = [...]
        response = await chat_with_model(analysis_messages, "claude-3.5-sonnet", self.llm_manager)
    """
    if llm_manager is None:
        llm_manager = LLMManager()
    
    # Find provider for this model
    provider_name = get_provider_for_model(model_name)
    
    if provider_name is None:
        logger.warning(f"No provider found for model '{model_name}', using default provider")
        return await llm_manager.chat(messages, tools=tools)
    
    logger.debug(f"Using provider '{provider_name}' for model '{model_name}'")
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


# Optional: Helper for task-based model recommendations
TASK_MODEL_RECOMMENDATIONS = {
    "sentiment_analysis": "gpt-5-mini",           # Fast, cheap for sentiment
    "classification": "gpt-5-mini",               # Fast for categorization
    "summarization": "gpt-4o-2024-11-20",         # Balanced for summaries
    "reasoning": "claude-3.5-sonnet",             # Best for complex reasoning
    "data_extraction": "gpt-5-mini",              # Fast for structured data
    "creative_writing": "gpt-4o-2024-11-20",      # Balanced for creative tasks
    "code_generation": "gpt-4o-2024-11-20",       # Good for code
    "analysis": "claude-3.5-sonnet",              # Deep analysis
}


def get_recommended_model_for_task(task_type: str) -> str:
    """
    Get the recommended model for a specific task type.
    
    This is just a helper/guide - you can always use any model you want.
    
    Args:
        task_type: Type of task (e.g., "sentiment_analysis", "reasoning", etc.)
    
    Returns:
        Recommended model name
    
    Example:
        model = get_recommended_model_for_task("sentiment_analysis")  # Returns "gpt-5-mini"
        response = await chat_with_model(messages, model)
    """
    return TASK_MODEL_RECOMMENDATIONS.get(task_type, "gpt-4o-2024-11-20")

