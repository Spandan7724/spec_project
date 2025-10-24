#!/usr/bin/env python3
"""
Practical Example: Using gpt-5-mini for news sentiment analysis

This shows how to use different models for different tasks within your agent.
"""

import asyncio
import logging
from src.llm.manager import LLMManager
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def analyze_news_sentiment_simple(news_text: str) -> dict:
    """
    Simple example: Directly specify the model
    """
    messages = [
        {
            "role": "system",
            "content": "You are a financial news sentiment analyzer. Respond with JSON containing 'sentiment' (positive/negative/neutral) and 'confidence' (0-1)."
        },
        {
            "role": "user",
            "content": f"Analyze sentiment: {news_text}"
        }
    ]
    
    # Directly use gpt-5-mini for fast sentiment analysis
    response = await chat_with_model(messages, "gpt-5-mini")
    
    return {
        "sentiment": response.content,
        "model": response.model,
        "tokens": response.usage.get("total_tokens") if response.usage else None
    }


async def analyze_news_sentiment_recommended(news_text: str) -> dict:
    """
    Using task recommendations helper
    """
    messages = [
        {
            "role": "system",
            "content": "You are a financial news sentiment analyzer. Respond with JSON containing 'sentiment' (positive/negative/neutral) and 'confidence' (0-1)."
        },
        {
            "role": "user",
            "content": f"Analyze sentiment: {news_text}"
        }
    ]
    
    # Get recommended model for sentiment analysis (returns gpt-5-mini)
    model = get_recommended_model_for_task("sentiment_analysis")
    response = await chat_with_model(messages, model)
    
    return {
        "sentiment": response.content,
        "model": response.model,
        "tokens": response.usage.get("total_tokens") if response.usage else None
    }


class EconomicAgentExample:
    """
    Example agent that uses different models for different tasks
    """
    
    def __init__(self):
        self.llm_manager = LLMManager()
    
    async def analyze_news_sentiment(self, news_text: str) -> dict:
        """Quick sentiment with gpt-5-mini"""
        messages = [
            {
                "role": "system",
                "content": "Analyze financial news sentiment. Return JSON with 'sentiment' and 'score'."
            },
            {
                "role": "user",
                "content": news_text
            }
        ]
        
        # Use gpt-5-mini for this fast task
        response = await chat_with_model(messages, "gpt-5-mini", self.llm_manager)
        return {"result": response.content, "model": response.model}
    
    async def deep_economic_analysis(self, data: str) -> dict:
        """Complex analysis with Claude"""
        messages = [
            {
                "role": "system",
                "content": "You are an expert economist. Provide detailed analysis."
            },
            {
                "role": "user",
                "content": data
            }
        ]
        
        # Use Claude for complex reasoning
        response = await chat_with_model(messages, "claude-3.5-sonnet", self.llm_manager)
        return {"result": response.content, "model": response.model}
    
    async def summarize_data(self, data: str) -> dict:
        """General summary with gpt-4o"""
        messages = [
            {
                "role": "system",
                "content": "Provide a concise summary."
            },
            {
                "role": "user",
                "content": data
            }
        ]
        
        # Use gpt-4o for balanced tasks
        response = await chat_with_model(messages, "gpt-4o-2024-11-20", self.llm_manager)
        return {"result": response.content, "model": response.model}


async def main():
    print("=" * 70)
    print("Practical Example: News Sentiment Analysis with gpt-5-mini")
    print("=" * 70)
    
    # Example news articles
    news_items = [
        "Federal Reserve signals potential interest rate cuts, market rallies",
        "European Central Bank maintains hawkish stance amid inflation concerns",
        "USD strengthens as jobless claims fall below expectations"
    ]
    
    print("\n📰 Method 1: Direct Model Selection")
    print("-" * 70)
    for news in news_items:
        result = await analyze_news_sentiment_simple(news)
        print(f"News: {news[:50]}...")
        print(f"  Model: {result['model']}")
        print(f"  Sentiment: {result['sentiment'][:100]}...")
        print()
    
    print("\n📰 Method 2: Using Task Recommendations")
    print("-" * 70)
    result = await analyze_news_sentiment_recommended(news_items[0])
    print(f"News: {news_items[0]}")
    print(f"  Model: {result['model']}")
    print(f"  Sentiment: {result['sentiment'][:150]}...")
    
    print("\n📊 Method 3: Within an Agent Class")
    print("-" * 70)
    agent = EconomicAgentExample()
    
    # Fast sentiment with gpt-5-mini
    print("Task: News Sentiment (using gpt-5-mini)")
    result = await agent.analyze_news_sentiment(news_items[1])
    print(f"  Model: {result['model']}")
    print(f"  Result: {result['result'][:100]}...")
    print()
    
    # Complex analysis with Claude
    print("Task: Deep Analysis (using claude-3.5-sonnet)")
    result = await agent.deep_economic_analysis("GDP growth slowing, inflation persistent")
    print(f"  Model: {result['model']}")
    print(f"  Result: {result['result'][:100]}...")
    print()
    
    # Summary with gpt-4o
    print("Task: Summary (using gpt-4o)")
    result = await agent.summarize_data("Market volatility increased today")
    print(f"  Model: {result['model']}")
    print(f"  Result: {result['result'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\nKey Points:")
    print("  • Use chat_with_model(messages, 'gpt-5-mini') for fast tasks")
    print("  • Use chat_with_model(messages, 'claude-3.5-sonnet') for reasoning")
    print("  • Use chat_with_model(messages, 'gpt-4o-2024-11-20') for balanced tasks")
    print("  • Mix models within the same agent based on the task!")


if __name__ == "__main__":
    asyncio.run(main())

