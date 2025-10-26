"""Test LLM API latency for batch vs individual classification.

This script compares:
1. Individual calls (parallel) - 4 separate LLM calls
2. Batch call (single) - 1 LLM call with 4 articles

Run: uv run python test_llm_timing.py
"""

import asyncio
import time
import json
from typing import Dict

from src.llm.manager import LLMManager
from src.llm.agent_helpers import chat_with_model, get_recommended_model_for_task


# Sample articles
SAMPLE_ARTICLES = [
    {
        "title": "ECB announces interest rate decision amid inflation concerns",
        "snippet": "The European Central Bank announced its latest interest rate decision today, with markets watching closely for signals on future monetary policy.",
        "url": "https://example.com/article1",
        "source": "reuters.com"
    },
    {
        "title": "USD strengthens as Fed signals hawkish stance on rates",
        "snippet": "The US dollar gained ground against major currencies as Federal Reserve officials suggested further rate hikes may be necessary to combat inflation.",
        "url": "https://example.com/article2",
        "source": "bloomberg.com"
    },
    {
        "title": "Euro area GDP grows faster than expected in Q3",
        "snippet": "Preliminary data shows the Eurozone economy expanded at a faster pace than forecasters had predicted, driven by strong consumption and exports.",
        "url": "https://example.com/article3",
        "source": "ft.com"
    },
    {
        "title": "Currency markets volatile amid geopolitical tensions",
        "snippet": "Foreign exchange markets saw increased volatility as investors weighed geopolitical risks against economic fundamentals.",
        "url": "https://example.com/article4",
        "source": "wsj.com"
    }
]

CURRENCIES = ["USD", "EUR"]


async def classify_individual(article: Dict, llm_manager: LLMManager) -> Dict:
    """Classify one article individually."""
    prompt = f"""Analyze this financial news article.

Title: {article['title']}
Snippet: {article['snippet']}
Currencies: {', '.join(CURRENCIES)}

Return JSON with:
relevance: per-currency 0.0-1.0
sentiment: per-currency -1.0..+1.0
quality_flags: clickbait, rumor_speculative, non_econ
"""

    messages = [
        {"role": "system", "content": "Return ONLY JSON."},
        {"role": "user", "content": prompt},
    ]
    model = get_recommended_model_for_task("classification")
    resp = await chat_with_model(messages, model, llm_manager)
    content = resp.content.strip()
    
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    
    try:
        return json.loads(content)
    except Exception:
        return {
            "relevance": {c: 0.0 for c in CURRENCIES},
            "sentiment": {c: 0.0 for c in CURRENCIES},
            "quality_flags": {"clickbait": False, "rumor_speculative": False, "non_econ": False}
        }


async def test_individual_calls():
    """Test 4 individual parallel calls."""
    print("\n" + "="*60)
    print("TEST 1: Individual Parallel Calls (4 separate LLM calls)")
    print("="*60)
    
    llm_manager = LLMManager()
    
    start = time.time()
    
    # Create tasks for all 4 articles
    tasks = [classify_individual(article, llm_manager) for article in SAMPLE_ARTICLES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end = time.time()
    elapsed = end - start
    
    print(f"\n⏱️  Time: {elapsed:.2f} seconds")
    print(f"📊 Successful: {sum(1 for r in results if not isinstance(r, Exception))}/4")
    print(f"❌ Failed: {sum(1 for r in results if isinstance(r, Exception))}/4")
    
    return elapsed


async def test_batch_call():
    """Test 1 batch call with all articles."""
    print("\n" + "="*60)
    print("TEST 2: Batch Call (1 LLM call with 4 articles)")
    print("="*60)
    
    llm_manager = LLMManager()
    
    # Build batch prompt
    articles_text = ""
    for i, article in enumerate(SAMPLE_ARTICLES, 1):
        articles_text += f"\nArticle {i}:\n"
        articles_text += f"Title: {article['title']}\n"
        articles_text += f"Snippet: {article['snippet']}\n"
        articles_text += f"Source: {article['source']}\n"
    
    prompt = f"""Analyze these 4 financial news articles for currency relevance and sentiment.

{articles_text}

Currencies: {', '.join(CURRENCIES)}

Return a JSON array with EXACTLY 4 objects, one for each article in order.
Each object must have:
- relevance: per-currency 0.0-1.0
- sentiment: per-currency -1.0..+1.0  
- quality_flags: clickbait, rumor_speculative, non_econ

Example format:
[
  {{"relevance": {{"USD": 0.8, "EUR": 0.7}}, "sentiment": {{"USD": 0.3, "EUR": -0.2}}, "quality_flags": {{"clickbait": false, "rumor_speculative": false, "non_econ": false}}}},
  ...
]
"""

    messages = [
        {"role": "system", "content": "Return ONLY a valid JSON array. No explanations."},
        {"role": "user", "content": prompt},
    ]
    
    start = time.time()
    
    model = get_recommended_model_for_task("classification")
    resp = await chat_with_model(messages, model, llm_manager)
    content = resp.content.strip()
    
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    
    try:
        data_array = json.loads(content)
        if not isinstance(data_array, list):
            raise ValueError("Not a JSON array")
        print(f"\n✅ Successfully parsed {len(data_array)} results")
    except Exception as e:
        print(f"\n❌ JSON parse failed: {e}")
        data_array = []
    
    end = time.time()
    elapsed = end - start
    
    print(f"⏱️  Time: {elapsed:.2f} seconds")
    print(f"📊 Results: {len(data_array)}/4 expected")
    
    return elapsed


async def main():
    print("\n🔬 LLM API Latency Comparison Test")
    print("="*60)
    
    individual_time = await test_individual_calls()
    batch_time = await test_batch_call()
    
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"Individual Calls: {individual_time:.2f}s")
    print(f"Batch Call:        {batch_time:.2f}s")
    print(f"Difference:        {abs(individual_time - batch_time):.2f}s")
    
    if individual_time < batch_time:
        percentage = ((batch_time - individual_time) / individual_time) * 100
        print(f"\n✅ Individual calls are {percentage:.1f}% FASTER")
        print("💡 Conclusion: Parallel individual calls are better for this use case")
    else:
        percentage = ((individual_time - batch_time) / individual_time) * 100
        print(f"\n✅ Batch call is {percentage:.1f}% FASTER")
        print("💡 Conclusion: Batch processing is better for this use case")


if __name__ == "__main__":
    asyncio.run(main())
