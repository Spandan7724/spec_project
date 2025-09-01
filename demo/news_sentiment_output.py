#!/usr/bin/env python3
"""
Demo: Live News Sentiment Output for Multi-Agent System
Shows actual news data from the financial news scraper system
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.news.news_scraper import FinancialNewsScraper
from src.data_collection.news.news_models import NewsArticle

async def get_live_news_sentiment_output():
    """
    Get live news sentiment output from the actual news scraping system.
    This demonstrates what the FinancialNewsScraper actually returns.
    """
    
    print("Scraping live financial news...")
    
    try:
        # Initialize the news scraper
        async with FinancialNewsScraper() as scraper:
            # Get latest financial news
            print("Collecting latest financial news...")
            articles = await scraper.get_latest_news(hours_back=24)
            all_articles = articles if articles else []
        
        return all_articles
        
    except Exception as e:
        print(f"Error collecting live news: {e}")
        print("This is expected if news sources are unavailable or rate limited.")
        return []

def format_for_agents(articles: list[NewsArticle]) -> dict:
    """
    Format the news articles for consumption by the multi-agent system.
    This is the standardized format agents will receive.
    """
    
    if not articles:
        return {
            "data_type": "news_sentiment",
            "timestamp": datetime.now().isoformat(),
            "articles_found": 0,
            "error": "No articles collected"
        }
    
    # Analyze sentiment distribution
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        if article.sentiment:
            sentiment_counts[article.sentiment.value] += 1
    
    # Get recent high-impact articles
    recent_articles = [a for a in articles if a.is_recent]
    high_impact_articles = [a for a in articles if a.market_impact == "high"]
    
    # Currency-specific analysis
    currency_sentiment = {}
    for currency in ["USD", "EUR", "GBP"]:
        relevant_articles = [a for a in articles if currency in a.affected_currencies]
        if relevant_articles:
            positive = len([a for a in relevant_articles if a.is_bullish])
            negative = len([a for a in relevant_articles if a.is_bearish])
            total = len(relevant_articles)
            
            currency_sentiment[currency] = {
                "total_articles": total,
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": total - positive - negative,
                "sentiment_score": (positive - negative) / total if total > 0 else 0,
                "bias": "bullish" if positive > negative else "bearish" if negative > positive else "neutral"
            }
    
    agent_data = {
        "data_type": "news_sentiment",
        "timestamp": datetime.now().isoformat(),
        "collection_period_hours": 24,
        
        # Article summary
        "summary": {
            "total_articles": len(articles),
            "recent_articles": len(recent_articles),
            "high_impact_articles": len(high_impact_articles),
            "sentiment_distribution": sentiment_counts,
            "sources": list(set([a.source for a in articles])),
            "coverage_currencies": list(set([c for a in articles for c in a.affected_currencies]))
        },
        
        # Key articles for agent attention
        "priority_articles": [
            {
                "article_id": article.article_id,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "published_date": article.published_date.isoformat() if article.published_date else None,
                "affected_currencies": article.affected_currencies,
                "currency_pairs": article.currency_pairs,
                "sentiment": article.sentiment.value if article.sentiment else None,
                "sentiment_score": article.sentiment_score,
                "market_impact": article.market_impact,
                "category": article.category,
                "tags": article.tags,
                "summary": article.summary,
                "is_recent": article.is_recent,
                "age_hours": (datetime.utcnow() - article.published_date).total_seconds() / 3600 if article.published_date else None
            }
            for article in high_impact_articles[:10]  # Top 10 high impact articles
        ],
        
        # Currency-specific sentiment analysis
        "currency_sentiment": currency_sentiment,
        
        # Market themes and trends
        "market_themes": {
            "trending_topics": [],  # Would analyze article tags/categories
            "central_bank_focus": len([a for a in articles if "monetary_policy" in (a.tags or [])]),
            "economic_data_focus": len([a for a in articles if "economic_data" in (a.tags or [])]),
            "geopolitical_focus": len([a for a in articles if "geopolitical" in (a.tags or [])]),
            "dominant_narrative": "neutral"  # Could be derived from sentiment analysis
        },
        
        # Agent decision support
        "agent_context": {
            "overall_market_sentiment": "bullish" if sentiment_counts["positive"] > sentiment_counts["negative"] else "bearish" if sentiment_counts["negative"] > sentiment_counts["positive"] else "neutral",
            
            "sentiment_strength": abs(sentiment_counts["positive"] - sentiment_counts["negative"]) / max(len(articles), 1),
            
            "news_flow_intensity": "high" if len(articles) > 20 else "moderate" if len(articles) > 10 else "low",
            
            "breaking_news_alerts": len([a for a in recent_articles if (datetime.utcnow() - a.published_date).total_seconds() < 3600 if a.published_date]),
            
            "market_moving_potential": len(high_impact_articles) > 0,
            
            "recommended_monitoring": [
                f"{currency}: {data['bias']} bias ({data['total_articles']} articles)"
                for currency, data in currency_sentiment.items()
                if data['total_articles'] >= 3
            ],
            
            "risk_factors": [
                factor for factor, condition in [
                    ("high_negative_sentiment", sentiment_counts["negative"] > sentiment_counts["positive"] * 2),
                    ("breaking_news_present", len(recent_articles) > 5),
                    ("conflicting_signals", abs(sentiment_counts["positive"] - sentiment_counts["negative"]) < 2),
                    ("limited_coverage", len(articles) < 5)
                ] if condition
            ]
        }
    }
    
    return agent_data

async def main():
    """Generate and display live news sentiment output for agents"""
    
    print("=== Live News Sentiment Output Demo for Multi-Agent System ===\n")
    
    # Get live data from actual system
    articles = await get_live_news_sentiment_output()
    
    print(f"Collected {len(articles)} articles from news sources")
    
    if not articles:
        print("No articles collected. Check news source availability.")
        return
    
    print(f"Sources: {', '.join(set([a.source for a in articles]))}")
    print(f"Currencies covered: {', '.join(set([c for a in articles for c in a.affected_currencies]))}")
    print()
    
    # Format for agents
    agent_data = format_for_agents(articles)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Insights for Agent Decision Making ===")
    print(f"• Overall Market Sentiment: {agent_data['agent_context']['overall_market_sentiment']}")
    print(f"• News Flow Intensity: {agent_data['agent_context']['news_flow_intensity']}")
    print(f"• High Impact Articles: {agent_data['summary']['high_impact_articles']}")
    print(f"• Breaking News Alerts: {agent_data['agent_context']['breaking_news_alerts']}")
    print(f"• Market Moving Potential: {agent_data['agent_context']['market_moving_potential']}")
    
    if agent_data['agent_context']['recommended_monitoring']:
        print(f"• Currency Monitoring: {', '.join(agent_data['agent_context']['recommended_monitoring'])}")
    
    if agent_data['agent_context']['risk_factors']:
        print(f"• Risk Factors: {', '.join(agent_data['agent_context']['risk_factors'])}")

if __name__ == "__main__":
    asyncio.run(main())