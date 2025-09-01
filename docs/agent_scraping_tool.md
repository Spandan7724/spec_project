# Agent Web Scraping Tool

## Overview

Conversational gap-filling web scraping tool that enables AI agents to fetch up-to-date information during user conversations. Designed for scenarios where existing APIs and cached data are insufficient.

## When Agents Should Use This Tool

### ✅ **Automatic Triggers**
- **Time-sensitive queries**: "latest", "today", "current", "just announced"
- **Provider-specific questions**: User mentions Wise, Remitly, XE, Revolut, etc.
- **Knowledge gaps**: "I heard that...", "But what about...", "Did you know..."
- **Recent events**: "this week", "recently", "breaking news"

### ❌ **Don't Use For**
- Educational questions: "How do exchange rates work?"
- General explanations: "What affects currency values?"
- Historical analysis: Available through existing APIs
- Basic conversions: Use existing rate APIs

## Agent Interface Methods

```python
from tools.agent_interface import AgentScrapingInterface

# Check if scraping is recommended
decision = await AgentScrapingInterface.should_i_scrape(
    user_query="What about Wise's latest fees?",
    conversation_context="Previous USD/EUR conversion discussion"
)

# Get current information (main method)
info = await AgentScrapingInterface.get_current_info(
    user_query="Did the ECB announce anything today?",
    conversation_context="Currency timing discussion"
)

# Specific use cases
provider_info = await AgentScrapingInterface.check_provider_updates("Wise")
economic_news = await AgentScrapingInterface.check_economic_events("Fed policy")
verification = await AgentScrapingInterface.verify_claim("Remitly changed fees")
```

## Response Format

```python
{
    'success': True,
    'results_found': 2,
    'sources_checked': 2,
    'data': {
        'currencies_mentioned': ['USD', 'EUR'],
        'conversions_found': [('1000', 'USD', '855.86', 'EUR')],
        'content_type': 'exchange_rates'
    },
    'citations': ['Source: xe.com (accessed 2025-08-30 15:23)'],
    'errors': []
}
```

## Intelligent Caching

**Cache TTL by Content Type:**
- Exchange rates: 5 minutes
- Economic news: 30 minutes
- Provider policies: 24 hours
- Regulatory changes: 6 hours
- General info: 1 hour

**Cache Bypass Triggers:**
- Query contains: "latest", "today", "current", "now", "breaking"
- User explicitly requests fresh data
- Time-sensitive conversation context

## Example Conversation Flow

```
User: "Convert $1000 USD to EUR"
Agent: [Uses existing APIs for comprehensive plan]

User: "But what about Wise's new fee structure?"
Agent: [Tool triggers automatically]
       → should_i_scrape() returns True
       → get_current_info() scrapes Wise pages
       → Agent: "Based on current Wise information..."

User: "Did the ECB announce anything today?"  
Agent: [Tool triggers due to time-sensitivity]
       → Scrapes ECB/financial news
       → Agent: "Yes, according to today's announcements..."

User: "How do exchange rates work?"
Agent: [No scraping - uses existing knowledge]
```

## Error Handling

- **Network failures**: Graceful degradation with error messages
- **Protected sites**: Returns appropriate error without breaking flow
- **Partial failures**: Continues with successful sources
- **Rate limiting**: Built-in retry logic with exponential backoff

## Usage Notes

1. **Conservative by design**: Only scrapes when clearly beneficial
2. **Caching first**: Always checks cache before new requests
3. **Multiple sources**: Can scrape multiple URLs per query
4. **Context aware**: Uses conversation history for better decisions
5. **Citation support**: Tracks sources for transparency

## Integration with Currency System

This tool complements existing data sources:
- **APIs**: FRED, ECB, Yahoo Finance (primary data)
- **Scraping Tool**: Gap-filling for recent events, provider updates
- **Cached Data**: Historical analysis, technical indicators

The tool does not replace existing APIs but fills gaps when APIs lack current information or specific provider details.