# Data Requirements for Currency Conversion Timing System

*Refined specifications based on actual user needs and decision-making requirements*

## Essential Data Requirements

### 1. Current Exchange Rates ⭐ CRITICAL
**Purpose**: Foundation for all conversion calculations
**What exactly needed**:
- Real-time spot rates for major currency pairs
- Bid/ask spread (when source provides it) for accurate timing/volatility measures
- Mid-market rates as baseline
**Currency pairs priority**: Major pairs (USD/EUR/GBP/JPY) first, expand to emerging markets later
**Update frequency**: 1-5 minute polling (realistic for free APIs)
**Future enhancement**: Streaming data (OANDA style) for real-time updates

### 2. Rate Trends & Volatility ⭐ CRITICAL
**Purpose**: Historical context for forecasts and "good timing" detection
**What exactly needed**:
- Minimum 30 days historical data, ideally 90+ days for robust signals
- Rolling technical indicators:
  - Moving averages (20-day, 50-day)
  - Bollinger bands for volatility channels
  - Realized volatility calculations
- Intraday candles (15min/1hour) if API supports - helps short-term advice
**Update frequency**: Daily historical updates, real-time for current trends

### 3. Major Economic Events ⭐ CRITICAL
**Purpose**: Event-driven volatility is the biggest FX mover
**What exactly needed**:
- Event dates AND expected vs actual values
  - Example: "US CPI expected 3.2%, actual 3.6%"
- Impact classification: Low/Medium/High priority tagging
- Key events focus:
  - Central bank meetings (Fed, ECB, BOE)
  - Major releases: GDP, inflation (CPI/PPI), employment
  - Interest rate decisions and policy statements
**Update frequency**: Weekly calendar sync, daily pulls ahead of major events

### 4. Currency Provider Costs ⭐ IMPORTANT
**Purpose**: Differentiate from "just another FX chart" - real conversion costs
**What exactly needed**:
- Rates from major providers: Wise, Revolut, banks, XE.com
- Normalized into "effective rate" (net received / amount sent)
- Transfer fees, spreads, and total cost breakdowns
- Quote validity periods (e.g., Wise locks for ~30 minutes)
**Update frequency**: On-demand (user requests) or periodic for "cheapest provider" alerts

### 5. Market-Moving News 🔶 NICE-TO-HAVE
**Purpose**: Explanatory power and recommendation justification
**What exactly needed**:
- FX-sensitive news feeds only:
  - Central bank communications
  - Geopolitical events affecting currencies
  - Major commodity price moves (oil, gold)
- Headline-level sentiment analysis ("hawkish Fed" → USD up)
- Focus on quality over quantity - avoid noise
**Update frequency**: Real-time streaming ideal, 5-15 minute lag acceptable via RSS/NewsAPI

### 6. Seasonal Patterns 🔶 NICE-TO-HAVE
**Purpose**: Background context and user education
**What exactly needed**:
- Historical seasonal trends by currency pair
- Examples:
  - USD demand spikes around year-end settlements
  - INR weakens in Q3 due to oil import cycles
- Treat as background model, not real-time decision factor
**Update frequency**: Annual analysis and model updates

## Data We DON'T Need (Avoid Over-Engineering)

❌ **Social media sentiment** - Too noisy for reliable signals
❌ **Complex technical indicators** (RSI, MACD) - Basic trends sufficient for user decisions  
❌ **Minute-by-minute tick data** - Daily data adequate for retail users
❌ **Minor economic indicators** - Focus only on market-moving events
❌ **Real-time order flow data** - Too complex and expensive for target users

## Implementation Priorities

### Phase 1: Foundation
1. Current exchange rates with reliable failover
2. 30-90 day historical rate data
3. Basic volatility calculations

### Phase 2: Intelligence  
1. Economic calendar integration
2. Provider cost comparison
3. Simple trend analysis

### Phase 3: Enhancement
1. News feed integration
2. Seasonal pattern analysis
3. Advanced notification system

## Success Criteria

- **Data freshness**: <5 minutes for rates, <24 hours for events
- **Coverage**: Top 10 currency pairs minimum
- **Reliability**: 99%+ uptime with graceful failover
- **Accuracy**: Cross-validation between multiple rate sources
- **Cost efficiency**: Stay within free/low-cost API tiers initially

*Last updated: 2025-08-28*