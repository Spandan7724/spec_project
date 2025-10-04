# Currency Assistant CLI

🤖 **Interactive currency conversion timing advisor** - Get AI-powered advice on when to convert currencies for optimal timing.

## Quick Start

### Installation
```bash
# Install dependencies
source .venv/bin/activate
uv pip install .

# Or use the test script directly
python test_cli.py --help
```

### Basic Usage

#### 1. Get instant advice
```bash
currency-assistant ask "Should I convert $1000 USD to EUR this week?"
```

#### 2. Start interactive chat
```bash
currency-assistant chat
```

#### 3. Check system status
```bash
currency-assistant status
```

#### 4. Configure preferences
```bash
# Show current configuration
currency-assistant config --show

# Set default risk tolerance
currency-assistant config --set "default_risk_tolerance=low"

# Set default currencies
currency-assistant config --set "default_base_currency=GBP"
currency-assistant config --set "default_quote_currency=USD"
```

## Features

### 🎯 **Natural Language Processing**
- Understands questions like:
  - "Should I convert $1000 USD to EUR this week?"
  - "What's the outlook for GBP/USD over the next 30 days?"
  - "I need to exchange 5000 JPY to EUR, what's your recommendation?"

### 💬 **Interactive Chat Mode**
- Persistent conversation history
- Session resumption support
- Context-aware follow-up questions
- Command help and status checking

### 📊 **Comprehensive Analysis**
- **Market Analysis**: Technical indicators, ML predictions, current rates
- **Economic Calendar**: High-impact events, risk assessment
- **Risk Evaluation**: Volatility analysis, VaR calculations
- **Actionable Recommendations**: Clear advice with confidence scores

### 🎨 **Beautiful Interface**
- Rich, colorful output with panels and tables
- Progress indicators for long operations
- Structured data presentation
- Chat-optimized formatting

## Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `ask` | Get single recommendation | `currency-assistant ask "Convert $1000 USD to EUR?"` |
| `chat` | Start interactive session | `currency-assistant chat` |
| `status` | Check system health | `currency-assistant status --verbose` |
| `config` | Manage settings | `currency-assistant config --show` |
| `demo` | Run showcase demo | `currency-assistant demo` |

## Chat Commands

When in chat mode, use these commands:
- `help` - Show available commands
- `status` - Check system status
- `history` - Show conversation history
- `clear` - Clear screen
- `export [filename]` - Export session to file
- `exit` or `quit` - End session

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `default_base_currency` | USD | Default base currency |
| `default_quote_currency` | EUR | Default quote currency |
| `default_amount` | 1000.0 | Default amount for analysis |
| `default_risk_tolerance` | moderate | Risk preference (low/moderate/high) |
| `default_timeframe_days` | 7 | Default analysis timeframe |
| `show_processing_time` | true | Show processing duration |
| `auto_save_sessions` | true | Auto-save chat sessions |

## Example Questions

### Basic Questions
- "Should I convert $1000 USD to EUR this week?"
- "Convert 5000 GBP to USD with low risk"
- "What's the outlook for EUR/JPY this month?"

### Advanced Questions
- "I have 2000 CAD, should I convert to USD now or wait 2 weeks?"
- "Compare converting $5000 to EUR vs GBP for the next 10 days"
- "What's the risk level for converting JPY to AUD over 30 days?"

## Sample Output

```
💡 Currency Advice for Analysis for USD/EUR conversion of 1000.0 USD

Recommendation: Wait
Confidence: Medium (65%)
Timeline: Reassess in ~1 day(s) after the imminent events conclude

Key Reasons:
• Market technicals suggest a bearish USD/EUR trend
• Machine learning 7d forecast predicts minimal upside
• High-impact economic events within 7 days could introduce volatility

⚠️ Warnings:
• Economic data releases could cause sudden market movements
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you're in the virtual environment
   ```bash
   source .venv/bin/activate
   ```

2. **API Key Issues**: Check your `.env` file for required API keys
   ```bash
   # Required keys
   ALPHA_VANTAGE_API_KEY=your-key
   EXCHANGE_RATE_HOST_API_KEY=your-key
   ```

3. **Long Processing Times**: The system analyzes multiple data sources
   - First run may be slower as models load
   - Subsequent queries are faster with caching

### Debug Mode
Enable debug logging for troubleshooting:
```bash
currency-assistant ask "question" --debug
```

## Architecture

The CLI integrates with your existing currency assistant backend:

```
CLI Interface → Service Layer → Agentic Workflow → Analysis Engines
     ↓              ↓              ↓                ↓
  Rich Display  Natural Language  LangGraph      Market/Economic/
  Chat Session   Question Parser  Orchestration   Risk/ML Analysis
```

## Next Steps

1. **FastAPI Backend**: Build REST API for web/mobile integration
2. **Web Frontend**: React/Vue.js dashboard
3. **Mobile App**: React Native for on-the-go advice
4. **Enhanced Features**: Portfolio optimization, historical tracking

## Support

For issues or questions:
- Check system status: `currency-assistant status`
- Enable debug mode for detailed logs
- Review configuration: `currency-assistant config --show`

---

**Enjoy your AI-powered currency advisor!** 💰🚀