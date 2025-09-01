#!/usr/bin/env python3
"""
Demo: Technical Indicators Output for Multi-Agent System
Shows the data structure that the technical analysis component sends to agents
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.analysis.technical_indicators import TechnicalIndicators, TechnicalIndicatorEngine

async def get_live_technical_indicators_output() -> TechnicalIndicators:
    """
    Get REAL technical indicators from the actual data ingestion system.
    No more hardcoded values - this uses your real data collection system!
    """
    
    print("Loading technical indicators calculator...")
    
    try:
        # Initialize the real technical indicators calculator
        calculator = TechnicalIndicatorEngine()
        
        print("Calculating real technical indicators for USD/EUR...")
        
        # Get actual technical indicators for USD/EUR
        indicators = await calculator.calculate_indicators(
            currency_pair="USD/EUR",
            force_refresh=True  # Get fresh data
        )
        
        if indicators is None:
            raise ValueError("Failed to calculate technical indicators - check data availability")
            
        print(f"✅ Successfully calculated REAL technical indicators for {indicators.currency_pair}")
        print(f"Data date: {indicators.data_date}")
        return indicators
        
    except Exception as e:
        print(f"❌ Error getting real technical indicators: {e}")
        print("This might happen if:")
        print("   - Historical data is not available") 
        print("   - Data providers are not working")
        print("   - Network connectivity issues")
        raise e

def format_for_agents(indicators: TechnicalIndicators) -> dict:
    """
    Format the REAL technical indicators for consumption by the multi-agent system.
    Uses only the actual fields available in the TechnicalIndicators dataclass.
    """
    
    # Get current price from the latest data (or estimate)
    current_price = getattr(indicators, 'current_price', None)
    if current_price is None:
        # Estimate from middle of Bollinger Bands if available
        current_price = indicators.bb_middle or 0.85  # Fallback
    
    agent_data = {
        "data_type": "technical_analysis",
        "timestamp": indicators.data_date.isoformat(),
        "currency_pair": indicators.currency_pair,
        "current_price": current_price,
        "data_quality": 1.0,  # Real data from your system
        
        # Moving Average Analysis (using actual fields)
        "moving_averages": {
            "sma_20": indicators.sma_20,
            "sma_50": indicators.sma_50,
            "ema_12": indicators.ema_12,
            "ema_26": indicators.ema_26,
            "signals": {
                "sma_trend": "bullish" if (indicators.sma_20 and indicators.sma_50 and indicators.sma_20 > indicators.sma_50) else "bearish",
                "ema_trend": "bullish" if (indicators.ema_12 and indicators.ema_26 and indicators.ema_12 > indicators.ema_26) else "bearish"
            }
        },
        
        # Bollinger Bands Analysis (using actual fields)
        "bollinger_bands": {
            "upper": indicators.bb_upper,
            "middle": indicators.bb_middle,
            "lower": indicators.bb_lower,
            "width": indicators.bb_width,
            "position": indicators.bb_position,
            "signals": {
                "squeeze": "yes" if (indicators.bb_width and indicators.bb_width < 0.005) else "no",
                "position_signal": "upper" if (indicators.bb_position and indicators.bb_position > 0.7) else "lower" if (indicators.bb_position and indicators.bb_position < 0.3) else "middle"
            }
        },
        
        # RSI Analysis (using actual field rsi_14)
        "rsi": {
            "value": indicators.rsi_14,
            "signals": {
                "overbought": indicators.rsi_14 > 70 if indicators.rsi_14 else False,
                "oversold": indicators.rsi_14 < 30 if indicators.rsi_14 else False,
                "neutral": 30 <= (indicators.rsi_14 or 50) <= 70
            }
        },
        
        # MACD Analysis (using actual fields)
        "macd": {
            "macd_line": indicators.macd,
            "signal_line": indicators.macd_signal,
            "histogram": indicators.macd_histogram,
            "signals": {
                "bullish_crossover": (indicators.macd and indicators.macd_signal and indicators.macd > indicators.macd_signal),
                "momentum": "bullish" if (indicators.macd_histogram and indicators.macd_histogram > 0) else "bearish"
            }
        },
        
        # Volatility Analysis (using actual fields)
        "volatility": {
            "atr_14": indicators.atr_14,
            "realized_volatility": indicators.realized_volatility,
            "volatility_regime": indicators.volatility_regime,
            "bb_width": indicators.bb_width
        },
        
        # Support and Resistance (using actual fields)
        "levels": {
            "support": indicators.support_level,
            "resistance": indicators.resistance_level
        },
        
        # Overall Trend Analysis (using actual fields)
        "trend": {
            "direction": indicators.trend_direction,
            "strength": indicators.trend_strength
        },
        
        # Agent Decision Support - simplified to use only real fields
        "agent_context": {
            "overall_signal": _determine_overall_signal(indicators),
            "signal_strength": "strong" if (indicators.trend_strength and indicators.trend_strength > 0.7) else "moderate",
            "recommended_action": _determine_recommended_action(indicators),
            "confidence_score": indicators.trend_strength or 0.5,
            "is_bullish": indicators.is_bullish,
            "key_insights": _generate_key_insights(indicators)
        }
    }
    
    return agent_data


def _determine_overall_signal(indicators: TechnicalIndicators) -> str:
    """Determine overall signal from real indicator values"""
    bullish_count = 0
    total_signals = 0
    
    # Check trend direction
    if indicators.trend_direction:
        total_signals += 1
        if indicators.trend_direction == "up":
            bullish_count += 1
    
    # Check MACD
    if indicators.macd and indicators.macd_signal:
        total_signals += 1
        if indicators.macd > indicators.macd_signal:
            bullish_count += 1
    
    # Check RSI
    if indicators.rsi_14:
        total_signals += 1
        if 30 < indicators.rsi_14 < 70:  # Not overbought/oversold
            bullish_count += 1
    
    if total_signals == 0:
        return "neutral"
    
    bullish_ratio = bullish_count / total_signals
    if bullish_ratio >= 0.7:
        return "bullish"
    elif bullish_ratio <= 0.3:
        return "bearish"
    else:
        return "neutral"


def _determine_recommended_action(indicators: TechnicalIndicators) -> str:
    """Determine recommended action from real indicators"""
    if indicators.is_bullish:
        return "buy"
    elif indicators.trend_direction == "down":
        return "sell" 
    else:
        return "hold"


def _generate_key_insights(indicators: TechnicalIndicators) -> list:
    """Generate key insights from real indicators"""
    insights = []
    
    if indicators.trend_direction:
        insights.append(f"Trend direction: {indicators.trend_direction}")
    
    if indicators.rsi_14:
        if indicators.rsi_14 > 70:
            insights.append("RSI overbought")
        elif indicators.rsi_14 < 30:
            insights.append("RSI oversold")
        else:
            insights.append("RSI in neutral zone")
    
    if indicators.volatility_regime:
        insights.append(f"Volatility: {indicators.volatility_regime}")
    
    return insights

async def main():
    """Generate and display LIVE technical indicators output for agents"""
    
    print("=== LIVE Technical Indicators Output Demo for Multi-Agent System ===\n")
    
    # Get REAL technical indicators from your data ingestion system
    try:
        indicators = await get_live_technical_indicators_output()
        
        print("Raw Technical Indicators (REAL DATA):")
        print(f"Currency Pair: {indicators.currency_pair}")
        print(f"Data Date: {indicators.data_date}")
        
        # Display available indicators
        if indicators.rsi_14 is not None:
            print(f"RSI (14): {indicators.rsi_14:.1f}")
        if indicators.macd is not None and indicators.macd_signal is not None:
            print(f"MACD: {indicators.macd:.6f}, Signal: {indicators.macd_signal:.6f}")
        if indicators.sma_20 is not None and indicators.sma_50 is not None:
            print(f"SMA 20: {indicators.sma_20:.4f}, SMA 50: {indicators.sma_50:.4f}")
        if indicators.bb_upper is not None:
            print(f"Bollinger Bands: Upper={indicators.bb_upper:.4f}, Lower={indicators.bb_lower:.4f}")
        if indicators.trend_direction is not None:
            print(f"Trend: {indicators.trend_direction} (strength: {indicators.trend_strength:.1%})" if indicators.trend_strength else f"Trend: {indicators.trend_direction}")
        print()
        
    except Exception as e:
        print(f"Could not get live technical indicators: {e}")
        return
    
    # Format for agents
    agent_data = format_for_agents(indicators)
    
    print("=== Formatted Output for Multi-Agent System ===")
    print(json.dumps(agent_data, indent=2, default=str))
    
    print("\n=== Key Signals for Agent Decision Making ===")
    print(f"• Overall Signal: {agent_data['agent_context']['overall_signal']}")
    print(f"• Signal Strength: {agent_data['agent_context']['signal_strength']}")
    print(f"• Recommended Action: {agent_data['agent_context']['recommended_action']}")
    print(f"• Confidence Score: {agent_data['agent_context']['confidence_score']:.1%}")
    print(f"• Is Bullish: {agent_data['agent_context']['is_bullish']}")
    print(f"• Key Insights: {', '.join(agent_data['agent_context']['key_insights'])}")
    print(f"• Support Level: {agent_data['levels']['support']:.4f}")
    print(f"• Resistance Level: {agent_data['levels']['resistance']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())