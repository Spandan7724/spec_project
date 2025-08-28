#!/usr/bin/env python3
"""Debug historical data APIs."""

import asyncio
import sys
import os
import yfinance as yf
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.providers.yahoo_finance import YahooFinanceProvider
from data_collection.providers.alpha_vantage import AlphaVantageProvider

async def debug_yahoo_finance():
    """Debug Yahoo Finance historical data."""
    print("🔍 Debugging Yahoo Finance...")
    
    try:
        # Test direct yfinance first
        print("Testing direct yfinance...")
        ticker = yf.Ticker("EURUSD=X")
        hist = ticker.history(period="1mo")
        print(f"Direct yfinance result: {len(hist)} rows")
        if not hist.empty:
            print(hist.head())
        else:
            print("❌ No data from direct yfinance")
        
        # Test our provider
        print("\nTesting YahooFinanceProvider...")
        async with YahooFinanceProvider() as provider:
            result = await provider.get_historical_data("USD", "EUR", "1mo")
            print(f"Provider result: {result}")
            
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        import traceback
        traceback.print_exc()

async def debug_alpha_vantage():
    """Debug Alpha Vantage historical data."""
    print("\n🔍 Debugging Alpha Vantage...")
    
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            print("❌ No Alpha Vantage API key")
            return
            
        async with AlphaVantageProvider(api_key) as provider:
            result = await provider.get_intraday_data("USD", "EUR", "60min")
            print(f"Alpha Vantage result keys: {list(result.keys()) if result else 'None'}")
            if result:
                print(f"Full response: {json.dumps(result, indent=2)}")
    
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    await debug_yahoo_finance()
    await debug_alpha_vantage()

if __name__ == "__main__":
    asyncio.run(main())