"""
Yahoo Finance provider implementation using yfinance library.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import asyncio
import yfinance as yf

from .base import BaseRateProvider, DataProviderError
from ..models import ExchangeRate, DataSource


logger = logging.getLogger(__name__)


class YahooFinanceProvider(BaseRateProvider):
    """
    Provider for Yahoo Finance FX data using yfinance library.
    
    Yahoo Finance is free but unofficial API - can be unstable.
    Good as a backup source for exchange rates.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self._max_requests_per_window = 30  # Conservative limit
    
    @property
    def source(self) -> DataSource:
        return DataSource.YAHOO_FINANCE
    
    @property
    def base_url(self) -> str:
        return "https://finance.yahoo.com"
    
    def _get_yahoo_symbol(self, base_currency: str, quote_currency: str) -> str:
        """
        Convert currency pair to Yahoo Finance symbol format.
        
        Examples:
        - USD/EUR -> USDEUR=X
        - GBP/USD -> GBPUSD=X
        """
        return f"{base_currency}{quote_currency}=X"
    
    async def _fetch_rate(self, base_currency: str, quote_currency: str) -> ExchangeRate:
        """
        Fetch exchange rate from Yahoo Finance using yfinance.
        
        Since yfinance is synchronous, we run it in a thread pool.
        """
        symbol = self._get_yahoo_symbol(base_currency, quote_currency)
        
        try:
            # Run yfinance in thread pool since it's blocking
            ticker_data = await asyncio.to_thread(self._get_ticker_data, symbol)
            
            if not ticker_data:
                raise DataProviderError(f"No data returned for symbol {symbol}")
            
            # Extract current price
            current_price = ticker_data.get('regularMarketPrice')
            if current_price is None:
                # Try alternative fields
                current_price = ticker_data.get('price', ticker_data.get('previousClose'))
            
            if current_price is None:
                raise DataProviderError(f"No price data found for {symbol}")
            
            rate_value = float(current_price)
            
            # Validate the rate
            if not self._validate_rate(rate_value, base_currency, quote_currency):
                raise DataProviderError(f"Invalid rate value: {rate_value}")
            
            # Extract timestamp
            timestamp = datetime.utcnow()  # Yahoo doesn't provide exact timestamp for current price
            
            # Try to get bid/ask if available
            bid = ticker_data.get('bid')
            ask = ticker_data.get('ask')
            spread = None
            
            try:
                if bid and ask:
                    bid = float(bid)
                    ask = float(ask) 
                    spread = ask - bid
            except (ValueError, TypeError):
                bid = ask = spread = None
                logger.debug(f"Could not parse bid/ask for {symbol}")
            
            return ExchangeRate(
                base_currency=base_currency,
                quote_currency=quote_currency,
                rate=rate_value,
                timestamp=timestamp,
                source=self.source,
                bid=bid,
                ask=ask,
                spread=spread,
                raw_data=ticker_data
            )
            
        except Exception as e:
            if isinstance(e, DataProviderError):
                raise
            raise DataProviderError(f"Failed to fetch rate from Yahoo Finance: {str(e)}")
    
    def _get_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data using yfinance (blocking operation).
        
        Args:
            symbol: Yahoo Finance symbol (e.g., "USDEUR=X")
            
        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Yahoo sometimes returns empty dict or None
            if not info or not isinstance(info, dict):
                raise DataProviderError(f"No ticker info for {symbol}")
            
            return info
            
        except Exception as e:
            raise DataProviderError(f"yfinance error for {symbol}: {str(e)}")
    
    async def get_historical_data(self, base_currency: str, quote_currency: str,
                                period: str = "1mo") -> Optional[Dict[str, Any]]:
        """
        Get historical exchange rate data (for future volatility analysis).
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            
        Returns:
            Historical data or None if failed
        """
        symbol = self._get_yahoo_symbol(base_currency, quote_currency)
        
        try:
            # Run in thread pool
            hist_data = await asyncio.to_thread(self._get_historical_data_sync, symbol, period)
            return hist_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _get_historical_data_sync(self, symbol: str, period: str) -> Dict[str, Any]:
        """
        Synchronous helper for historical data fetching.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise DataProviderError(f"No historical data for {symbol}")
            
            # Convert to dict format with dates
            data_records = []
            for date, row in hist.iterrows():
                data_records.append({
                    'Date': date,
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'Dividends': row.get('Dividends', 0.0),
                    'Stock Splits': row.get('Stock Splits', 0.0)
                })
            
            return {
                'symbol': symbol,
                'period': period,
                'data': data_records,
                'latest_close': float(hist['Close'].iloc[-1]) if len(hist) > 0 else None
            }
            
        except Exception as e:
            raise DataProviderError(f"Historical data error for {symbol}: {str(e)}")
    
    async def __aenter__(self):
        """Override to skip HTTP client initialization since yfinance handles it."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Override since we don't have HTTP client to close."""
        pass