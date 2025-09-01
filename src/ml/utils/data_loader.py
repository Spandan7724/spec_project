"""
Data loader for integrating with Layer 1 data sources
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add Layer 1 data collection modules to path
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class Layer1DataLoader:
    """
    Data loader that integrates with existing Layer 1 data collection systems
    """
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_historical_rates(self, 
                             currency_pair: str = "USD/EUR",
                             days: int = 365,
                             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load historical exchange rates from Layer 1 data collection
        """
        try:
            # Try to import and use existing Layer 1 collectors
            from data_collection.providers.yahoo_finance import YahooFinanceProvider
            from data_collection.providers.alpha_vantage import AlphaVantageProvider
            
            if end_date is None:
                end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use the historical data collector from Layer 1
            from data_collection.analysis.historical_data import HistoricalDataCollector
            import asyncio
            
            collector = HistoricalDataCollector()
            
            # Run the async method - handle both sync and async contexts
            try:
                # Check if we're already in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context - need to use create_task to avoid blocking
                logger.info("Running in async context, using proper async handling")
                import concurrent.futures
                
                # Run in thread pool to avoid blocking the async loop
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(collector.get_historical_data(currency_pair, days)))
                    historical_dataset = future.result()
                    
            except RuntimeError:
                # No running loop, we can create one
                historical_dataset = asyncio.run(collector.get_historical_data(currency_pair, days))
            
            if historical_dataset and historical_dataset.data:
                df = historical_dataset.to_dataframe()
                logger.info(f"Loaded {len(df)} records from Layer 1 historical data for {currency_pair}")
                
                # Add returns columns that our ML system expects
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                return df
                
        except ImportError:
            logger.warning("Layer 1 data providers not available, using mock data")
        
        # Fallback to mock data for testing
        return self._generate_mock_data(currency_pair, days, end_date)
    
    def load_technical_indicators(self,
                                currency_pair: str = "USD/EUR",
                                days: int = 365) -> pd.DataFrame:
        """
        Load technical indicators from Layer 1 rate trends analysis
        """
        try:
            # Try to import existing technical indicators
            from data_collection.analysis.technical_indicators import TechnicalIndicatorEngine
            
            # Load base price data first
            price_data = self.load_historical_rates(currency_pair, days)
            
            # Use the technical indicator engine from Layer 1
            import asyncio
            
            engine = TechnicalIndicatorEngine()
            
            # Run the async method - handle both sync and async contexts
            try:
                # Check if we're already in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread pool to avoid blocking
                logger.info("Running in async context, using proper async handling for technical indicators")
                import concurrent.futures
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(engine.calculate_indicators(currency_pair)))
                    tech_indicators = future.result()
                    
            except RuntimeError:
                # No running loop, we can create one
                tech_indicators = asyncio.run(engine.calculate_indicators(currency_pair))
            
            if tech_indicators:
                # Convert to DataFrame format expected by ML system
                indicator_data = pd.DataFrame(index=price_data.index)
                
                # Fill with the calculated indicators
                if tech_indicators.sma_20 is not None:
                    indicator_data.loc[price_data.index[-1], 'sma_20'] = tech_indicators.sma_20
                if tech_indicators.sma_50 is not None:
                    indicator_data.loc[price_data.index[-1], 'sma_50'] = tech_indicators.sma_50
                    
                if tech_indicators.ema_12 is not None:
                    indicator_data.loc[price_data.index[-1], 'ema_12'] = tech_indicators.ema_12
                if tech_indicators.ema_26 is not None:
                    indicator_data.loc[price_data.index[-1], 'ema_26'] = tech_indicators.ema_26
                    
                if tech_indicators.bb_upper is not None:
                    indicator_data.loc[price_data.index[-1], 'bb_upper'] = tech_indicators.bb_upper
                    indicator_data.loc[price_data.index[-1], 'bb_middle'] = tech_indicators.bb_middle
                    indicator_data.loc[price_data.index[-1], 'bb_lower'] = tech_indicators.bb_lower
                    
                if tech_indicators.rsi_14 is not None:
                    indicator_data.loc[price_data.index[-1], 'rsi'] = tech_indicators.rsi_14
                    
                if tech_indicators.macd is not None:
                    indicator_data.loc[price_data.index[-1], 'macd'] = tech_indicators.macd
                    indicator_data.loc[price_data.index[-1], 'macd_signal'] = tech_indicators.macd_signal
                    indicator_data.loc[price_data.index[-1], 'macd_histogram'] = tech_indicators.macd_histogram
                    
                if tech_indicators.realized_volatility is not None:
                    indicator_data.loc[price_data.index[-1], 'volatility'] = tech_indicators.realized_volatility
                
                # Forward fill missing values for older dates (simplified approach)
                indicator_data = indicator_data.fillna(method='bfill')
            
            logger.info(f"Loaded technical indicators for {currency_pair}")
            return indicator_data
            
        except ImportError:
            logger.warning("Layer 1 technical indicators not available, using mock indicators")
            return self._generate_mock_indicators(currency_pair, days)
    
    def load_economic_events(self,
                           days: int = 365,
                           currencies: List[str] = None) -> pd.DataFrame:
        """
        Load economic calendar events from Layer 1 economic data
        """
        if currencies is None:
            currencies = ['USD', 'EUR', 'GBP']
        
        try:
            # Try to import existing economic calendar
            from data_collection.economic.fred_provider import FREDProvider
            import asyncio
            import os
            
            # Check if we have FRED API key
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                logger.warning("No FRED API key found in environment, using mock economic events")
                return self._generate_mock_economic_events(days, currencies)
            
            # Initialize FRED provider with API key
            fred = FREDProvider(fred_api_key)
            events_data = []
            
            # Common economic indicators
            indicators = {
                'GDP': 'GDP',
                'CPI': 'CPIAUCSL',  # Consumer Price Index
                'UNEMPLOYMENT': 'UNRATE',
                'INTEREST_RATE': 'FEDFUNDS'
            }
            
            # Get upcoming economic events - handle both sync and async contexts
            try:
                # Check if we're already in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread pool to avoid blocking
                logger.info("Running in async context, using proper async handling for economic events")
                import concurrent.futures
                
                async def get_events():
                    async with fred:
                        return await fred.get_upcoming_releases(days)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(get_events()))
                    upcoming_events = future.result()
                    
            except RuntimeError:
                # No running loop, we can create one
                async def get_events():
                    async with fred:
                        return await fred.get_upcoming_releases(days)
                upcoming_events = asyncio.run(get_events())
                
                if upcoming_events:
                    for event in upcoming_events:
                        events_data.append({
                            'date': event.release_date,
                            'indicator': event.title,
                            'value': event.previous_value,
                            'currency': event.currency,
                            'impact': event.impact.value if hasattr(event.impact, 'value') else str(event.impact)
                        })
            except Exception as e:
                logger.warning(f"Failed to load FRED events: {e}")
            
            if events_data:
                df = pd.DataFrame(events_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logger.info(f"Loaded {len(df)} economic events")
                return df
                
        except ImportError:
            logger.warning("Layer 1 economic data not available, using mock events")
        
        return self._generate_mock_economic_events(days, currencies)
    
    def _format_ohlcv_data(self, data: List[Dict], currency_pair: str) -> pd.DataFrame:
        """Format OHLCV data into standardized DataFrame"""
        records = []
        for record in data:
            records.append({
                'date': pd.to_datetime(record['timestamp']),
                'open': float(record.get('open', record['rate'])),
                'high': float(record.get('high', record['rate'])),
                'low': float(record.get('low', record['rate'])),
                'close': float(record['rate']),
                'volume': float(record.get('volume', 0))
            })
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Add returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def _generate_mock_data(self, currency_pair: str, days: int, end_date: datetime = None) -> pd.DataFrame:
        """Generate mock OHLCV data for testing"""
        logger.info(f"Generating mock data for {currency_pair}")
        
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simple random walk with trend
        np.random.seed(42)  # For reproducibility
        base_rate = 0.85 if currency_pair == "USD/EUR" else 1.25
        returns = np.random.normal(0, 0.01, days)
        prices = [base_rate]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV from close prices
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.005, 0.02)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Add returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def _generate_mock_indicators(self, currency_pair: str, days: int) -> pd.DataFrame:
        """Generate mock technical indicators for testing"""
        price_data = self.load_historical_rates(currency_pair, days)
        
        # Simple technical indicators
        indicators = pd.DataFrame(index=price_data.index)
        
        # Moving averages
        indicators['sma_20'] = price_data['close'].rolling(20).mean()
        indicators['sma_50'] = price_data['close'].rolling(50).mean()
        indicators['sma_200'] = price_data['close'].rolling(200).mean()
        
        # Simple RSI approximation
        delta = price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        indicators['bb_middle'] = price_data['close'].rolling(20).mean()
        bb_std = price_data['close'].rolling(20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        
        # Volatility
        indicators['volatility'] = price_data['returns'].rolling(20).std()
        
        return indicators
    
    def _generate_mock_economic_events(self, days: int, currencies: List[str]) -> pd.DataFrame:
        """Generate mock economic events for testing"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        events = []
        
        # Generate monthly events for each currency
        current_date = start_date
        while current_date <= end_date:
            for currency in currencies:
                # GDP (quarterly)
                if current_date.month % 3 == 1:
                    events.append({
                        'date': current_date,
                        'indicator': 'GDP',
                        'value': np.random.uniform(1.5, 3.5),  # % growth
                        'currency': currency,
                        'impact': 'HIGH'
                    })
                
                # CPI (monthly)
                events.append({
                    'date': current_date,
                    'indicator': 'CPI',
                    'value': np.random.uniform(1.0, 4.0),  # % inflation
                    'currency': currency,
                    'impact': 'HIGH'
                })
                
                # Interest Rate (monthly)
                events.append({
                    'date': current_date,
                    'indicator': 'INTEREST_RATE',
                    'value': np.random.uniform(0.0, 5.0),  # %
                    'currency': currency,
                    'impact': 'MEDIUM'
                })
            
            current_date += timedelta(days=30)
        
        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def get_combined_dataset(self,
                           currency_pair: str = "USD/EUR",
                           days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get combined dataset with prices, indicators, and economic events
        """
        logger.info(f"Loading combined dataset for {currency_pair}")
        
        # Load all data sources
        prices = self.load_historical_rates(currency_pair, days)
        indicators = self.load_technical_indicators(currency_pair, days)
        
        # Get currencies from pair
        base_currency, quote_currency = currency_pair.split('/')
        events = self.load_economic_events(days, [base_currency, quote_currency])
        
        return prices, indicators, events