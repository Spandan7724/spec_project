"""
Historical FX data collection using Yahoo Finance (yfinance).

This module downloads and stores historical foreign exchange rate data
for training machine learning models with real market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class YFinanceDataCollector:
    """
    Collects historical FX data from Yahoo Finance for ML training.
    
    Yahoo Finance provides reliable historical data for major currency pairs
    with good coverage and reasonable data quality.
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Yahoo Finance FX symbol mapping
        self.fx_symbols = {
            'USD/EUR': 'EURUSD=X',
            'USD/GBP': 'GBPUSD=X', 
            'USD/JPY': 'USDJPY=X',
            'EUR/GBP': 'EURGBP=X',
            'EUR/JPY': 'EURJPY=X',
            'GBP/JPY': 'GBPJPY=X',
            'USD/CHF': 'USDCHF=X',
            'EUR/CHF': 'EURCHF=X',
            'GBP/CHF': 'GBPCHF=X',
            'AUD/USD': 'AUDUSD=X',
            'USD/CAD': 'USDCAD=X',
            'NZD/USD': 'NZDUSD=X'
        }
        
        logger.info(f"Initialized YFinance collector with data directory: {self.data_dir}")
    
    def download_historical_data(
        self,
        currency_pairs: List[str],
        period: str = "2y",
        interval: str = "1h",
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical FX data from Yahoo Finance.
        
        Args:
            currency_pairs: List of currency pairs (e.g., ['USD/EUR', 'USD/GBP'])
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            save: Whether to save data to disk
            
        Returns:
            Dictionary mapping currency pairs to DataFrames
        """
        logger.info(f"Downloading {len(currency_pairs)} currency pairs for period {period} with {interval} interval")
        
        results = {}
        
        for pair in currency_pairs:
            if pair not in self.fx_symbols:
                logger.warning(f"Currency pair {pair} not supported, skipping")
                continue
            
            yahoo_symbol = self.fx_symbols[pair]
            logger.info(f"Downloading {pair} ({yahoo_symbol})...")
            
            try:
                # Download data from Yahoo Finance
                ticker = yf.Ticker(yahoo_symbol)
                data = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )
                
                if data.empty:
                    logger.error(f"No data received for {pair}")
                    continue
                
                # Clean and prepare data
                df = self._prepare_dataframe(data, pair)
                results[pair] = df
                
                logger.info(f"Downloaded {len(df)} records for {pair} from {df.index.min()} to {df.index.max()}")
                
                # Save to disk if requested
                if save:
                    self._save_data(df, pair, period, interval)
                    
            except Exception as e:
                logger.error(f"Failed to download data for {pair}: {e}")
                continue
        
        logger.info(f"Downloaded {len(results)} currency pairs successfully")
        return results
    
    def _prepare_dataframe(self, data: pd.DataFrame, currency_pair: str) -> pd.DataFrame:
        """Prepare and clean Yahoo Finance data."""
        df = data.copy()
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Use close price as main rate
        df['rate'] = df['close']
        
        # Add currency pair column
        df['currency_pair'] = currency_pair
        df['provider'] = 'YahooFinance'
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        df = df.rename(columns={'Datetime': 'timestamp'})
        
        # Remove any NaN values
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _save_data(self, df: pd.DataFrame, currency_pair: str, period: str, interval: str):
        """Save DataFrame to disk."""
        filename = f"{currency_pair.replace('/', '_')}_{period}_{interval}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
        
        # Also save as pickle for faster loading
        pickle_path = filepath.with_suffix('.pkl')
        df.to_pickle(pickle_path)
    
    def load_historical_data(
        self,
        currency_pair: str,
        period: str = "2y", 
        interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """Load previously downloaded data from disk."""
        filename = f"{currency_pair.replace('/', '_')}_{period}_{interval}.pkl"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"No saved data found for {currency_pair} at {filepath}")
            return None
        
        try:
            df = pd.read_pickle(filepath)
            logger.info(f"Loaded {len(df)} records for {currency_pair} from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data for {currency_pair}: {e}")
            return None
    
    def get_available_data(self) -> List[Dict]:
        """Get list of available downloaded data files."""
        available = []
        
        for file_path in self.data_dir.glob("*.pkl"):
            try:
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    currency_pair = parts[0] + '/' + parts[1]
                    period = parts[2]
                    interval = parts[3] if len(parts) > 3 else '1h'
                    
                    # Get file info
                    df = pd.read_pickle(file_path)
                    
                    available.append({
                        'currency_pair': currency_pair,
                        'period': period,
                        'interval': interval,
                        'records': len(df),
                        'start_date': df['timestamp'].min(),
                        'end_date': df['timestamp'].max(),
                        'file_path': str(file_path)
                    })
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
        
        return available
    
    def combine_currency_data(
        self,
        currency_pairs: List[str],
        period: str = "2y",
        interval: str = "1h"
    ) -> pd.DataFrame:
        """Combine data from multiple currency pairs into single DataFrame."""
        all_data = []
        
        for pair in currency_pairs:
            df = self.load_historical_data(pair, period, interval)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            logger.error("No data available for any currency pairs")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['currency_pair', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Combined {len(combined)} records from {len(all_data)} currency pairs")
        return combined
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for the dataset."""
        if df.empty:
            return {"error": "Empty dataset"}
        
        summary = {
            "total_records": len(df),
            "currency_pairs": df['currency_pair'].nunique(),
            "pairs_list": sorted(df['currency_pair'].unique().tolist()),
            "date_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max(),
                "days": (df['timestamp'].max() - df['timestamp'].min()).days
            },
            "rate_statistics": {
                "mean": df['rate'].mean(),
                "std": df['rate'].std(),
                "min": df['rate'].min(),
                "max": df['rate'].max(),
                "median": df['rate'].median()
            },
            "missing_values": df.isnull().sum().sum(),
            "data_quality": {
                "complete_days": len(df.groupby(df['timestamp'].dt.date)),
                "avg_records_per_day": len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).days)
            }
        }
        
        return summary


def download_fx_data_for_training(
    currency_pairs: Optional[List[str]] = None,
    period: str = "2y",
    interval: str = "1h"
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download FX data for ML training.
    
    Args:
        currency_pairs: List of pairs to download (defaults to major pairs)
        period: Time period to download
        interval: Data frequency
        
    Returns:
        Dictionary of DataFrames with historical data
    """
    if currency_pairs is None:
        currency_pairs = ['USD/EUR', 'USD/GBP', 'USD/JPY', 'EUR/GBP']
    
    collector = YFinanceDataCollector()
    
    # Download and save data
    data = collector.download_historical_data(
        currency_pairs=currency_pairs,
        period=period,
        interval=interval,
        save=True
    )
    
    # Print summary
    for pair, df in data.items():
        summary = collector.get_data_summary(df)
        print(f"\n{pair} Summary:")
        print(f"  Records: {summary['total_records']}")
        print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Days: {summary['date_range']['days']}")
        print(f"  Rate Range: {summary['rate_statistics']['min']:.6f} - {summary['rate_statistics']['max']:.6f}")
        print(f"  Missing Values: {summary['missing_values']}")
    
    return data


if __name__ == "__main__":
    # Test the data collection
    logging.basicConfig(level=logging.INFO)
    
    print("Currency Assistant - Historical Data Collection")
    print("=" * 50)
    
    # Download data for major pairs
    data = download_fx_data_for_training(
        currency_pairs=['USD/EUR', 'USD/GBP', 'EUR/GBP'],
        period="1y",  # 1 year of data
        interval="1h"  # Hourly data
    )
    
    if data:
        print(f"\n✅ Downloaded {len(data)} currency pairs successfully!")
        print("Data saved to data/historical/ directory")
    else:
        print("❌ No data downloaded")