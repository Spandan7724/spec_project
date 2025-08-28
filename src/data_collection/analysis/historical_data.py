"""
Historical exchange rate data collection and management.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..providers.yahoo_finance import YahooFinanceProvider
from ..providers.alpha_vantage import AlphaVantageProvider
from ..models import DataSource


logger = logging.getLogger(__name__)


@dataclass
class HistoricalRateData:
    """
    Historical exchange rate data point.
    """
    date: datetime
    open_rate: float
    high_rate: float
    low_rate: float
    close_rate: float
    volume: Optional[float] = None
    source: DataSource = DataSource.YAHOO_FINANCE
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high_rate + self.low_rate + self.close_rate) / 3
    
    @property
    def range_percent(self) -> float:
        """Calculate daily range as percentage."""
        if self.low_rate > 0:
            return ((self.high_rate - self.low_rate) / self.low_rate) * 100
        return 0.0


@dataclass
class HistoricalDataset:
    """
    Complete historical dataset for a currency pair.
    """
    currency_pair: str
    data: List[HistoricalRateData]
    start_date: datetime
    end_date: datetime
    source: DataSource
    data_quality: float = 0.0  # Percentage of expected data points present
    
    def __post_init__(self):
        """Calculate data quality after initialization."""
        if self.data:
            # Sort data by date
            self.data.sort(key=lambda x: x.date)
            self.start_date = self.data[0].date
            self.end_date = self.data[-1].date
            
            # Calculate data quality (percentage of trading days covered)
            expected_days = (self.end_date - self.start_date).days
            actual_days = len(self.data)
            # Assuming ~260 trading days per year (5 days/week)
            expected_trading_days = expected_days * (5/7)
            self.data_quality = min(1.0, actual_days / expected_trading_days) if expected_trading_days > 0 else 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        if not self.data:
            return pd.DataFrame()
        
        df_data = []
        for rate in self.data:
            df_data.append({
                'date': rate.date,
                'open': rate.open_rate,
                'high': rate.high_rate,
                'low': rate.low_rate,
                'close': rate.close_rate,
                'volume': rate.volume,
                'typical_price': rate.typical_price,
                'range_percent': rate.range_percent
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('date', inplace=True)
        return df.sort_index()
    
    def get_rate_series(self, price_type: str = 'close') -> pd.Series:
        """Get a pandas Series of rates for analysis."""
        df = self.to_dataframe()
        if price_type in df.columns:
            return df[price_type]
        return pd.Series()
    
    def get_latest_rate(self) -> Optional[HistoricalRateData]:
        """Get the most recent rate data."""
        return self.data[-1] if self.data else None
    
    def get_rate_range(self, start_date: datetime, end_date: datetime) -> List[HistoricalRateData]:
        """Get rates within a specific date range."""
        return [
            rate for rate in self.data 
            if start_date <= rate.date <= end_date
        ]


class HistoricalDataCollector:
    """
    Collects and manages historical exchange rate data from multiple sources.
    """
    
    def __init__(self):
        self._data_cache: Dict[str, HistoricalDataset] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=6)  # Cache historical data for 6 hours
    
    async def get_historical_data(self, 
                                currency_pair: str, 
                                days: int = 90,
                                force_refresh: bool = False) -> Optional[HistoricalDataset]:
        """
        Get historical data for a currency pair.
        
        Args:
            currency_pair: Currency pair (e.g., 'USD/EUR')
            days: Number of days of historical data
            force_refresh: Whether to force refresh cached data
            
        Returns:
            HistoricalDataset or None if failed
        """
        cache_key = f"{currency_pair}_{days}"
        
        # Check cache first
        if not force_refresh and self._is_cached_valid(cache_key):
            logger.debug(f"Using cached historical data for {currency_pair}")
            return self._data_cache[cache_key]
        
        logger.info(f"Fetching {days} days of historical data for {currency_pair}")
        
        # Try multiple sources with preference order
        # Note: Alpha Vantage FX intraday requires premium subscription
        sources = [
            (self._fetch_yahoo_finance_data, DataSource.YAHOO_FINANCE),
            # (self._fetch_alpha_vantage_data, DataSource.ALPHA_VANTAGE),  # Premium only
        ]
        
        for fetch_func, source in sources:
            try:
                dataset = await fetch_func(currency_pair, days)
                if dataset and dataset.data_quality > 0.7:  # Require at least 70% data quality
                    logger.info(f"Successfully collected {len(dataset.data)} days of data "
                              f"from {source.value} (quality: {dataset.data_quality:.1%})")
                    
                    # Cache the result
                    self._data_cache[cache_key] = dataset
                    self._cache_timestamp[cache_key] = datetime.utcnow()
                    
                    return dataset
                else:
                    logger.warning(f"Low quality data from {source.value}: "
                                 f"{dataset.data_quality:.1%} quality" if dataset else "No data")
                    
            except Exception as e:
                logger.error(f"Failed to fetch from {source.value}: {e}")
                continue
        
        logger.error(f"Failed to get historical data for {currency_pair} from all sources")
        return None
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._data_cache or cache_key not in self._cache_timestamp:
            return False
        
        age = datetime.utcnow() - self._cache_timestamp[cache_key]
        return age < self._cache_ttl
    
    async def _fetch_yahoo_finance_data(self, currency_pair: str, days: int) -> Optional[HistoricalDataset]:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Handle both formats: 'USD/EUR' and 'USDEUR'
            if '/' in currency_pair:
                base_currency, quote_currency = currency_pair.split('/')
            else:
                # Split 6-char format like 'USDEUR' -> 'USD', 'EUR'
                if len(currency_pair) == 6:
                    base_currency = currency_pair[:3]
                    quote_currency = currency_pair[3:]
                else:
                    logger.error(f"Invalid currency pair format: {currency_pair}")
                    return None
            
            async with YahooFinanceProvider() as provider:
                # Determine period string for yfinance
                if days <= 7:
                    period = "7d"
                elif days <= 30:
                    period = "1mo"
                elif days <= 90:
                    period = "3mo"
                elif days <= 180:
                    period = "6mo"
                else:
                    period = "1y"
                
                hist_data = await provider.get_historical_data(base_currency, quote_currency, period)
                
                if not hist_data or 'data' not in hist_data:
                    return None
                
                # Convert to our format
                rate_data = []
                for record in hist_data['data']:
                    try:
                        # Robustly convert date to naive datetime
                        ts = record.get('Date')
                        dt_value = None
                        try:
                            if isinstance(ts, pd.Timestamp):
                                dt_value = ts.to_pydatetime()
                            elif isinstance(ts, datetime):
                                dt_value = ts
                            else:
                                dt_parsed = pd.to_datetime(ts, errors='coerce')
                                if pd.isna(dt_parsed):
                                    continue
                                dt_value = dt_parsed.to_pydatetime()
                            # Drop timezone info if present
                            if getattr(dt_value, 'tzinfo', None) is not None:
                                dt_value = dt_value.replace(tzinfo=None)
                        except Exception as e:
                            logger.debug(f"Failed to parse date {ts}: {e}")
                            continue

                        # Extract OHLC
                        open_v = float(record['Open'])
                        high_v = float(record['High'])
                        low_v = float(record['Low'])
                        close_v = float(record['Close'])

                        # Volume can be missing/NaN for FX; treat as None
                        vol_raw = record.get('Volume')
                        vol_value = None
                        try:
                            if vol_raw is not None and not (isinstance(vol_raw, float) and np.isnan(vol_raw)):
                                vol_value = float(vol_raw)
                        except Exception:
                            vol_value = None

                        rate_point = HistoricalRateData(
                            date=dt_value,
                            open_rate=open_v,
                            high_rate=high_v,
                            low_rate=low_v,
                            close_rate=close_v,
                            volume=vol_value,
                            source=DataSource.YAHOO_FINANCE
                        )
                        
                        # Basic validation
                        if self._validate_rate_data(rate_point, currency_pair):
                            rate_data.append(rate_point)
                        
                    except (ValueError, KeyError, TypeError) as e:
                        logger.debug(f"Skipping invalid data point: {e}")
                        continue
                
                if not rate_data:
                    return None
                
                # Create dataset
                dataset = HistoricalDataset(
                    currency_pair=currency_pair,
                    data=rate_data,
                    start_date=min(r.date for r in rate_data),
                    end_date=max(r.date for r in rate_data),
                    source=DataSource.YAHOO_FINANCE
                )
                
                return dataset
                
        except Exception as e:
            logger.error(f"Yahoo Finance historical data error: {e}")
            return None
    
    async def _fetch_alpha_vantage_data(self, currency_pair: str, days: int) -> Optional[HistoricalDataset]:
        """Fetch historical data from Alpha Vantage."""
        try:
            # Handle both formats: 'USD/EUR' and 'USDEUR'
            if '/' in currency_pair:
                base_currency, quote_currency = currency_pair.split('/')
            else:
                # Split 6-char format like 'USDEUR' -> 'USD', 'EUR'
                if len(currency_pair) == 6:
                    base_currency = currency_pair[:3]
                    quote_currency = currency_pair[3:]
                else:
                    logger.error(f"Invalid currency pair format: {currency_pair}")
                    return None
            
            # Alpha Vantage requires API key
            import os
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                logger.debug("No Alpha Vantage API key available")
                return None
            
            async with AlphaVantageProvider(api_key) as provider:
                # Use intraday data for more recent periods, daily for longer
                if days <= 30:
                    interval = "60min"  # Hourly data for recent periods
                else:
                    # For longer periods, we'd need to implement FX_DAILY
                    # For now, skip Alpha Vantage for long periods
                    logger.debug("Alpha Vantage long-term historical data not implemented yet")
                    return None
                
                hist_data = await provider.get_intraday_data(base_currency, quote_currency, interval)
                
                if not hist_data:
                    return None
                
                # Parse Alpha Vantage response format
                time_series_key = f"Time Series FX ({interval})"
                if time_series_key not in hist_data:
                    logger.debug(f"Expected key '{time_series_key}' not found in Alpha Vantage response")
                    return None
                
                time_series = hist_data[time_series_key]
                rate_data = []
                
                for timestamp_str, ohlcv in time_series.items():
                    try:
                        timestamp = pd.to_datetime(timestamp_str).to_pydatetime()
                        
                        # Only include data within our requested timeframe
                        cutoff_date = datetime.utcnow() - timedelta(days=days)
                        if timestamp < cutoff_date:
                            continue
                        
                        rate_point = HistoricalRateData(
                            date=timestamp,
                            open_rate=float(ohlcv['1. open']),
                            high_rate=float(ohlcv['2. high']),
                            low_rate=float(ohlcv['3. low']),
                            close_rate=float(ohlcv['4. close']),
                            source=DataSource.ALPHA_VANTAGE
                        )
                        
                        if self._validate_rate_data(rate_point, currency_pair):
                            rate_data.append(rate_point)
                            
                    except (ValueError, KeyError, TypeError) as e:
                        logger.debug(f"Skipping invalid Alpha Vantage data point: {e}")
                        continue
                
                if not rate_data:
                    return None
                
                # For intraday data, we might want to resample to daily
                df = pd.DataFrame([{
                    'date': r.date,
                    'open': r.open_rate,
                    'high': r.high_rate,
                    'low': r.low_rate,
                    'close': r.close_rate,
                } for r in rate_data])
                
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Resample to daily OHLC
                daily_df = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                
                # Convert back to our format
                daily_rate_data = []
                for date, row in daily_df.iterrows():
                    rate_point = HistoricalRateData(
                        date=date.to_pydatetime(),
                        open_rate=row['open'],
                        high_rate=row['high'],
                        low_rate=row['low'],
                        close_rate=row['close'],
                        source=DataSource.ALPHA_VANTAGE
                    )
                    daily_rate_data.append(rate_point)
                
                dataset = HistoricalDataset(
                    currency_pair=currency_pair,
                    data=daily_rate_data,
                    start_date=min(r.date for r in daily_rate_data),
                    end_date=max(r.date for r in daily_rate_data), 
                    source=DataSource.ALPHA_VANTAGE
                )
                
                return dataset
                
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {e}")
            return None
    
    def _validate_rate_data(self, rate_data: HistoricalRateData, currency_pair: str) -> bool:
        """Validate historical rate data point."""
        # Basic sanity checks
        if any(rate <= 0 for rate in [rate_data.open_rate, rate_data.high_rate, 
                                     rate_data.low_rate, rate_data.close_rate]):
            return False
        
        # High should be >= Open, Close, Low
        if rate_data.high_rate < max(rate_data.open_rate, rate_data.close_rate, rate_data.low_rate):
            return False
        
        # Low should be <= Open, Close, High
        if rate_data.low_rate > min(rate_data.open_rate, rate_data.close_rate, rate_data.high_rate):
            return False
        
        # Range sanity check - daily range shouldn't exceed 20% (extreme volatility)
        if rate_data.range_percent > 20.0:
            logger.warning(f"Extreme daily range detected: {rate_data.range_percent:.1f}% on {rate_data.date}")
            return False
        
        return True
    
    def get_cached_pairs(self) -> List[str]:
        """Get list of currency pairs with cached data."""
        pairs = set()
        for cache_key in self._data_cache.keys():
            pair = cache_key.rsplit('_', 1)[0]  # Remove the days suffix
            pairs.add(pair)
        return list(pairs)
    
    def clear_cache(self, currency_pair: Optional[str] = None) -> None:
        """Clear cached data for specific pair or all pairs."""
        if currency_pair:
            # Remove all cached entries for this pair
            keys_to_remove = [key for key in self._data_cache.keys() 
                            if key.startswith(currency_pair + '_')]
            for key in keys_to_remove:
                self._data_cache.pop(key, None)
                self._cache_timestamp.pop(key, None)
            logger.info(f"Cleared cache for {currency_pair}")
        else:
            self._data_cache.clear()
            self._cache_timestamp.clear()
            logger.info("Cleared all historical data cache")


# Convenience functions
async def get_historical_rates(currency_pair: str, 
                             days: int = 90) -> Optional[HistoricalDataset]:
    """
    Convenience function to get historical rates.
    
    Args:
        currency_pair: Currency pair (e.g., 'USD/EUR')
        days: Number of days of historical data
        
    Returns:
        HistoricalDataset or None if failed
    """
    collector = HistoricalDataCollector()
    return await collector.get_historical_data(currency_pair, days)


async def get_recent_volatility(currency_pair: str, 
                               days: int = 30) -> Optional[float]:
    """
    Get recent volatility for a currency pair.
    
    Args:
        currency_pair: Currency pair
        days: Number of days to calculate volatility over
        
    Returns:
        Annualized volatility or None if failed
    """
    dataset = await get_historical_rates(currency_pair, days)
    if not dataset or len(dataset.data) < 10:
        return None
    
    # Calculate daily returns
    rates = dataset.get_rate_series('close')
    if len(rates) < 2:
        return None
    
    returns = rates.pct_change().dropna()
    
    # Calculate annualized volatility (assuming 252 trading days per year)
    volatility = returns.std() * np.sqrt(252)
    return float(volatility)
