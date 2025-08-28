"""
Technical indicator calculations for currency rate analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .historical_data import HistoricalDataCollector

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """
    Container for technical indicator results for a currency pair.
    """
    currency_pair: str
    data_date: datetime
    
    # Moving Averages
    sma_20: Optional[float] = None  # 20-day Simple Moving Average
    sma_50: Optional[float] = None  # 50-day Simple Moving Average  
    ema_12: Optional[float] = None  # 12-day Exponential Moving Average
    ema_26: Optional[float] = None  # 26-day Exponential Moving Average
    
    # Bollinger Bands
    bb_upper: Optional[float] = None    # Upper Bollinger Band
    bb_middle: Optional[float] = None   # Middle Bollinger Band (20-day SMA)
    bb_lower: Optional[float] = None    # Lower Bollinger Band
    bb_width: Optional[float] = None    # Band width as percentage
    bb_position: Optional[float] = None # Current position within bands (0-1)
    
    # Volatility Indicators
    realized_volatility: Optional[float] = None  # 30-day realized volatility (annualized)
    volatility_regime: Optional[str] = None      # 'low', 'medium', 'high'
    atr_14: Optional[float] = None               # 14-day Average True Range
    
    # Momentum Indicators
    rsi_14: Optional[float] = None      # 14-day Relative Strength Index
    macd: Optional[float] = None        # MACD line (12 EMA - 26 EMA)
    macd_signal: Optional[float] = None # MACD signal line (9 EMA of MACD)
    macd_histogram: Optional[float] = None # MACD histogram
    
    # Trend Indicators
    trend_direction: Optional[str] = None    # 'up', 'down', 'sideways'
    trend_strength: Optional[float] = None   # Trend strength 0-1
    support_level: Optional[float] = None    # Nearest support level
    resistance_level: Optional[float] = None # Nearest resistance level
    
    @property
    def is_bullish(self) -> bool:
        """Check if current indicators suggest bullish sentiment."""
        bullish_signals = 0
        total_signals = 0
        
        # Moving average signals
        if self.sma_20 and self.sma_50:
            total_signals += 1
            if self.sma_20 > self.sma_50:  # Golden cross territory
                bullish_signals += 1
        
        # Bollinger band signals
        if self.bb_position is not None:
            total_signals += 1
            if self.bb_position > 0.8:  # Near upper band
                bullish_signals += 1
        
        # RSI signals
        if self.rsi_14 is not None:
            total_signals += 1
            if 30 < self.rsi_14 < 70:  # Not oversold/overbought
                bullish_signals += 1
        
        # MACD signals
        if self.macd is not None and self.macd_signal is not None:
            total_signals += 1
            if self.macd > self.macd_signal:  # MACD above signal
                bullish_signals += 1
        
        return bullish_signals / total_signals > 0.5 if total_signals > 0 else False
    
    @property
    def volatility_score(self) -> Optional[float]:
        """Get normalized volatility score 0-1 (low to high)."""
        if self.realized_volatility is None:
            return None
        
        # Normalize based on typical FX volatility ranges
        # Low: < 5%, Medium: 5-15%, High: > 15%
        normalized = min(self.realized_volatility / 0.25, 1.0)  # Cap at 25%
        return normalized
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'currency_pair': self.currency_pair,
            'data_date': self.data_date.isoformat(),
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'ema_12': self.ema_12,
            'ema_26': self.ema_26,
            'bb_upper': self.bb_upper,
            'bb_middle': self.bb_middle,
            'bb_lower': self.bb_lower,
            'bb_width': self.bb_width,
            'bb_position': self.bb_position,
            'realized_volatility': self.realized_volatility,
            'volatility_regime': self.volatility_regime,
            'atr_14': self.atr_14,
            'rsi_14': self.rsi_14,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'support_level': self.support_level,
            'resistance_level': self.resistance_level,
            'is_bullish': self.is_bullish,
            'volatility_score': self.volatility_score
        }


class TechnicalIndicatorEngine:
    """
    Calculates technical indicators for currency pairs using historical data.
    """
    
    def __init__(self):
        self.data_collector = HistoricalDataCollector()
        self._cache: Dict[str, TechnicalIndicators] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)  # Cache indicators for 30 minutes
    
    async def calculate_indicators(self, 
                                 currency_pair: str, 
                                 force_refresh: bool = False) -> Optional[TechnicalIndicators]:
        """
        Calculate all technical indicators for a currency pair.
        
        Args:
            currency_pair: Currency pair (e.g., 'USD/EUR')
            force_refresh: Whether to force refresh cached indicators
            
        Returns:
            TechnicalIndicators object or None if failed
        """
        cache_key = currency_pair
        
        # Check cache first
        if not force_refresh and self._is_cached_valid(cache_key):
            logger.debug(f"Using cached indicators for {currency_pair}")
            return self._cache[cache_key]
        
        logger.info(f"Calculating technical indicators for {currency_pair}")
        
        try:
            # Get historical data (we need at least 50 days for all indicators)
            dataset = await self.data_collector.get_historical_data(currency_pair, days=90)
            
            if not dataset or len(dataset.data) < 20:
                logger.warning(f"Insufficient data for {currency_pair}: {len(dataset.data) if dataset else 0} points")
                return None
            
            # Convert to DataFrame for indicator calculations
            df = dataset.to_dataframe()
            if df.empty:
                logger.warning(f"Empty DataFrame for {currency_pair}")
                return None
            
            # Calculate indicators
            indicators = TechnicalIndicators(
                currency_pair=currency_pair,
                data_date=datetime.utcnow()
            )
            
            # Moving Averages
            indicators.sma_20 = self._calculate_sma(df['close'], 20)
            indicators.sma_50 = self._calculate_sma(df['close'], 50) if len(df) >= 50 else None
            indicators.ema_12 = self._calculate_ema(df['close'], 12)
            indicators.ema_26 = self._calculate_ema(df['close'], 26)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['close'], 20, 2.0)
            if bb_data:
                indicators.bb_upper = bb_data['upper']
                indicators.bb_middle = bb_data['middle']  
                indicators.bb_lower = bb_data['lower']
                indicators.bb_width = bb_data['width']
                indicators.bb_position = bb_data['position']
            
            # Volatility Indicators
            indicators.realized_volatility = self._calculate_realized_volatility(df['close'], 30)
            indicators.volatility_regime = self._classify_volatility_regime(indicators.realized_volatility)
            indicators.atr_14 = self._calculate_atr(df, 14)
            
            # Momentum Indicators
            indicators.rsi_14 = self._calculate_rsi(df['close'], 14)
            macd_data = self._calculate_macd(df['close'], 12, 26, 9)
            if macd_data:
                indicators.macd = macd_data['macd']
                indicators.macd_signal = macd_data['signal']
                indicators.macd_histogram = macd_data['histogram']
            
            # Trend Analysis
            trend_data = self._analyze_trend(df['close'])
            if trend_data:
                indicators.trend_direction = trend_data['direction']
                indicators.trend_strength = trend_data['strength']
            
            # Support/Resistance levels
            sr_levels = self._calculate_support_resistance(df['close'], df['high'], df['low'])
            if sr_levels:
                indicators.support_level = sr_levels['support']
                indicators.resistance_level = sr_levels['resistance']
            
            # Cache the result
            self._cache[cache_key] = indicators
            self._cache_timestamp[cache_key] = datetime.utcnow()
            
            logger.info(f"Successfully calculated indicators for {currency_pair}")
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {currency_pair}: {e}")
            return None
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached indicators are still valid."""
        if cache_key not in self._cache or cache_key not in self._cache_timestamp:
            return False
        
        age = datetime.utcnow() - self._cache_timestamp[cache_key]
        return age < self._cache_ttl
    
    def _calculate_sma(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(series) < period:
            return None
        return float(series.tail(period).mean())
    
    def _calculate_ema(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(series) < period:
            return None
        return float(series.ewm(span=period).mean().iloc[-1])
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int, 
                                 std_dev: float) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands."""
        if len(series) < period:
            return None
        
        sma = series.tail(period).mean()
        std = series.tail(period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        current_price = series.iloc[-1]
        
        # Band width as percentage
        width = ((upper - lower) / sma) * 100 if sma != 0 else 0
        
        # Position within bands (0 = lower band, 1 = upper band)
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5
        
        return {
            'upper': float(upper),
            'middle': float(sma),
            'lower': float(lower),
            'width': float(width),
            'position': float(max(0, min(1, position)))  # Clamp between 0-1
        }
    
    def _calculate_realized_volatility(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate realized volatility (annualized)."""
        if len(series) < period + 1:
            return None
        
        returns = series.pct_change().dropna().tail(period)
        if len(returns) < 2:
            return None
        
        # Annualized volatility (assuming 252 trading days per year)
        volatility = returns.std() * np.sqrt(252)
        return float(volatility)
    
    def _classify_volatility_regime(self, volatility: Optional[float]) -> Optional[str]:
        """Classify volatility into regime categories."""
        if volatility is None:
            return None
        
        if volatility < 0.05:  # Less than 5%
            return 'low'
        elif volatility < 0.15:  # 5% to 15%
            return 'medium'
        else:  # Greater than 15%
            return 'high'
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> Optional[float]:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return None
        
        high = df['high']
        low = df['low'] 
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.tail(period).mean()
        
        return float(atr) if not pd.isna(atr) else None
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(series) < period + 1:
            return None
        
        delta = series.diff()
        gain = delta.where(delta > 0, 0).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        
        if loss == 0:
            return 100.0  # All gains, RSI = 100
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi) if not pd.isna(rsi) else None
    
    def _calculate_macd(self, series: pd.Series, fast: int, slow: int, 
                      signal: int) -> Optional[Dict[str, float]]:
        """Calculate MACD indicator."""
        if len(series) < slow + signal:
            return None
        
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    
    def _analyze_trend(self, series: pd.Series) -> Optional[Dict[str, any]]:
        """Analyze price trend direction and strength."""
        if len(series) < 20:
            return None
        
        # Use linear regression to determine trend
        x = np.arange(len(series))
        y = series.values
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend direction
        if abs(slope) < series.std() * 0.01:  # Very small slope
            direction = 'sideways'
            strength = 0.1
        elif slope > 0:
            direction = 'up'
            strength = min(1.0, abs(slope) / (series.std() * 0.1))
        else:
            direction = 'down'
            strength = min(1.0, abs(slope) / (series.std() * 0.1))
        
        return {
            'direction': direction,
            'strength': float(strength)
        }
    
    def _calculate_support_resistance(self, close: pd.Series, high: pd.Series, 
                                    low: pd.Series) -> Optional[Dict[str, float]]:
        """Calculate nearest support and resistance levels."""
        if len(close) < 20:
            return None
        
        current_price = close.iloc[-1]
        
        # Simple approach: use recent highs and lows
        recent_period = min(20, len(close))
        recent_highs = high.tail(recent_period)
        recent_lows = low.tail(recent_period)
        
        # Find resistance (nearest high above current price)
        resistance_candidates = recent_highs[recent_highs > current_price]
        resistance = resistance_candidates.min() if not resistance_candidates.empty else current_price * 1.02
        
        # Find support (nearest low below current price)
        support_candidates = recent_lows[recent_lows < current_price]
        support = support_candidates.max() if not support_candidates.empty else current_price * 0.98
        
        return {
            'support': float(support),
            'resistance': float(resistance)
        }
    
    def clear_cache(self, currency_pair: Optional[str] = None) -> None:
        """Clear cached indicators."""
        if currency_pair:
            self._cache.pop(currency_pair, None)
            self._cache_timestamp.pop(currency_pair, None)
            logger.info(f"Cleared indicator cache for {currency_pair}")
        else:
            self._cache.clear()
            self._cache_timestamp.clear()
            logger.info("Cleared all indicator cache")


# Convenience functions
async def get_technical_indicators(currency_pair: str) -> Optional[TechnicalIndicators]:
    """
    Convenience function to get technical indicators for a currency pair.
    
    Args:
        currency_pair: Currency pair (e.g., 'USD/EUR')
        
    Returns:
        TechnicalIndicators object or None if failed
    """
    engine = TechnicalIndicatorEngine()
    return await engine.calculate_indicators(currency_pair)


async def get_volatility_analysis(currency_pair: str) -> Optional[Dict[str, any]]:
    """
    Get focused volatility analysis for a currency pair.
    
    Args:
        currency_pair: Currency pair (e.g., 'USD/EUR')
        
    Returns:
        Dictionary with volatility metrics or None if failed
    """
    indicators = await get_technical_indicators(currency_pair)
    if not indicators:
        return None
    
    return {
        'currency_pair': currency_pair,
        'realized_volatility': indicators.realized_volatility,
        'volatility_regime': indicators.volatility_regime,
        'volatility_score': indicators.volatility_score,
        'atr_14': indicators.atr_14,
        'bb_width': indicators.bb_width,
        'analysis_date': indicators.data_date.isoformat()
    }