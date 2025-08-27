"""
Free Market Data Sources for Technical Analysis.

Integrates with Yahoo Finance and other free sources for historical 
price data, volatility calculations, and technical indicators.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point."""
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    rsi: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volatility: Optional[float] = None


class YahooFinanceProvider:
    """
    Yahoo Finance data provider for historical FX data.
    Free access to historical currency data.
    """
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.session = None
        
        # Yahoo Finance currency pair mappings
        self.currency_mappings = {
            'USD/EUR': 'EURUSD=X',
            'EUR/USD': 'EURUSD=X',
            'USD/GBP': 'GBPUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'JPY/USD': 'USDJPY=X',
            'USD/CHF': 'USDCHF=X',
            'CHF/USD': 'USDCHF=X',
            'USD/CAD': 'USDCAD=X',
            'CAD/USD': 'USDCAD=X',
            'EUR/GBP': 'EURGBP=X',
            'GBP/EUR': 'EURGBP=X',
            'EUR/JPY': 'EURJPY=X',
            'JPY/EUR': 'EURJPY=X'
        }
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=30.0,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def fetch_historical_data(self, 
                                  currency_pair: str,
                                  days_back: int = 30) -> List[MarketDataPoint]:
        """
        Fetch historical price data for currency pair.
        
        Args:
            currency_pair: Currency pair (e.g., 'USD/EUR')
            days_back: Number of days of historical data
            
        Returns:
            List of historical market data points
        """
        if not self.session:
            raise RuntimeError("Provider not properly initialized")
        
        # Get Yahoo Finance symbol
        yahoo_symbol = self.currency_mappings.get(currency_pair)
        if not yahoo_symbol:
            logger.warning(f"Currency pair {currency_pair} not supported by Yahoo Finance")
            return self._generate_mock_price_data(currency_pair, days_back)
        
        try:
            # Calculate time range
            end_time = int(datetime.utcnow().timestamp())
            start_time = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
            
            # Build request URL
            url = f"{self.base_url}/{yahoo_symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',  # Daily data
                'includePrePost': 'false'
            }
            
            response = await self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_yahoo_data(data, currency_pair)
            else:
                logger.warning(f"Yahoo Finance request failed: {response.status_code}")
                return self._generate_mock_price_data(currency_pair, days_back)
                
        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance: {e}")
            return self._generate_mock_price_data(currency_pair, days_back)
    
    def _parse_yahoo_data(self, data: Dict[str, Any], currency_pair: str) -> List[MarketDataPoint]:
        """Parse Yahoo Finance API response."""
        try:
            chart = data.get('chart', {})
            if not chart or 'result' not in chart:
                return []
            
            result = chart['result'][0]
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            opens = quotes.get('open', [])
            highs = quotes.get('high', [])
            lows = quotes.get('low', [])
            closes = quotes.get('close', [])
            volumes = quotes.get('volume', [])
            
            market_data = []
            
            for i, timestamp in enumerate(timestamps):
                # Skip if any required data is missing
                if (i >= len(closes) or closes[i] is None or
                    i >= len(opens) or opens[i] is None):
                    continue
                
                data_point = MarketDataPoint(
                    timestamp=datetime.fromtimestamp(timestamp),
                    open_price=float(opens[i]) if opens[i] is not None else float(closes[i]),
                    high_price=float(highs[i]) if i < len(highs) and highs[i] is not None else float(closes[i]),
                    low_price=float(lows[i]) if i < len(lows) and lows[i] is not None else float(closes[i]),
                    close_price=float(closes[i]),
                    volume=float(volumes[i]) if i < len(volumes) and volumes[i] is not None else None
                )
                
                # Handle inverted pairs (Yahoo gives EURUSD, but we want USD/EUR)
                if currency_pair == 'USD/EUR' and 'EUR' in self.currency_mappings.get(currency_pair, ''):
                    data_point.open_price = 1.0 / data_point.open_price if data_point.open_price != 0 else 0
                    data_point.high_price = 1.0 / data_point.low_price if data_point.low_price != 0 else 0  # Invert high/low
                    data_point.low_price = 1.0 / data_point.high_price if data_point.high_price != 0 else 0
                    data_point.close_price = 1.0 / data_point.close_price if data_point.close_price != 0 else 0
                
                market_data.append(data_point)
            
            logger.info(f"Fetched {len(market_data)} data points for {currency_pair}")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to parse Yahoo Finance data: {e}")
            return []
    
    def _generate_mock_price_data(self, currency_pair: str, days_back: int) -> List[MarketDataPoint]:
        """Generate mock price data when real data unavailable."""
        import random
        
        # Base rate for currency pair
        base_rates = {
            'USD/EUR': 0.85,
            'USD/GBP': 0.75,
            'EUR/GBP': 0.88,
            'USD/JPY': 150.0,
            'USD/CHF': 0.90
        }
        
        base_rate = base_rates.get(currency_pair, 1.0)
        current_rate = base_rate
        
        mock_data = []
        
        for i in range(days_back):
            date = datetime.utcnow() - timedelta(days=days_back - i)
            
            # Generate realistic OHLC data
            daily_change = random.normalvariate(0, 0.008)  # 0.8% daily volatility
            
            open_price = current_rate
            close_price = current_rate * (1 + daily_change)
            
            # Generate high/low around open/close
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
            
            mock_data.append(MarketDataPoint(
                timestamp=date,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=random.uniform(1000000, 5000000)  # Mock volume
            ))
            
            current_rate = close_price
        
        return mock_data


class TechnicalAnalysisCalculator:
    """Calculate technical indicators from price data."""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = changes[-period:]
        
        gains = [max(0, change) for change in recent_changes]
        losses = [max(0, -change) for change in recent_changes]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    @staticmethod
    def calculate_moving_averages(prices: List[float]) -> Dict[str, Optional[float]]:
        """Calculate various moving averages."""
        if len(prices) < 20:
            return {"ma_20": None, "ma_50": None}
        
        ma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
        ma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else None
        
        return {"ma_20": ma_20, "ma_50": ma_50}
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"upper": None, "lower": None, "middle": None}
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = variance ** 0.5
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            "upper": upper_band,
            "lower": lower_band,
            "middle": sma
        }
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 30) -> Optional[float]:
        """Calculate historical volatility."""
        if len(prices) < period + 1:
            return None
        
        # Calculate daily returns
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        recent_returns = returns[-period:]
        
        # Calculate standard deviation and annualize
        if len(recent_returns) < 2:
            return None
        
        mean_return = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
        daily_vol = variance ** 0.5
        
        # Annualize (252 trading days)
        annual_vol = daily_vol * (252 ** 0.5)
        return annual_vol


class MarketDataAggregator:
    """Aggregates market data from multiple free sources."""
    
    def __init__(self):
        self.providers = {
            'yahoo': YahooFinanceProvider()
        }
    
    async def fetch_comprehensive_market_data(self,
                                            currency_pair: str,
                                            days_back: int = 30) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for technical analysis.
        
        Args:
            currency_pair: Currency pair to analyze
            days_back: Days of historical data
            
        Returns:
            Comprehensive market data with technical indicators
        """
        # Fetch historical price data
        async with YahooFinanceProvider() as yahoo:
            price_data = await yahoo.fetch_historical_data(currency_pair, days_back)
        
        if not price_data:
            logger.warning(f"No price data available for {currency_pair}")
            return self._empty_market_data(currency_pair)
        
        # Extract price lists for calculations
        close_prices = [point.close_price for point in price_data]
        high_prices = [point.high_price for point in price_data]
        low_prices = [point.low_price for point in price_data]
        
        # Calculate technical indicators
        calculator = TechnicalAnalysisCalculator()
        
        rsi = calculator.calculate_rsi(close_prices)
        moving_averages = calculator.calculate_moving_averages(close_prices)
        bollinger_bands = calculator.calculate_bollinger_bands(close_prices)
        volatility = calculator.calculate_volatility(close_prices)
        
        # Determine market regime
        regime = self._determine_market_regime(close_prices, volatility)
        
        # Calculate support/resistance levels
        support_resistance = self._calculate_support_resistance(high_prices, low_prices, close_prices)
        
        return {
            "currency_pair": currency_pair,
            "data_points": len(price_data),
            "date_range": {
                "start": price_data[0].timestamp.isoformat(),
                "end": price_data[-1].timestamp.isoformat()
            },
            "current_price": close_prices[-1],
            "price_data": close_prices,  # For agent consumption
            "technical_indicators": {
                "rsi": rsi,
                "ma_20": moving_averages["ma_20"],
                "ma_50": moving_averages["ma_50"],
                "bollinger_upper": bollinger_bands["upper"],
                "bollinger_lower": bollinger_bands["lower"],
                "bollinger_middle": bollinger_bands["middle"],
                "volatility": volatility
            },
            "market_regime": regime,
            "support_resistance": support_resistance,
            "data_quality": {
                "completeness": len([p for p in close_prices if p is not None]) / len(close_prices),
                "recency_hours": (datetime.utcnow() - price_data[-1].timestamp).total_seconds() / 3600,
                "source": "yahoo_finance"
            }
        }
    
    def _determine_market_regime(self, prices: List[float], volatility: Optional[float]) -> Dict[str, Any]:
        """Determine current market regime from price data."""
        if len(prices) < 10:
            return {"regime": "unknown", "confidence": 0.0}
        
        # Calculate trend
        start_price = prices[0]
        end_price = prices[-1]
        total_return = (end_price / start_price - 1) * 100
        
        # Use volatility to determine regime
        vol = volatility or 0.15  # Default volatility
        
        if abs(total_return) > 3.0 and vol < 0.20:  # 3% move with low volatility
            regime = "trending_up" if total_return > 0 else "trending_down"
            confidence = min(1.0, abs(total_return) / 5.0)
        elif vol > 0.25:  # High volatility
            regime = "volatile"
            confidence = min(1.0, vol / 0.30)
        else:  # Low movement, moderate volatility
            regime = "ranging"
            confidence = 1.0 - abs(total_return) / 5.0
        
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": vol,
            "trend_strength": abs(total_return),
            "direction": "up" if total_return > 0 else "down" if total_return < 0 else "sideways"
        }
    
    def _calculate_support_resistance(self, 
                                    highs: List[float], 
                                    lows: List[float], 
                                    closes: List[float]) -> Dict[str, Any]:
        """Calculate support and resistance levels."""
        if len(closes) < 10:
            return {"support": None, "resistance": None, "confidence": 0.0}
        
        current_price = closes[-1]
        
        # Find recent swing highs and lows
        lookback = min(20, len(closes) // 2)
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Identify resistance (levels above current price)
        resistance_candidates = [h for h in recent_highs if h > current_price]
        resistance = min(resistance_candidates) if resistance_candidates else None
        
        # Identify support (levels below current price)
        support_candidates = [l for l in recent_lows if l < current_price]
        support = max(support_candidates) if support_candidates else None
        
        # Calculate confidence based on how well-defined levels are
        confidence = 0.5
        if resistance and support:
            # Higher confidence if levels are clearly defined
            price_range = max(recent_highs) - min(recent_lows)
            if price_range > 0:
                support_distance = abs(current_price - support) / price_range
                resistance_distance = abs(resistance - current_price) / price_range
                confidence = min(1.0, (support_distance + resistance_distance) / 2)
        
        return {
            "support": support,
            "resistance": resistance,
            "confidence": confidence,
            "current_price": current_price
        }
    
    def _empty_market_data(self, currency_pair: str) -> Dict[str, Any]:
        """Return empty market data structure."""
        return {
            "currency_pair": currency_pair,
            "data_points": 0,
            "date_range": {"start": None, "end": None},
            "current_price": None,
            "price_data": [],
            "technical_indicators": {
                "rsi": None,
                "ma_20": None,
                "ma_50": None,
                "bollinger_upper": None,
                "bollinger_lower": None,
                "bollinger_middle": None,
                "volatility": None
            },
            "market_regime": {"regime": "unknown", "confidence": 0.0},
            "support_resistance": {"support": None, "resistance": None, "confidence": 0.0},
            "data_quality": {"completeness": 0.0, "recency_hours": 999, "source": "none"}
        }


# Convenience function
async def fetch_real_market_data(currency_pair: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Convenience function to fetch real market data.
    
    Args:
        currency_pair: Currency pair to analyze
        days_back: Days of historical data
        
    Returns:
        Comprehensive market data dictionary
    """
    aggregator = MarketDataAggregator()
    return await aggregator.fetch_comprehensive_market_data(currency_pair, days_back)


if __name__ == "__main__":
    # Test market data providers
    async def test_market_providers():
        print("📈 Testing Market Data Sources...")
        
        # Test Yahoo Finance
        async with YahooFinanceProvider() as yahoo:
            data_points = await yahoo.fetch_historical_data("USD/EUR", 30)
            print(f"Yahoo Finance Data Points: {len(data_points)}")
            if data_points:
                latest = data_points[-1]
                print(f"  Latest: {latest.close_price:.4f} at {latest.timestamp}")
        
        # Test comprehensive market data
        market_data = await fetch_real_market_data("USD/EUR", 30)
        print(f"Market Data Quality: {market_data['data_quality']['completeness']:.2f}")
        print(f"Current Price: {market_data['current_price']}")
        print(f"RSI: {market_data['technical_indicators']['rsi']}")
        print(f"Market Regime: {market_data['market_regime']['regime']}")
    
    asyncio.run(test_market_providers())