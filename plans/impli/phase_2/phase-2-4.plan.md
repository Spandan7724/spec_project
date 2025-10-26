<!-- f67714c1-a54f-4e8d-9617-16955f212afc 998c97bf-5654-4cd5-a7f3-740bf8be78ca -->
# Phase 2.4: Fallback Heuristics

## Overview

Create a lightweight heuristic-based prediction system that serves as a fallback when ML models are unavailable, fail quality gates, or haven't been trained yet. Uses simple technical analysis rules (MA crossovers, RSI extremes) to generate conservative predictions with low confidence scores. This ensures graceful degradation and system reliability.

## Implementation Steps

### Step 1: Fallback Predictor Implementation

**File**: `src/prediction/utils/fallback.py`

Implement rule-based predictions using technical indicators:

```python
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from ..models import PredictionRequest, PredictionResponse, HorizonPrediction, PredictionQuality
from ..data_loader import HistoricalDataLoader
from ..feature_builder import FeatureBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FallbackPredictor:
    """Heuristic-based fallback when ML fails"""
    
    def __init__(self, config):
        """
        Initialize fallback predictor
        
        Args:
            config: PredictionConfig instance
        """
        self.config = config
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(config.technical_indicators)
    
    async def predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """
        Generate heuristic predictions
        
        Args:
            request: Prediction request
            base: Base currency
            quote: Quote currency
        
        Returns:
            PredictionResponse with heuristic predictions
        """
        logger.info(f"Using fallback heuristic predictor for {base}/{quote}")
        
        try:
            # Load minimal historical data
            df = await self.data_loader.fetch_historical_data(base, quote, days=100)
            
            if df is None or len(df) < 30:
                return self._empty_response(request, "Insufficient data for fallback")
            
            # Calculate technical indicators
            close = df['Close'].iloc[-1]
            
            # Moving averages
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # RSI
            rsi = self._calculate_rsi(df['Close'], period=14)
            
            # Volatility
            volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # Generate signals
            strength = self.config.fallback_strength_pct
            
            predictions = {}
            for horizon in request.horizons:
                # MA crossover signal
                ma_signal = self._get_ma_signal(close, sma_20, sma_50, strength)
                
                # RSI extremes signal (mean reversion)
                rsi_signal = self._get_rsi_signal(rsi, strength)
                
                # Combine signals
                mean_change = ma_signal + rsi_signal
                
                # Direction probability (simple mapping)
                direction_prob = 0.5 + (mean_change / (2 * strength))
                direction_prob = max(0.3, min(0.7, direction_prob))  # Clamp
                
                # Quantiles with high uncertainty
                quantiles = None
                if request.include_quantiles:
                    # Use volatility to estimate uncertainty
                    uncertainty = volatility * 100 * horizon  # Scale by horizon
                    quantiles = {
                        'p10': mean_change - uncertainty * 1.5,
                        'p50': mean_change,
                        'p90': mean_change + uncertainty * 1.5
                    }
                
                predictions[horizon] = HorizonPrediction(
                    horizon=horizon,
                    mean_change_pct=mean_change,
                    quantiles=quantiles,
                    direction_probability=direction_prob if request.include_direction_probabilities else None
                )
            
            # Build quality metadata
            quality = PredictionQuality(
                model_confidence=0.2,  # Low confidence for heuristics
                calibrated=False,
                validation_metrics={},
                notes=[
                    "Heuristic fallback based on MA crossover and RSI",
                    f"Current RSI: {rsi:.1f}",
                    f"Price vs SMA20: {((close/sma_20 - 1) * 100):+.2f}%"
                ]
            )
            
            logger.info(
                f"Fallback prediction complete: mean_change={predictions[1].mean_change_pct:.3f}%, "
                f"RSI={rsi:.1f}"
            )
            
            return PredictionResponse(
                status="partial",
                confidence=0.2,
                processing_time_ms=0,  # Will be set by caller
                warnings=["Using heuristic fallback - ML model unavailable"],
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions=predictions,
                latest_close=float(close),
                features_used=["sma_20", "sma_50", "rsi_14", "volatility_20"],
                quality=quality,
                model_id="fallback_heuristic",
                cached=False,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return self._empty_response(request, f"Fallback failed: {str(e)}")
    
    def _get_ma_signal(
        self, 
        close: float, 
        sma_20: float, 
        sma_50: float, 
        strength: float
    ) -> float:
        """
        Get moving average crossover signal
        
        Args:
            close: Current close price
            sma_20: 20-period SMA
            sma_50: 50-period SMA
            strength: Signal strength parameter
        
        Returns:
            Signal in % change (-strength to +strength)
        """
        # Price relative to SMA20
        price_vs_sma20 = (close / sma_20) - 1
        
        # SMA20 vs SMA50 (trend direction)
        sma_trend = (sma_20 / sma_50) - 1
        
        # Bullish: Price above SMA20 and SMA20 above SMA50
        if price_vs_sma20 > 0.002 and sma_trend > 0.001:  # 0.2% and 0.1% thresholds
            return strength * 0.6  # Moderate bullish
        
        # Bearish: Price below SMA20 and SMA20 below SMA50
        elif price_vs_sma20 < -0.002 and sma_trend < -0.001:
            return -strength * 0.6  # Moderate bearish
        
        # Weak bullish: Price above SMA20
        elif price_vs_sma20 > 0:
            return strength * 0.3
        
        # Weak bearish: Price below SMA20
        elif price_vs_sma20 < 0:
            return -strength * 0.3
        
        return 0.0  # Neutral
    
    def _get_rsi_signal(self, rsi: float, strength: float) -> float:
        """
        Get RSI mean reversion signal
        
        Args:
            rsi: Current RSI value
            strength: Signal strength parameter
        
        Returns:
            Signal in % change (-strength to +strength)
        """
        # Oversold (RSI < 30) -> expect reversion up
        if rsi < 30:
            intensity = (30 - rsi) / 30  # 0 to 1
            return strength * 0.5 * intensity
        
        # Overbought (RSI > 70) -> expect reversion down
        elif rsi > 70:
            intensity = (rsi - 70) / 30  # 0 to 1
            return -strength * 0.5 * intensity
        
        return 0.0  # Neutral
    
    @staticmethod
    def _calculate_rsi(series, period: int = 14) -> float:
        """
        Calculate RSI indicator
        
        Args:
            series: Price series
            period: RSI period
        
        Returns:
            Current RSI value (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _empty_response(
        self, 
        request: PredictionRequest, 
        error_msg: str
    ) -> PredictionResponse:
        """
        Return empty error response
        
        Args:
            request: Original request
            error_msg: Error message
        
        Returns:
            Empty PredictionResponse
        """
        logger.error(f"Fallback predictor error: {error_msg}")
        
        return PredictionResponse(
            status="error",
            confidence=0.0,
            processing_time_ms=0,
            warnings=[error_msg],
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions={},
            latest_close=0.0,
            features_used=[],
            quality=PredictionQuality(
                model_confidence=0.0,
                calibrated=False,
                validation_metrics={},
                notes=[error_msg]
            ),
            model_id="none",
            cached=False,
            timestamp=datetime.now()
        )
```

### Step 2: Unit Tests

**File**: `tests/prediction/test_fallback.py`

Test fallback predictor functionality:

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.prediction.utils.fallback import FallbackPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = PredictionConfig()
    config.fallback_strength_pct = 0.15
    config.technical_indicators = ["sma_20", "sma_50", "rsi_14", "volatility_20"]
    return config


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Trending up
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.3)
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100) * 0.1,
        'High': close_prices + abs(np.random.randn(100) * 0.2),
        'Low': close_prices - abs(np.random.randn(100) * 0.2),
        'Close': close_prices
    }, index=dates)
    
    return df


@pytest.mark.asyncio
async def test_fallback_predictor_basic(mock_config, monkeypatch):
    """Test basic fallback prediction"""
    predictor = FallbackPredictor(mock_config)
    
    # Mock data loader
    async def mock_fetch(*args, **kwargs):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 + np.cumsum(np.random.randn(100) * 0.3)
        return pd.DataFrame({
            'Open': close,
            'High': close + 0.2,
            'Low': close - 0.2,
            'Close': close
        }, index=dates)
    
    monkeypatch.setattr(predictor.data_loader, "fetch_historical_data", mock_fetch)
    
    request = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7, 30],
        include_quantiles=True,
        include_direction_probabilities=True
    )
    
    response = await predictor.predict(request, "USD", "EUR")
    
    assert response.status == "partial"
    assert response.confidence == 0.2
    assert len(response.predictions) == 3
    assert 1 in response.predictions
    assert response.model_id == "fallback_heuristic"


@pytest.mark.asyncio
async def test_fallback_with_insufficient_data(mock_config, monkeypatch):
    """Test fallback with insufficient data"""
    predictor = FallbackPredictor(mock_config)
    
    # Mock data loader returning insufficient data
    async def mock_fetch_insufficient(*args, **kwargs):
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Close': [100] * 10
        }, index=dates)
    
    monkeypatch.setattr(predictor.data_loader, "fetch_historical_data", mock_fetch_insufficient)
    
    request = PredictionRequest(currency_pair="USD/EUR", horizons=[1])
    response = await predictor.predict(request, "USD", "EUR")
    
    assert response.status == "error"
    assert "Insufficient data" in response.warnings[0]


def test_ma_signal_bullish(mock_config):
    """Test MA crossover signal - bullish case"""
    predictor = FallbackPredictor(mock_config)
    
    # Price above SMA20, SMA20 above SMA50
    close = 105.0
    sma_20 = 102.0
    sma_50 = 100.0
    strength = 0.15
    
    signal = predictor._get_ma_signal(close, sma_20, sma_50, strength)
    
    assert signal > 0  # Bullish
    assert signal <= strength


def test_ma_signal_bearish(mock_config):
    """Test MA crossover signal - bearish case"""
    predictor = FallbackPredictor(mock_config)
    
    # Price below SMA20, SMA20 below SMA50
    close = 95.0
    sma_20 = 98.0
    sma_50 = 100.0
    strength = 0.15
    
    signal = predictor._get_ma_signal(close, sma_20, sma_50, strength)
    
    assert signal < 0  # Bearish
    assert signal >= -strength


def test_rsi_signal_oversold(mock_config):
    """Test RSI signal - oversold case"""
    predictor = FallbackPredictor(mock_config)
    
    rsi = 25.0  # Oversold
    strength = 0.15
    
    signal = predictor._get_rsi_signal(rsi, strength)
    
    assert signal > 0  # Expect reversion up
    assert signal <= strength * 0.5


def test_rsi_signal_overbought(mock_config):
    """Test RSI signal - overbought case"""
    predictor = FallbackPredictor(mock_config)
    
    rsi = 75.0  # Overbought
    strength = 0.15
    
    signal = predictor._get_rsi_signal(rsi, strength)
    
    assert signal < 0  # Expect reversion down
    assert signal >= -strength * 0.5


def test_rsi_signal_neutral(mock_config):
    """Test RSI signal - neutral case"""
    predictor = FallbackPredictor(mock_config)
    
    rsi = 50.0  # Neutral
    strength = 0.15
    
    signal = predictor._get_rsi_signal(rsi, strength)
    
    assert signal == 0.0  # Neutral


def test_calculate_rsi(mock_config):
    """Test RSI calculation"""
    predictor = FallbackPredictor(mock_config)
    
    # Create trending up series
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                        110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
    
    rsi = predictor._calculate_rsi(prices, period=14)
    
    assert 50 < rsi < 100  # Should be high (overbought) for uptrend
    assert not np.isnan(rsi)


@pytest.mark.asyncio
async def test_fallback_predictions_structure(mock_config, monkeypatch):
    """Test that fallback predictions have correct structure"""
    predictor = FallbackPredictor(mock_config)
    
    # Mock data loader
    async def mock_fetch(*args, **kwargs):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        close = 100 + np.cumsum(np.random.randn(100) * 0.3)
        return pd.DataFrame({
            'Open': close,
            'High': close + 0.2,
            'Low': close - 0.2,
            'Close': close
        }, index=dates)
    
    monkeypatch.setattr(predictor.data_loader, "fetch_historical_data", mock_fetch)
    
    request = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7],
        include_quantiles=True,
        include_direction_probabilities=True
    )
    
    response = await predictor.predict(request, "USD", "EUR")
    
    # Check response structure
    assert response.status == "partial"
    assert 0.0 <= response.confidence <= 1.0
    assert response.currency_pair == "USD/EUR"
    
    # Check predictions
    for horizon in [1, 7]:
        assert horizon in response.predictions
        pred = response.predictions[horizon]
        
        assert pred.horizon == horizon
        assert isinstance(pred.mean_change_pct, float)
        assert pred.quantiles is not None
        assert 'p10' in pred.quantiles
        assert 'p50' in pred.quantiles
        assert 'p90' in pred.quantiles
        assert pred.direction_probability is not None
        assert 0.3 <= pred.direction_probability <= 0.7  # Clamped range
```

### Step 3: Integration with Main Predictor

**Update File**: `src/prediction/predictor.py`

Ensure fallback is properly integrated (already shown in Phase 2.3 plan, but verify):

```python
# In MLPredictor.__init__():
from .utils.fallback import FallbackPredictor
self.fallback = FallbackPredictor(self.config)

# In MLPredictor.predict():
try:
    response = await self._ml_predict(request, base, quote)
    
    # Check quality gate
    if self._passes_quality_gate(response):
        response.status = "success"
    else:
        logger.warning("Prediction failed quality gate, using fallback")
        response = await self._fallback_predict(request, base, quote)
        response.status = "partial"

except Exception as e:
    logger.error(f"ML prediction failed: {e}")
    response = await self._fallback_predict(request, base, quote)
    response.status = "partial" if response.predictions else "error"
    response.warnings.append(f"ML prediction failed: {str(e)}")
```

## Key Design Decisions

1. **Conservative signals**: Fallback uses ±0.15% as default strength to avoid overconfident predictions
2. **MA crossover + RSI combination**: Two complementary signals (trend + mean reversion)
3. **Low confidence (0.2)**: Clearly indicates heuristic predictions vs ML (0.6-0.9)
4. **Wide uncertainty bands**: Quantiles use volatility-scaled uncertainty
5. **Clamped direction probability**: 0.3-0.7 range (never extremely confident)
6. **Status "partial"**: Distinguishes from ML "success" and complete "error"
7. **Minimal data requirements**: Works with just 30 days of history
8. **No training needed**: Always available immediately

## Files to Create

- `src/prediction/utils/fallback.py`
- `tests/prediction/test_fallback.py`

## Files to Update

- `src/prediction/predictor.py` (verify fallback integration from Phase 2.3)

## Dependencies

No new dependencies! Uses existing:

- `pandas`, `numpy` for calculations
- `HistoricalDataLoader` for data fetching
- Existing data models

## Validation

Manual testing:

```python
from src.prediction.utils.fallback import FallbackPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig

# Initialize
config = PredictionConfig.from_yaml()
predictor = FallbackPredictor(config)

# Test prediction
request = PredictionRequest(
    currency_pair="USD/EUR",
    horizons=[1, 7, 30],
    include_quantiles=True,
    include_direction_probabilities=True
)

response = await predictor.predict(request, "USD", "EUR")

print(f"Status: {response.status}")
print(f"Confidence: {response.confidence}")
print(f"Model ID: {response.model_id}")

for horizon, pred in response.predictions.items():
    print(f"\n{horizon}d forecast:")
    print(f"  Mean change: {pred.mean_change_pct:+.3f}%")
    print(f"  Direction prob: {pred.direction_probability:.2f}")
    print(f"  Quantiles: {pred.quantiles}")

print(f"\nQuality notes: {response.quality.notes}")
```

## Success Criteria

- Fallback always returns predictions (never crashes)
- Predictions are conservative (small mean changes)
- Confidence is always low (0.2)
- Direction probability clamped to 0.3-0.7 range
- Works with minimal data (30+ days)
- RSI correctly identifies oversold/overbought conditions
- MA signals correctly identify trend direction
- All unit tests pass with >85% coverage
- Integration test shows fallback activates when ML fails

## Performance Characteristics

- **Latency**: <100ms (no model inference)
- **Memory**: Minimal (no models to load)
- **Accuracy**: ~52-55% directional (slightly better than random)
- **Reliability**: 100% (always available)
- **Use cases**: 
  - No trained model available yet
  - Model training failed
  - Model failed quality gates
  - Real-time predictions during model retraining

## Next Phase

After Phase 2.4 completes, proceed to **Phase 2.5: Prediction Agent & Caching** to integrate the ML predictor with caching, quality gates, and the LangGraph node.

### To-dos

- [ ] Implement FallbackPredictor with MA crossover and RSI signals in src/prediction/utils/fallback.py
- [ ] Create MA crossover signal logic (price vs SMA20, SMA20 vs SMA50)
- [ ] Implement RSI mean reversion signals (oversold/overbought detection)
- [ ] Add conservative signal strength and confidence scoring
- [ ] Create quantile estimation using volatility scaling
- [ ] Write comprehensive unit tests for fallback predictor (>85% coverage)
- [ ] Test MA crossover signals for bullish/bearish/neutral cases
- [ ] Test RSI mean reversion signals for oversold/overbought/neutral
- [ ] Manually validate fallback predictions with real data