<!-- f67714c1-a54f-4e8d-9617-16955f212afc fd9fbccc-e657-4d02-bf58-1ab06db90916 -->
# Phase 2.1: Data Pipeline & Feature Engineering

## Overview

Build the data pipeline that feeds the ML prediction models. This includes fetching historical OHLC data from yfinance and engineering technical features (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volatility). The pipeline supports both "price_only" mode (Phase 1) and "price_plus_intel" mode (Phase 2+) for future Market Intelligence integration.

## Implementation Steps

### Step 1: Data Contracts

**File**: `src/prediction/models.py`

Define all data structures for the prediction system:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class PredictionRequest:
    """Request for price prediction"""
    currency_pair: str
    horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    include_quantiles: bool = True
    include_direction_probabilities: bool = True
    max_age_hours: int = 1
    features_mode: str = "price_only"  # price_only | price_plus_intel
    correlation_id: Optional[str] = None

@dataclass
class HorizonPrediction:
    """Prediction for a single horizon"""
    horizon: int
    mean_change_pct: float  # Predicted % change
    quantiles: Optional[Dict[str, float]] = None  # {p10, p50, p90}
    direction_probability: Optional[float] = None  # P(up)

@dataclass
class PredictionQuality:
    """Quality metadata"""
    model_confidence: float  # 0-1
    calibrated: bool
    validation_metrics: Dict[str, float]
    notes: List[str] = field(default_factory=list)

@dataclass
class PredictionResponse:
    """Complete prediction response"""
    status: str  # success | partial | error
    confidence: float  # Overall confidence 0-1
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)
    
    # Data
    currency_pair: str
    horizons: List[int]
    predictions: Dict[int, HorizonPrediction]  # horizon -> prediction
    latest_close: float
    features_used: List[str]
    quality: PredictionQuality
    
    # Metadata
    model_id: str
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModelMetadata:
    """Model registry metadata"""
    model_id: str
    model_type: str  # lightgbm | lstm
    currency_pair: str
    trained_at: datetime
    version: str
    
    # Validation metrics
    validation_metrics: Dict[str, float]
    min_samples: int
    calibration_ok: bool
    
    # Feature info
    features_used: List[str]
    horizons: List[int]
    
    # Storage
    model_path: str
    scaler_path: Optional[str] = None
```

### Step 2: Configuration

**File**: `src/prediction/config.py`

```python
from dataclasses import dataclass, field
from typing import List
import os
import yaml

@dataclass
class PredictionConfig:
    """Prediction system configuration"""
    
    # General
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    cache_ttl_hours: int = 1
    
    # Backend
    predictor_backend: str = "lightgbm"  # lightgbm | lstm
    
    # Features
    features_mode: str = "price_only"  # price_only | price_plus_intel
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_5", "sma_20", "sma_50",
        "ema_12", "ema_26",
        "rsi_14",
        "macd", "macd_signal",
        "bb_upper", "bb_lower", "bb_middle",
        "atr_14",
        "volatility_20"
    ])
    
    # Data
    min_history_days: int = 100
    max_history_days: int = 365
    
    # Quality gates
    min_samples_required: int = 500
    min_validation_coverage: float = 0.85  # 85% quantile coverage
    min_directional_accuracy: float = 0.52  # Better than coin flip
    
    # Fallback
    enable_fallback_heuristics: bool = True
    fallback_strength_pct: float = 0.15  # ±0.15% default signal
    
    # Model registry
    model_registry_path: str = "models/prediction_registry.json"
    model_storage_dir: str = "models/prediction/"
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml"):
        """Load from YAML config file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                pred_config = full_config.get('prediction', {})
                return cls(**pred_config)
        return cls()
```

**Update**: `config.yaml` (add prediction section)

```yaml
prediction:
  prediction_horizons: [1, 7, 30]
  cache_ttl_hours: 1
  predictor_backend: "lightgbm"
  features_mode: "price_only"
  
  technical_indicators:
    - sma_5
    - sma_20
    - sma_50
    - ema_12
    - ema_26
    - rsi_14
    - macd
    - macd_signal
    - bb_upper
    - bb_lower
    - bb_middle
    - atr_14
    - volatility_20
  
  min_history_days: 100
  max_history_days: 365
  
  quality_gates:
    min_samples_required: 500
    min_validation_coverage: 0.85
    min_directional_accuracy: 0.52
  
  fallback:
    enable_heuristics: true
    strength_pct: 0.15
  
  model_registry_path: "models/prediction_registry.json"
  model_storage_dir: "models/prediction/"
```

### Step 3: Historical Data Loader

**File**: `src/prediction/data_loader.py`

Fetch OHLC data from yfinance:

```python
import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from src.utils.logging import get_logger

logger = get_logger(__name__)

class HistoricalDataLoader:
    """Load historical OHLC data for prediction"""
    
    @staticmethod
    def get_yahoo_symbol(base: str, quote: str) -> str:
        """Convert currency pair to Yahoo symbol"""
        return f"{base}{quote}=X"
    
    async def fetch_historical_data(
        self, 
        base: str, 
        quote: str, 
        days: int = 365
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from yfinance
        
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
        """
        symbol = self.get_yahoo_symbol(base, quote)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use yfinance to download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return None
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df[required_cols]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
```

### Step 4: Feature Builder

**File**: `src/prediction/feature_builder.py`

Build technical indicators from OHLC data:

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)

class FeatureBuilder:
    """Build features for ML models"""
    
    def __init__(self, indicators: List[str]):
        self.indicators = indicators
    
    def build_features(
        self, 
        df: pd.DataFrame,
        mode: str = "price_only",
        market_intel: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Build feature set from OHLC data
        
        Args:
            df: OHLC DataFrame
            mode: "price_only" or "price_plus_intel"
            market_intel: Optional market intelligence data
        
        Returns:
            DataFrame with features
        """
        features = df.copy()
        
        # Price-based features
        features = self._add_technical_indicators(features)
        
        # Market intelligence features (if enabled)
        if mode == "price_plus_intel" and market_intel:
            features = self._add_market_intel_features(features, market_intel)
        
        return features.dropna()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        
        # Simple Moving Averages
        if "sma_5" in self.indicators:
            df['sma_5'] = df['Close'].rolling(window=5).mean()
        if "sma_20" in self.indicators:
            df['sma_20'] = df['Close'].rolling(window=20).mean()
        if "sma_50" in self.indicators:
            df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        if "ema_12" in self.indicators:
            df['ema_12'] = df['Close'].ewm(span=12).mean()
        if "ema_26" in self.indicators:
            df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        if "rsi_14" in self.indicators:
            df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        if "macd" in self.indicators or "macd_signal" in self.indicators:
            macd, signal = self._calculate_macd(df['Close'])
            if "macd" in self.indicators:
                df['macd'] = macd
            if "macd_signal" in self.indicators:
                df['macd_signal'] = signal
        
        # Bollinger Bands
        if any(ind in self.indicators for ind in ["bb_upper", "bb_lower", "bb_middle"]):
            upper, middle, lower = self._calculate_bollinger_bands(df['Close'])
            if "bb_upper" in self.indicators:
                df['bb_upper'] = upper
            if "bb_middle" in self.indicators:
                df['bb_middle'] = middle
            if "bb_lower" in self.indicators:
                df['bb_lower'] = lower
        
        # ATR
        if "atr_14" in self.indicators:
            df['atr_14'] = self._calculate_atr(df, 14)
        
        # Volatility
        if "volatility_20" in self.indicators:
            df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df
    
    def _add_market_intel_features(
        self, 
        df: pd.DataFrame, 
        market_intel: Dict
    ) -> pd.DataFrame:
        """Add Market Intelligence features (for future use)"""
        
        # News sentiment
        if 'news' in market_intel:
            df['news_pair_bias'] = market_intel['news'].get('pair_bias', 0.0)
            df['news_confidence'] = self._confidence_to_numeric(
                market_intel['news'].get('confidence', 'low')
            )
        
        # Calendar events
        if 'calendar' in market_intel and market_intel['calendar'].get('next_high_event'):
            next_event = market_intel['calendar']['next_high_event']
            df['next_event_hours'] = next_event.get('proximity_minutes', 10000) / 60
        else:
            df['next_event_hours'] = 10000  # Far future
        
        return df
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    @staticmethod
    def _calculate_bollinger_bands(series: pd.Series, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _confidence_to_numeric(confidence: str) -> float:
        """Convert confidence string to numeric"""
        mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
        return mapping.get(confidence, 0.5)
    
    def build_targets(
        self, 
        df: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Build target variables (future returns)
        
        Args:
            df: DataFrame with Close prices
            horizons: List of forecast horizons in days
        
        Returns:
            DataFrame with target columns
        """
        targets = pd.DataFrame(index=df.index)
        
        for horizon in horizons:
            # Calculate percentage change
            targets[f'target_{horizon}d'] = (
                (df['Close'].shift(-horizon) - df['Close']) / df['Close'] * 100
            )
            
            # Direction (up=1, down=0)
            targets[f'direction_{horizon}d'] = (
                df['Close'].shift(-horizon) > df['Close']
            ).astype(int)
        
        return targets
```

### Step 5: Unit Tests

**File**: `tests/prediction/test_data_loader.py`

```python
import pytest
from src.prediction.data_loader import HistoricalDataLoader

@pytest.mark.asyncio
async def test_fetch_historical_data():
    """Test fetching historical data from yfinance"""
    loader = HistoricalDataLoader()
    
    # Test with major pair
    df = await loader.fetch_historical_data("USD", "EUR", days=90)
    
    assert df is not None
    assert len(df) > 50  # Should have at least 50 days
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])


@pytest.mark.asyncio
async def test_yahoo_symbol_conversion():
    """Test currency pair to Yahoo symbol conversion"""
    loader = HistoricalDataLoader()
    
    symbol = loader.get_yahoo_symbol("USD", "EUR")
    assert symbol == "USDEUR=X"
```

**File**: `tests/prediction/test_feature_builder.py`

```python
import pytest
import pandas as pd
import numpy as np
from src.prediction.feature_builder import FeatureBuilder

@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing"""
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100) * 0.1,
        'High': close_prices + abs(np.random.randn(100) * 0.2),
        'Low': close_prices - abs(np.random.randn(100) * 0.2),
        'Close': close_prices
    }, index=dates)
    
    return df


def test_build_features_price_only(sample_ohlc_data):
    """Test feature building in price_only mode"""
    indicators = ["sma_5", "sma_20", "rsi_14", "macd", "bb_upper", "atr_14", "volatility_20"]
    builder = FeatureBuilder(indicators)
    
    features = builder.build_features(sample_ohlc_data, mode="price_only")
    
    assert len(features) > 0
    assert 'sma_5' in features.columns
    assert 'sma_20' in features.columns
    assert 'rsi_14' in features.columns
    assert 'macd' in features.columns
    assert not features['rsi_14'].isna().all()


def test_calculate_rsi(sample_ohlc_data):
    """Test RSI calculation"""
    rsi = FeatureBuilder._calculate_rsi(sample_ohlc_data['Close'], period=14)
    
    # RSI should be between 0 and 100
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_build_targets(sample_ohlc_data):
    """Test target variable creation"""
    builder = FeatureBuilder([])
    horizons = [1, 7, 30]
    
    targets = builder.build_targets(sample_ohlc_data, horizons)
    
    assert 'target_1d' in targets.columns
    assert 'target_7d' in targets.columns
    assert 'target_30d' in targets.columns
    assert 'direction_1d' in targets.columns
```

## Key Design Decisions

1. **yfinance for data**: Simple, reliable, free historical FX data
2. **Pandas-based processing**: Standard library for financial data
3. **Modular feature builder**: Easy to add/remove indicators
4. **Future-ready for intelligence**: Stub for Market Intelligence features
5. **Target generation included**: Both regression (% change) and classification (direction)
6. **Quality validation**: Checks for sufficient data before proceeding

## Files to Create

- `src/prediction/__init__.py`
- `src/prediction/models.py`
- `src/prediction/config.py`
- `src/prediction/data_loader.py`
- `src/prediction/feature_builder.py`
- `tests/prediction/__init__.py`
- `tests/prediction/test_data_loader.py`
- `tests/prediction/test_feature_builder.py`

## Configuration Updates

Update `config.yaml` to add the `prediction:` section shown in Step 2.

## Dependencies

Add to `pyproject.toml`:

```toml
yfinance = ">=0.2.0"
pandas = ">=2.0.0"
numpy = ">=1.24.0"
```

## Validation

Manual testing:

```python
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.config import PredictionConfig

# Load config
config = PredictionConfig.from_yaml()

# Load data
loader = HistoricalDataLoader()
df = await loader.fetch_historical_data("USD", "EUR", days=90)
print(f"Loaded {len(df)} days of data")

# Build features
builder = FeatureBuilder(config.technical_indicators)
features = builder.build_features(df, mode="price_only")
print(f"Features shape: {features.shape}")
print(f"Columns: {features.columns.tolist()}")

# Build targets
targets = builder.build_targets(df, horizons=[1, 7, 30])
print(f"Targets shape: {targets.shape}")
```

## Success Criteria

- Historical data fetches successfully from yfinance for major pairs
- All technical indicators calculate correctly without errors
- Feature DataFrame has no NaN values (after dropna)
- Target variables correctly represent future returns
- Unit tests pass with >80% coverage
- Sample EUR/USD data loads and processes in <5 seconds

## Next Phase

After Phase 2.1 completes, proceed to **Phase 2.2: Model Registry & Storage** to implement JSON-based model metadata tracking and pickle-based model storage.

### To-dos

- [ ] Create data contracts (PredictionRequest, HorizonPrediction, PredictionResponse) in src/prediction/models.py
- [ ] Implement HistoricalDataLoader with yfinance integration for OHLC data fetching
- [ ] Create FeatureBuilder with technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, volatility)
- [ ] Add target variable generation for multiple horizons
- [ ] Implement configuration management with PredictionConfig class
- [ ] Write comprehensive unit tests for data loader and feature builder (>80% coverage)
- [ ] Manually validate data pipeline with real EUR/USD data