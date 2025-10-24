<!-- 537cf47b-219b-4e1b-b98e-4b6d556a014a 7a0e11ea-da21-467e-a2ff-6c9a7464fc52 -->
# Price Prediction Agent Implementation Plan

## Overview

Build a lean, maintainable ML prediction system for FX price forecasting with:

- Pluggable backends: LightGBM (Phase 1) + LSTM (Phase 2)
- Multi-horizon forecasts (1, 7, 30 days)
- Quantile predictions for uncertainty
- Quality gates and graceful degradation
- Integration with existing agentic system
- Future-ready for Market Intelligence features

## Architecture Design

### Core Components

```
src/prediction/
├── __init__.py                    # Public API
├── config.py                      # Configuration
├── predictor.py                   # Main predictor (caching, quality gates)
├── feature_builder.py             # Feature engineering
├── data_loader.py                 # Historical OHLC fetching (yfinance)
├── registry.py                    # Model registry (JSON + pickle)
├── models.py                      # Data contracts
├── backends/
│   ├── __init__.py
│   ├── base.py                    # Base predictor interface
│   ├── lightgbm_backend.py        # LightGBM implementation
│   └── lstm_backend.py            # LSTM implementation (Phase 2)
└── utils/
    ├── __init__.py
    ├── calibration.py             # Quality metrics, calibration
    └── fallback.py                # Heuristic fallbacks

src/agentic/nodes/
└── prediction.py                  # NEW: Prediction agent node

tests/prediction/
├── test_feature_builder.py
├── test_predictor.py
├── test_backends.py
└── test_integration.py
```

## Phase 1: Core Infrastructure

### 1. Data Contracts

**File: `src/prediction/models.py`**

Define all data structures:

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

### 2. Configuration

**File: `src/prediction/config.py`**

```python
from dataclasses import dataclass
from typing import List, Dict
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

**Update: `config.yaml`** (add prediction section)

```yaml
prediction:
  prediction_horizons: [1, 7, 30]
  cache_ttl_hours: 1
  predictor_backend: "lightgbm"  # lightgbm | lstm
  features_mode: "price_only"  # price_only | price_plus_intel
  
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

### 3. Data Loading

**File: `src/prediction/data_loader.py`**

```python
import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

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

### 4. Feature Engineering

**File: `src/prediction/feature_builder.py`**

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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

### 5. Model Registry

**File: `src/prediction/registry.py`**

```python
import json
import os
import pickle
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

from .models import ModelMetadata

class ModelRegistry:
    """Simple JSON-based model registry"""
    
    def __init__(self, registry_path: str, storage_dir: str):
        self.registry_path = registry_path
        self.storage_dir = storage_dir
        
        # Ensure directories exist
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from JSON"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save registry to JSON"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(
        self, 
        metadata: ModelMetadata,
        model_obj: any,
        scaler_obj: Optional[any] = None
    ):
        """Register a new model"""
        
        # Save model files
        model_path = os.path.join(self.storage_dir, f"{metadata.model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_obj, f)
        
        metadata.model_path = model_path
        
        # Save scaler if provided
        if scaler_obj:
            scaler_path = os.path.join(self.storage_dir, f"{metadata.model_id}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_obj, f)
            metadata.scaler_path = scaler_path
        
        # Add to registry
        self.registry[metadata.model_id] = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "currency_pair": metadata.currency_pair,
            "trained_at": metadata.trained_at.isoformat(),
            "version": metadata.version,
            "validation_metrics": metadata.validation_metrics,
            "min_samples": metadata.min_samples,
            "calibration_ok": metadata.calibration_ok,
            "features_used": metadata.features_used,
            "horizons": metadata.horizons,
            "model_path": metadata.model_path,
            "scaler_path": metadata.scaler_path
        }
        
        self._save_registry()
    
    def get_model(self, currency_pair: str, model_type: str = "lightgbm") -> Optional[Dict]:
        """Get latest model for currency pair"""
        
        # Find matching models
        matches = [
            m for m in self.registry.values()
            if m['currency_pair'] == currency_pair and m['model_type'] == model_type
        ]
        
        if not matches:
            return None
        
        # Return most recent
        return max(matches, key=lambda m: m['trained_at'])
    
    def load_model_objects(self, model_metadata: Dict) -> tuple:
        """Load model and scaler from disk"""
        
        with open(model_metadata['model_path'], 'rb') as f:
            model = pickle.load(f)
        
        scaler = None
        if model_metadata.get('scaler_path'):
            with open(model_metadata['scaler_path'], 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler
    
    def list_models(self, currency_pair: Optional[str] = None) -> List[Dict]:
        """List all models, optionally filtered by currency pair"""
        models = list(self.registry.values())
        
        if currency_pair:
            models = [m for m in models if m['currency_pair'] == currency_pair]
        
        return sorted(models, key=lambda m: m['trained_at'], reverse=True)
```

## Phase 2: ML Backends

### 6. Base Backend Interface

**File: `src/prediction/backends/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class BasePredictorBackend(ABC):
    """Base interface for prediction backends"""
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int]
    ) -> Dict:
        """
        Train model
        
        Returns: validation_metrics dict
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int],
        include_quantiles: bool = True
    ) -> Dict[int, Dict]:
        """
        Make predictions
        
        Returns: {
            horizon: {
                "mean_change": float,
                "quantiles": {p10, p50, p90} (if enabled),
                "direction_prob": float
            }
        }
        """
        pass
    
    @abstractmethod
    def get_model_confidence(self) -> float:
        """Return overall model confidence 0-1"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass
```

### 7. LightGBM Backend

**File: `src/prediction/backends/lightgbm_backend.py`**

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

from .base import BasePredictorBackend

logger = logging.getLogger(__name__)

class LightGBMBackend(BasePredictorBackend):
    """LightGBM-based predictor"""
    
    def __init__(self):
        self.models = {}  # horizon -> model
        self.quantile_models = {}  # horizon -> {p10, p50, p90}
        self.direction_models = {}  # horizon -> classifier
        self.scaler = StandardScaler()
        self.validation_metrics = {}
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None
    ) -> Dict:
        """Train LightGBM models for each horizon"""
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
        
        metrics = {}
        
        for horizon in horizons:
            logger.info(f"Training LightGBM for {horizon}d horizon")
            
            target_col = f'target_{horizon}d'
            direction_col = f'direction_{horizon}d'
            
            if target_col not in y_train.columns:
                logger.warning(f"Target {target_col} not found, skipping")
                continue
            
            # Align X and y (drop NaN targets)
            valid_idx = ~y_train[target_col].isna()
            X_h = X_scaled[valid_idx]
            y_h = y_train[target_col][valid_idx]
            y_dir = y_train[direction_col][valid_idx]
            
            if len(X_h) < 100:
                logger.warning(f"Insufficient samples for {horizon}d: {len(X_h)}")
                continue
            
            # Split train/validation
            split_idx = int(len(X_h) * 0.8)
            X_tr, X_val = X_h[:split_idx], X_h[split_idx:]
            y_tr, y_val = y_h[:split_idx], y_h[split_idx:]
            y_dir_tr, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
            
            # Train mean regression
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models[horizon] = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Train quantile models (p10, p50, p90)
            self.quantile_models[horizon] = {}
            for quantile in [0.1, 0.5, 0.9]:
                q_params = params.copy()
                q_params['objective'] = 'quantile'
                q_params['alpha'] = quantile
                
                self.quantile_models[horizon][f'p{int(quantile*100)}'] = lgb.train(
                    q_params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
            
            # Train direction classifier
            dir_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1
            }
            
            dir_train_data = lgb.Dataset(X_tr, label=y_dir_tr)
            dir_val_data = lgb.Dataset(X_val, label=y_dir_val, reference=dir_train_data)
            
            self.direction_models[horizon] = lgb.train(
                dir_params,
                dir_train_data,
                num_boost_round=100,
                valid_sets=[dir_val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Calculate validation metrics
            y_pred = self.models[horizon].predict(X_val)
            y_dir_pred = self.direction_models[horizon].predict(X_val)
            
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mae = np.mean(np.abs(y_val - y_pred))
            dir_accuracy = np.mean((y_dir_pred > 0.5) == y_dir_val)
            
            # Check quantile coverage
            q10_pred = self.quantile_models[horizon]['p10'].predict(X_val)
            q90_pred = self.quantile_models[horizon]['p90'].predict(X_val)
            coverage = np.mean((y_val >= q10_pred) & (y_val <= q90_pred))
            
            metrics[f'{horizon}d'] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'directional_accuracy': float(dir_accuracy),
                'quantile_coverage': float(coverage),
                'n_samples': len(X_val)
            }
            
            logger.info(f"  {horizon}d - RMSE: {rmse:.4f}, Dir Acc: {dir_accuracy:.3f}, Coverage: {coverage:.3f}")
        
        self.validation_metrics = metrics
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int],
        include_quantiles: bool = True
    ) -> Dict[int, Dict]:
        """Make predictions"""
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        predictions = {}
        
        for horizon in horizons:
            if horizon not in self.models:
                logger.warning(f"No model for {horizon}d horizon")
                continue
            
            pred = {}
            
            # Mean prediction
            pred['mean_change'] = float(self.models[horizon].predict(X_scaled)[0])
            
            # Quantiles
            if include_quantiles and horizon in self.quantile_models:
                pred['quantiles'] = {
                    'p10': float(self.quantile_models[horizon]['p10'].predict(X_scaled)[0]),
                    'p50': float(self.quantile_models[horizon]['p50'].predict(X_scaled)[0]),
                    'p90': float(self.quantile_models[horizon]['p90'].predict(X_scaled)[0])
                }
            
            # Direction probability
            if horizon in self.direction_models:
                pred['direction_prob'] = float(self.direction_models[horizon].predict(X_scaled)[0])
            
            predictions[horizon] = pred
        
        return predictions
    
    def get_model_confidence(self) -> float:
        """Calculate overall model confidence"""
        if not self.validation_metrics:
            return 0.0
        
        # Average directional accuracy across horizons
        accuracies = [
            m['directional_accuracy'] 
            for m in self.validation_metrics.values()
        ]
        
        if not accuracies:
            return 0.0
        
        avg_accuracy = np.mean(accuracies)
        
        # Map accuracy to confidence (0.5 accuracy -> 0.0 confidence, 1.0 accuracy -> 1.0 confidence)
        confidence = max(0.0, (avg_accuracy - 0.5) * 2)
        
        return float(confidence)
    
    def save(self, path: str):
        """Save models"""
        import pickle
        
        state = {
            'models': {h: m.model_to_string() for h, m in self.models.items()},
            'quantile_models': {
                h: {q: m.model_to_string() for q, m in qm.items()}
                for h, qm in self.quantile_models.items()
            },
            'direction_models': {h: m.model_to_string() for h, m in self.direction_models.items()},
            'scaler': self.scaler,
            'validation_metrics': self.validation_metrics
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load models"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.models = {h: lgb.Booster(model_str=s) for h, s in state['models'].items()}
        self.quantile_models = {
            h: {q: lgb.Booster(model_str=s) for q, s in qm.items()}
            for h, qm in state['quantile_models'].items()
        }
        self.direction_models = {h: lgb.Booster(model_str=s) for h, s in state['direction_models'].items()}
        self.scaler = state['scaler']
        self.validation_metrics = state['validation_metrics']
```

### 8. LSTM Backend (Phase 2 - Stub for now)

**File: `src/prediction/backends/lstm_backend.py`**

```python
from .base import BasePredictorBackend
import logging

logger = logging.getLogger(__name__)

class LSTMBackend(BasePredictorBackend):
    """LSTM-based predictor (Phase 2 implementation)"""
    
    def __init__(self):
        logger.info("LSTM backend not yet implemented - will be added in Phase 2")
        raise NotImplementedError("LSTM backend coming in Phase 2")
    
    def train(self, X_train, y_train, horizons):
        raise NotImplementedError()
    
    def predict(self, X, horizons, include_quantiles=True):
        raise NotImplementedError()
    
    def get_model_confidence(self):
        raise NotImplementedError()
    
    def save(self, path):
        raise NotImplementedError()
    
    def load(self, path):
        raise NotImplementedError()
```

## Phase 3: Main Predictor

### 9. Predictor with Caching and Quality Gates

**File: `src/prediction/predictor.py`**

```python
import time
import hashlib
import json
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from .models import PredictionRequest, PredictionResponse, HorizonPrediction, PredictionQuality
from .data_loader import HistoricalDataLoader
from .feature_builder import FeatureBuilder
from .registry import ModelRegistry
from .config import PredictionConfig
from .backends.lightgbm_backend import LightGBMBackend
from .utils.fallback import FallbackPredictor

logger = logging.getLogger(__name__)

class MLPredictor:
    """Main prediction service with caching and quality gates"""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig.from_yaml()
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(self.config.technical_indicators)
        self.registry = ModelRegistry(
            self.config.model_registry_path,
            self.config.model_storage_dir
        )
        self.fallback = FallbackPredictor(self.config)
        self.cache = {}  # Simple in-memory cache
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction with caching and quality gates"""
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            cached_response, cached_time = self.cache[cache_key]
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            if age_hours <= request.max_age_hours:
                logger.info(f"Returning cached prediction (age: {age_hours:.1f}h)")
                cached_response.cached = True
                return cached_response
        
        # Parse currency pair
        base, quote = request.currency_pair.split('/')
        
        # Try ML prediction
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
        
        response.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache response
        self.cache[cache_key] = (response, datetime.now())
        
        return response
    
    async def _ml_predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """ML-based prediction"""
        
        # Load historical data
        df = await self.data_loader.fetch_historical_data(
            base, quote, days=self.config.max_history_days
        )
        
        if df is None or len(df) < self.config.min_history_days:
            raise ValueError(f"Insufficient historical data: {len(df) if df is not None else 0} days")
        
        # Build features
        features_df = self.feature_builder.build_features(
            df,
            mode=request.features_mode,
            market_intel=None  # TODO: Pass market intel when available
        )
        
        if len(features_df) < 50:
            raise ValueError(f"Insufficient feature data: {len(features_df)} rows")
        
        # Get latest features (most recent row)
        latest_features = features_df.iloc[[-1]]
        latest_close = df['Close'].iloc[-1]
        
        # Load model from registry
        model_metadata = self.registry.get_model(
            request.currency_pair,
            model_type=self.config.predictor_backend
        )
        
        if not model_metadata:
            raise ValueError(f"No model found for {request.currency_pair}")
        
        # Load model objects
        model_obj, scaler_obj = self.registry.load_model_objects(model_metadata)
        
        # Create backend
        backend = self._create_backend(model_metadata['model_type'])
        backend.models = model_obj
        backend.scaler = scaler_obj
        backend.validation_metrics = model_metadata['validation_metrics']
        
        # Make predictions
        raw_predictions = backend.predict(
            latest_features,
            request.horizons,
            include_quantiles=request.include_quantiles
        )
        
        # Build response
        predictions = {}
        for horizon, pred_data in raw_predictions.items():
            predictions[horizon] = HorizonPrediction(
                horizon=horizon,
                mean_change_pct=pred_data['mean_change'],
                quantiles=pred_data.get('quantiles'),
                direction_probability=pred_data.get('direction_prob')
            )
        
        # Quality metadata
        quality = PredictionQuality(
            model_confidence=backend.get_model_confidence(),
            calibrated=model_metadata['calibration_ok'],
            validation_metrics=model_metadata['validation_metrics'],
            notes=[]
        )
        
        response = PredictionResponse(
            status="success",
            confidence=quality.model_confidence,
            processing_time_ms=0,  # Will be set by caller
            currency_pair=request.currency_pair,
            horizons=request.horizons,
            predictions=predictions,
            latest_close=float(latest_close),
            features_used=model_metadata['features_used'],
            quality=quality,
            model_id=model_metadata['model_id'],
            cached=False
        )
        
        return response
    
    async def _fallback_predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """Fallback heuristic-based prediction"""
        
        logger.info("Using fallback heuristic predictor")
        
        return await self.fallback.predict(request, base, quote)
    
    def _create_backend(self, model_type: str):
        """Create backend instance"""
        if model_type == "lightgbm":
            return LightGBMBackend()
        elif model_type == "lstm":
            from .backends.lstm_backend import LSTMBackend
            return LSTMBackend()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _passes_quality_gate(self, response: PredictionResponse) -> bool:
        """Check if prediction passes quality gates"""
        
        if not response.quality:
            return False
        
        # Check confidence threshold
        if response.confidence < 0.3:
            logger.warning(f"Low confidence: {response.confidence}")
            return False
        
        # Check calibration
        if not response.quality.calibrated:
            logger.warning("Model not calibrated")
            return False
        
        # Check sample size
        metrics = response.quality.validation_metrics
        if metrics:
            min_samples = min(m.get('n_samples', 0) for m in metrics.values())
            if min_samples < self.config.min_samples_required:
                logger.warning(f"Insufficient validation samples: {min_samples}")
                return False
        
        return True
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key"""
        key_data = {
            'pair': request.currency_pair,
            'horizons': sorted(request.horizons),
            'mode': request.features_mode
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
```

### 10. Fallback Predictor

**File: `src/prediction/utils/fallback.py`**

```python
import logging
from typing import Dict
from ..models import PredictionRequest, PredictionResponse, HorizonPrediction, PredictionQuality
from ..data_loader import HistoricalDataLoader
from ..feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

class FallbackPredictor:
    """Heuristic-based fallback when ML fails"""
    
    def __init__(self, config):
        self.config = config
        self.data_loader = HistoricalDataLoader()
        self.feature_builder = FeatureBuilder(config.technical_indicators)
    
    async def predict(
        self, 
        request: PredictionRequest,
        base: str,
        quote: str
    ) -> PredictionResponse:
        """Generate heuristic predictions"""
        
        try:
            # Load minimal historical data
            df = await self.data_loader.fetch_historical_data(base, quote, days=100)
            
            if df is None or len(df) < 30:
                return self._empty_response(request, "Insufficient data for fallback")
            
            # Calculate simple technical signals
            close = df['Close'].iloc[-1]
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            strength = self.config.fallback_strength_pct
            
            predictions = {}
            for horizon in request.horizons:
                # MA crossover signal
                ma_signal = 0.0
                if close > sma_20:
                    ma_signal = strength / 2
                elif close < sma_20:
                    ma_signal = -strength / 2
                
                # RSI extremes
                rsi_signal = 0.0
                if rsi > 70:  # Overbought -> reversion
                    rsi_signal = -strength / 2
                elif rsi < 30:  # Oversold -> reversion
                    rsi_signal = strength / 2
                
                # Combine signals
                mean_change = ma_signal + rsi_signal
                
                # Direction probability (simple sigmoid-like mapping)
                direction_prob = 0.5 + (mean_change / (2 * strength))
                direction_prob = max(0.3, min(0.7, direction_prob))  # Clamp to reasonable range
                
                # Wide symmetric quantiles (high uncertainty)
                quantiles = {
                    'p10': mean_change - strength * 2,
                    'p50': mean_change,
                    'p90': mean_change + strength * 2
                } if request.include_quantiles else None
                
                predictions[horizon] = HorizonPrediction(
                    horizon=horizon,
                    mean_change_pct=mean_change,
                    quantiles=quantiles,
                    direction_probability=direction_prob if request.include_direction_probabilities else None
                )
            
            quality = PredictionQuality(
                model_confidence=0.2,  # Low confidence for heuristics
                calibrated=False,
                validation_metrics={},
                notes=["Heuristic fallback based on MA crossover and RSI"]
            )
            
            return PredictionResponse(
                status="partial",
                confidence=0.2,
                processing_time_ms=0,
                warnings=["Using heuristic fallback - ML model unavailable"],
                currency_pair=request.currency_pair,
                horizons=request.horizons,
                predictions=predictions,
                latest_close=float(close),
                features_used=["sma_20", "sma_50", "rsi_14"],
                quality=quality,
                model_id="fallback_heuristic",
                cached=False
            )
        
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return self._empty_response(request, f"Fallback failed: {str(e)}")
    
    def _empty_response(self, request: PredictionRequest, error_msg: str) -> PredictionResponse:
        """Return empty error response"""
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
            cached=False
        )
```

## Phase 4: Agent Integration

### 11. Prediction Agent Node

**File: `src/agentic/nodes/prediction.py`**

```python
import logging
from typing import Optional

from ..state import AgentGraphState
from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig

logger = logging.getLogger(__name__)

class PredictionAgent:
    """ML-based price prediction agent"""
    
    def __init__(self, predictor: Optional[MLPredictor] = None):
        self.predictor = predictor or MLPredictor(PredictionConfig.from_yaml())
    
    async def __call__(self, state: AgentGraphState) -> AgentGraphState:
        """Execute prediction and update state"""
        
        request = state.request
        correlation_id = state.meta.correlation_id or "n/a"
        
        logger.info(f"[{correlation_id}] Starting price prediction for {request.currency_pair}")
        
        # Build prediction request
        pred_request = PredictionRequest(
            currency_pair=request.currency_pair,
            horizons=[1, 7, request.timeframe_days],  # 1d, 7d, and user's timeframe
            include_quantiles=True,
            include_direction_probabilities=True,
            max_age_hours=1,
            features_mode="price_only",  # TODO: Switch to price_plus_intel when ready
            correlation_id=correlation_id
        )
        
        # Get prediction
        try:
            pred_response = await self.predictor.predict(pred_request)
            
            # Update market state with predictions
            state.market.ml_forecasts = {
                str(h): {
                    "mean_change_pct": p.mean_change_pct,
                    "quantiles": p.quantiles,
                    "direction_prob": p.direction_probability
                }
                for h, p in pred_response.predictions.items()
            }
            
            # Set primary forecast (user's timeframe)
            if request.timeframe_days in pred_response.predictions:
                primary = pred_response.predictions[request.timeframe_days]
                state.market.primary_forecast = {
                    "horizon_days": request.timeframe_days,
                    "mean_change_pct": primary.mean_change_pct,
                    "direction_prob": primary.direction_probability,
                    "confidence": pred_response.confidence
                }
                state.market.primary_forecast_horizon = request.timeframe_days
            
            # Add to errors if partial/error
            if pred_response.status != "success":
                state.market.errors.extend(pred_response.warnings)
            
            # Add data source notes
            state.market.data_source_notes.append(
                f"ML prediction: {pred_response.model_id} (confidence: {pred_response.confidence:.2f})"
            )
            
            if pred_response.cached:
                state.market.data_source_notes.append("Prediction served from cache")
            
            logger.info(f"[{correlation_id}] Prediction complete: {pred_response.status}")
        
        except Exception as e:
            logger.error(f"[{correlation_id}] Prediction failed: {e}")
            state.market.errors.append(f"Price prediction failed: {str(e)}")
        
        return state
```

### 12. Update Agent Graph

**File: `src/agentic/graph.py`** (MODIFY - add prediction node)

```python
# Add to imports
from .nodes.prediction import PredictionAgent

# In build_graph function, add prediction node:
prediction_agent = PredictionAgent()

# Add to graph workflow (after market analysis, before decision)
workflow.add_node("prediction", prediction_agent)
workflow.add_edge("market", "prediction")
workflow.add_edge("prediction", "decision")
```

## Phase 5: Training Script

### 13. Model Training Script

**File: `scripts/train_prediction_model.py`**

```python
#!/usr/bin/env python3
"""
Train prediction model for a currency pair

Usage:
  python scripts/train_prediction_model.py --pair USD/EUR --backend lightgbm
"""

import argparse
import asyncio
import logging
from datetime import datetime

from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.registry import ModelRegistry
from src.prediction.models import ModelMetadata
from src.prediction.config import PredictionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train_model(currency_pair: str, backend_type: str = "lightgbm"):
    """Train and register a prediction model"""
    
    config = PredictionConfig.from_yaml()
    
    logger.info(f"Training {backend_type} model for {currency_pair}")
    
    # Parse pair
    base, quote = currency_pair.split('/')
    
    # Load historical data
    loader = HistoricalDataLoader()
    df = await loader.fetch_historical_data(base, quote, days=config.max_history_days)
    
    if df is None or len(df) < config.min_history_days:
        logger.error(f"Insufficient data: {len(df) if df is not None else 0} days")
        return
    
    logger.info(f"Loaded {len(df)} days of historical data")
    
    # Build features
    feature_builder = FeatureBuilder(config.technical_indicators)
    features_df = feature_builder.build_features(df, mode="price_only")
    
    logger.info(f"Built features: {len(features_df)} rows, {len(features_df.columns)} features")
    
    # Build targets
    targets_df = feature_builder.build_targets(df, config.prediction_horizons)
    
    # Align features and targets
    common_idx = features_df.index.intersection(targets_df.index)
    X = features_df.loc[common_idx]
    y = targets_df.loc[common_idx]
    
    logger.info(f"Training data: {len(X)} samples")
    
    if len(X) < config.min_samples_required:
        logger.error(f"Insufficient samples: {len(X)} < {config.min_samples_required}")
        return
    
    # Create backend and train
    if backend_type == "lightgbm":
        backend = LightGBMBackend()
    else:
        logger.error(f"Unknown backend: {backend_type}")
        return
    
    logger.info("Training models...")
    validation_metrics = backend.train(X, y, config.prediction_horizons)
    
    logger.info("Validation metrics:")
    for horizon, metrics in validation_metrics.items():
        logger.info(f"  {horizon}: {metrics}")
    
    # Check quality gates
    calibration_ok = all(
        m['quantile_coverage'] >= config.min_validation_coverage and
        m['directional_accuracy'] >= config.min_directional_accuracy
        for m in validation_metrics.values()
    )
    
    if not calibration_ok:
        logger.warning("Model did not pass calibration checks!")
    
    # Generate model ID
    model_id = f"{currency_pair.replace('/', '')}__{backend_type}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create metadata
    metadata = ModelMetadata(
        model_id=model_id,
        model_type=backend_type,
        currency_pair=currency_pair,
        trained_at=datetime.now(),
        version="1.0",
        validation_metrics=validation_metrics,
        min_samples=len(X),
        calibration_ok=calibration_ok,
        features_used=list(features_df.columns),
        horizons=config.prediction_horizons,
        model_path=""  # Will be set by registry
    )
    
    # Register model
    registry = ModelRegistry(config.model_registry_path, config.model_storage_dir)
    registry.register_model(metadata, backend, backend.scaler)
    
    logger.info(f"Model registered: {model_id}")
    logger.info(f"Confidence: {backend.get_model_confidence():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Train prediction model")
    parser.add_argument("--pair", required=True, help="Currency pair (e.g., USD/EUR)")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "lstm"])
    
    args = parser.parse_args()
    
    asyncio.run(train_model(args.pair, args.backend))

if __name__ == "__main__":
    main()
```

## Testing Plan

### Unit Tests

1. **test_feature_builder.py**: Test technical indicator calculations
2. **test_data_loader.py**: Test historical data fetching
3. **test_lightgbm_backend.py**: Test LightGBM training and prediction
4. **test_registry.py**: Test model save/load
5. **test_predictor.py**: Test caching, quality gates
6. **test_fallback.py**: Test heuristic predictor

### Integration Tests

1. **test_end_to_end.py**: Full prediction pipeline
2. **test_agent_integration.py**: Prediction agent in workflow

## Implementation Todos

- [ ] Create data contracts (`src/prediction/models.py`)
- [ ] Create configuration (`src/prediction/config.py`, update `config.yaml`)
- [ ] Implement historical data loader (`src/prediction/data_loader.py`)
- [ ] Implement feature builder (`src/prediction/feature_builder.py`)
- [ ] Implement model registry (`src/prediction/registry.py`)
- [ ] Implement base backend interface (`src/prediction/backends/base.py`)
- [ ] Implement LightGBM backend (`src/prediction/backends/lightgbm_backend.py`)
- [ ] Create LSTM backend stub (`src/prediction/backends/lstm_backend.py`)
- [ ] Implement fallback predictor (`src/prediction/utils/fallback.py`)
- [ ] Implement main predictor (`src/prediction/predictor.py`)
- [ ] Create prediction agent node (`src/agentic/nodes/prediction.py`)
- [ ] Update agent graph (`src/agentic/graph.py`)
- [ ] Create training script (`scripts/train_prediction_model.py`)
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Add lightgbm to dependencies
- [ ] Create example usage script

### To-dos

- [ ] Create data contracts (PredictionRequest, PredictionResponse, ModelMetadata) in src/prediction/models.py
- [ ] Create PredictionConfig in src/prediction/config.py and add prediction section to config.yaml
- [ ] Implement HistoricalDataLoader using yfinance in src/prediction/data_loader.py
- [ ] Implement FeatureBuilder with technical indicators in src/prediction/feature_builder.py
- [ ] Implement ModelRegistry (JSON + pickle) in src/prediction/registry.py
- [ ] Create BasePredictorBackend interface in src/prediction/backends/base.py
- [ ] Implement LightGBMBackend with quantile regression in src/prediction/backends/lightgbm_backend.py
- [ ] Create LSTM backend stub in src/prediction/backends/lstm_backend.py
- [ ] Implement FallbackPredictor with heuristics in src/prediction/utils/fallback.py
- [ ] Implement MLPredictor with caching and quality gates in src/prediction/predictor.py
- [ ] Create PredictionAgent node in src/agentic/nodes/prediction.py
- [ ] Update agent graph to include prediction node in src/agentic/graph.py
- [ ] Create training script in scripts/train_prediction_model.py
- [ ] Add lightgbm, scikit-learn to pyproject.toml dependencies
- [ ] Write unit tests for feature_builder, data_loader, backends, registry
- [ ] Write integration tests for end-to-end prediction pipeline