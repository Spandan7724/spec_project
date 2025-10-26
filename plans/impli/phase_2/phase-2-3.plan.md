<!-- f67714c1-a54f-4e8d-9617-16955f212afc a2f7c1dc-fee5-408e-b953-70ec229d7c82 -->
# Phase 2.3: LightGBM Backend + SHAP Explainability

## Overview

Implement the machine learning prediction engine using LightGBM for gradient boosting. This includes training quantile regression models for uncertainty estimates, binary classifiers for direction probability, and SHAP integration for model explainability. The backend provides the core forecasting capability for the Price Prediction Agent.

## Implementation Steps

### Step 1: Base Predictor Interface

**File**: `src/prediction/backends/base.py`

Define the interface that all prediction backends must implement:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class BasePredictorBackend(ABC):
    """Base interface for prediction backends"""
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Train model
        
        Args:
            X_train: Training features
            y_train: Training targets
            horizons: Forecast horizons in days
            params: Optional model parameters
        
        Returns:
            validation_metrics dict
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
        
        Args:
            X: Feature DataFrame
            horizons: Forecast horizons
            include_quantiles: Whether to include uncertainty estimates
        
        Returns:
            {
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
        """
        Return overall model confidence 0-1
        
        Based on validation metrics (directional accuracy)
        """
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

### Step 2: LightGBM Backend Implementation

**File**: `src/prediction/backends/lightgbm_backend.py`

Implement LightGBM with quantile regression and direction classification:

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from src.utils.logging import get_logger

from .base import BasePredictorBackend

logger = get_logger(__name__)

class LightGBMBackend(BasePredictorBackend):
    """LightGBM-based predictor with quantile regression and direction classification"""
    
    def __init__(self):
        self.models = {}  # horizon -> mean regression model
        self.quantile_models = {}  # horizon -> {p10, p50, p90}
        self.direction_models = {}  # horizon -> binary classifier
        self.scaler = StandardScaler()
        self.validation_metrics = {}
        self.feature_names = []
    
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
        
        logger.info("Starting LightGBM training")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
        
        metrics = {}
        
        for horizon in horizons:
            logger.info(f"Training models for {horizon}d horizon")
            
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
            
            # Split train/validation (80/20)
            split_idx = int(len(X_h) * 0.8)
            X_tr, X_val = X_h[:split_idx], X_h[split_idx:]
            y_tr, y_val = y_h[:split_idx], y_h[split_idx:]
            y_dir_tr, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
            
            # 1. Train mean regression model
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models[horizon] = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # 2. Train quantile models (p10, p50, p90)
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
            
            # 3. Train direction classifier
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
            
            # 4. Calculate validation metrics
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
            
            logger.info(
                f"  {horizon}d - RMSE: {rmse:.4f}, "
                f"Dir Acc: {dir_accuracy:.3f}, "
                f"Coverage: {coverage:.3f}"
            )
        
        self.validation_metrics = metrics
        logger.info("Training complete")
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
                pred['direction_prob'] = float(
                    self.direction_models[horizon].predict(X_scaled)[0]
                )
            
            predictions[horizon] = pred
        
        return predictions
    
    def get_model_confidence(self) -> float:
        """Calculate overall model confidence based on directional accuracy"""
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
        
        # Map accuracy to confidence
        # 0.5 accuracy -> 0.0 confidence (coin flip)
        # 1.0 accuracy -> 1.0 confidence (perfect)
        confidence = max(0.0, (avg_accuracy - 0.5) * 2)
        
        return float(confidence)
    
    def get_feature_importance(self, horizon: int, top_n: int = 10) -> Dict[str, float]:
        """Get feature importance for a specific horizon"""
        if horizon not in self.models:
            return {}
        
        importance = self.models[horizon].feature_importance(importance_type='gain')
        feature_imp = dict(zip(self.feature_names, importance))
        
        # Sort by importance and take top N
        sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:top_n])
    
    def save(self, path: str):
        """Save models to disk"""
        import pickle
        
        state = {
            'models': {h: m.model_to_string() for h, m in self.models.items()},
            'quantile_models': {
                h: {q: m.model_to_string() for q, m in qm.items()}
                for h, qm in self.quantile_models.items()
            },
            'direction_models': {
                h: m.model_to_string() for h, m in self.direction_models.items()
            },
            'scaler': self.scaler,
            'validation_metrics': self.validation_metrics,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved models to {path}")
    
    def load(self, path: str):
        """Load models from disk"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.models = {h: lgb.Booster(model_str=s) for h, s in state['models'].items()}
        self.quantile_models = {
            h: {q: lgb.Booster(model_str=s) for q, s in qm.items()}
            for h, qm in state['quantile_models'].items()
        }
        self.direction_models = {
            h: lgb.Booster(model_str=s) for h, s in state['direction_models'].items()
        }
        self.scaler = state['scaler']
        self.validation_metrics = state['validation_metrics']
        self.feature_names = state.get('feature_names', [])
        
        logger.info(f"Loaded models from {path}")
```

### Step 3: SHAP Explainability Integration

**File**: `src/prediction/explainer.py`

Create SHAP-based explainability for web UI:

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)

class PredictionExplainer:
    """SHAP-based model explainability for web UI"""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer
        
        Args:
            model: Trained LightGBM model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
    
    def get_feature_importance(self, top_n: int = 5) -> Dict[str, float]:
        """
        Get top N most important features
        
        Returns:
            Dict mapping feature names to importance scores
        """
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = dict(zip(self.feature_names, importance))
        
        # Sort and take top N
        sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:top_n])
    
    def generate_waterfall_plot(
        self, 
        X: pd.DataFrame,
        output_format: str = "base64"
    ) -> Optional[str]:
        """
        Generate SHAP waterfall plot showing feature contributions
        
        Args:
            X: Feature DataFrame (single row)
            output_format: "base64" for web display or "file" for saving
        
        Returns:
            Base64-encoded PNG image or None if error
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value,
                    data=X.iloc[0].values,
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            if output_format == "base64":
                # Convert to base64 for web display
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                return image_base64
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating waterfall plot: {e}")
            return None
    
    def generate_force_plot(
        self, 
        X: pd.DataFrame,
        output_format: str = "base64"
    ) -> Optional[str]:
        """
        Generate SHAP force plot
        
        Args:
            X: Feature DataFrame (single row)
            output_format: "base64" for web display
        
        Returns:
            Base64-encoded HTML or None if error
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Generate force plot
            force_plot = shap.force_plot(
                self.explainer.expected_value,
                shap_values[0],
                X.iloc[0],
                feature_names=self.feature_names,
                matplotlib=False
            )
            
            if output_format == "base64":
                # Convert to HTML
                html = shap.getjs() + force_plot.html()
                html_base64 = base64.b64encode(html.encode()).decode()
                return html_base64
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating force plot: {e}")
            return None
    
    def get_top_contributing_features(
        self, 
        X: pd.DataFrame, 
        top_n: int = 3
    ) -> List[Dict[str, any]]:
        """
        Get top features contributing to the prediction
        
        Args:
            X: Feature DataFrame (single row)
            top_n: Number of top features to return
        
        Returns:
            List of dicts with feature name, value, and SHAP value
        """
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Get absolute SHAP values
            abs_shap = np.abs(shap_values[0])
            
            # Get top N indices
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'feature': self.feature_names[idx],
                    'value': float(X.iloc[0, idx]),
                    'shap_value': float(shap_values[0][idx]),
                    'contribution': 'positive' if shap_values[0][idx] > 0 else 'negative'
                })
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return []
```

### Step 4: Calibration Utilities

**File**: `src/prediction/utils/calibration.py`

Quality metrics and calibration checks:

```python
import numpy as np
from typing import Dict, List
from src.utils.logging import get_logger

logger = get_logger(__name__)

class CalibrationChecker:
    """Quality metrics and calibration validation"""
    
    @staticmethod
    def check_quantile_coverage(
        y_true: np.ndarray,
        q10: np.ndarray,
        q90: np.ndarray,
        target_coverage: float = 0.80
    ) -> Dict[str, any]:
        """
        Check if 80% prediction interval has proper coverage
        
        Args:
            y_true: Actual values
            q10: 10th percentile predictions
            q90: 90th percentile predictions
            target_coverage: Expected coverage (default 0.80)
        
        Returns:
            Dict with coverage metrics
        """
        actual_coverage = np.mean((y_true >= q10) & (y_true <= q90))
        
        is_calibrated = abs(actual_coverage - target_coverage) < 0.05
        
        return {
            'target_coverage': target_coverage,
            'actual_coverage': float(actual_coverage),
            'is_calibrated': is_calibrated,
            'deviation': float(abs(actual_coverage - target_coverage))
        }
    
    @staticmethod
    def check_directional_accuracy(
        y_true: np.ndarray,
        direction_probs: np.ndarray,
        threshold: float = 0.5,
        min_accuracy: float = 0.52
    ) -> Dict[str, any]:
        """
        Check directional prediction accuracy
        
        Args:
            y_true: Actual values
            direction_probs: Predicted probabilities of up movement
            threshold: Probability threshold for classification
            min_accuracy: Minimum acceptable accuracy
        
        Returns:
            Dict with accuracy metrics
        """
        y_direction = (y_true > 0).astype(int)
        y_pred_direction = (direction_probs > threshold).astype(int)
        
        accuracy = np.mean(y_direction == y_pred_direction)
        
        is_acceptable = accuracy >= min_accuracy
        
        return {
            'accuracy': float(accuracy),
            'min_required': min_accuracy,
            'is_acceptable': is_acceptable,
            'improvement_over_random': float(accuracy - 0.5)
        }
    
    @staticmethod
    def check_prediction_quality(
        validation_metrics: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        Overall quality gate check
        
        Args:
            validation_metrics: Metrics from model training
        
        Returns:
            Quality assessment
        """
        if not validation_metrics:
            return {'passed': False, 'reason': 'No validation metrics'}
        
        checks = []
        
        for horizon, metrics in validation_metrics.items():
            # Check directional accuracy
            dir_acc = metrics.get('directional_accuracy', 0)
            dir_pass = dir_acc >= 0.52
            
            # Check quantile coverage
            coverage = metrics.get('quantile_coverage', 0)
            cov_pass = coverage >= 0.75  # Allow some slack
            
            # Check sample size
            n_samples = metrics.get('n_samples', 0)
            sample_pass = n_samples >= 100
            
            horizon_pass = dir_pass and cov_pass and sample_pass
            
            checks.append({
                'horizon': horizon,
                'passed': horizon_pass,
                'dir_accuracy': dir_acc,
                'coverage': coverage,
                'n_samples': n_samples
            })
        
        all_passed = all(c['passed'] for c in checks)
        
        return {
            'passed': all_passed,
            'checks': checks,
            'summary': f"{sum(c['passed'] for c in checks)}/{len(checks)} horizons passed"
        }
```

### Step 5: Unit Tests

**File**: `tests/prediction/backends/test_lightgbm.py`

Comprehensive tests for LightGBM backend:

```python
import pytest
import pandas as pd
import numpy as np
from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.feature_builder import FeatureBuilder

@pytest.fixture
def sample_training_data():
    """Generate sample data for training"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(n_samples) * 0.1,
        'High': close_prices + abs(np.random.randn(n_samples) * 0.2),
        'Low': close_prices - abs(np.random.randn(n_samples) * 0.2),
        'Close': close_prices
    }, index=dates)
    
    # Build features
    builder = FeatureBuilder(['sma_5', 'sma_20', 'rsi_14', 'macd'])
    features = builder.build_features(df, mode='price_only')
    
    # Build targets
    targets = builder.build_targets(df, horizons=[1, 7])
    
    # Align
    common_idx = features.index.intersection(targets.index)
    X = features.loc[common_idx]
    y = targets.loc[common_idx]
    
    return X, y


def test_lightgbm_backend_training(sample_training_data):
    """Test LightGBM backend trains successfully"""
    X, y = sample_training_data
    
    backend = LightGBMBackend()
    metrics = backend.train(X, y, horizons=[1, 7])
    
    assert '1d' in metrics
    assert '7d' in metrics
    assert metrics['1d']['rmse'] > 0
    assert 0 <= metrics['1d']['directional_accuracy'] <= 1


def test_lightgbm_prediction(sample_training_data):
    """Test making predictions"""
    X, y = sample_training_data
    
    backend = LightGBMBackend()
    backend.train(X, y, horizons=[1, 7])
    
    # Predict on last sample
    X_test = X.iloc[[-1]]
    predictions = backend.predict(X_test, horizons=[1, 7], include_quantiles=True)
    
    assert 1 in predictions
    assert 7 in predictions
    assert 'mean_change' in predictions[1]
    assert 'quantiles' in predictions[1]
    assert 'direction_prob' in predictions[1]


def test_model_confidence(sample_training_data):
    """Test confidence calculation"""
    X, y = sample_training_data
    
    backend = LightGBMBackend()
    backend.train(X, y, horizons=[1, 7])
    
    confidence = backend.get_model_confidence()
    
    assert 0 <= confidence <= 1


def test_feature_importance(sample_training_data):
    """Test feature importance extraction"""
    X, y = sample_training_data
    
    backend = LightGBMBackend()
    backend.train(X, y, horizons=[1, 7])
    
    importance = backend.get_feature_importance(horizon=1, top_n=3)
    
    assert len(importance) <= 3
    assert all(isinstance(v, (int, float)) for v in importance.values())


def test_save_load(sample_training_data, tmp_path):
    """Test model save and load"""
    X, y = sample_training_data
    
    backend = LightGBMBackend()
    backend.train(X, y, horizons=[1])
    
    # Save
    model_path = tmp_path / "test_model.pkl"
    backend.save(str(model_path))
    
    assert model_path.exists()
    
    # Load
    backend2 = LightGBMBackend()
    backend2.load(str(model_path))
    
    # Verify predictions match
    X_test = X.iloc[[-1]]
    pred1 = backend.predict(X_test, horizons=[1])
    pred2 = backend2.predict(X_test, horizons=[1])
    
    assert abs(pred1[1]['mean_change'] - pred2[1]['mean_change']) < 0.001
```

## Key Design Decisions

1. **Three model types per horizon**: Mean regression, quantile regression (uncertainty), direction classification
2. **StandardScaler**: Feature scaling for stable training
3. **Early stopping**: Prevents overfitting with 10-round patience
4. **80/20 train/val split**: Standard validation approach
5. **SHAP for explainability**: Visual feature importance for web UI
6. **Confidence from accuracy**: Maps directional accuracy to 0-1 scale
7. **Model-to-string serialization**: LightGBM's native format for pickle
8. **Quality gates**: Automated calibration and accuracy checks

## Files to Create

- `src/prediction/backends/__init__.py`
- `src/prediction/backends/base.py`
- `src/prediction/backends/lightgbm_backend.py`
- `src/prediction/explainer.py`
- `src/prediction/utils/__init__.py`
- `src/prediction/utils/calibration.py`
- `tests/prediction/backends/__init__.py`
- `tests/prediction/backends/test_lightgbm.py`
- `tests/prediction/test_explainer.py`
- `tests/prediction/test_calibration.py`

## Dependencies

Add to `pyproject.toml`:

```toml
lightgbm = ">=4.0.0"
scikit-learn = ">=1.3.0"
shap = ">=0.44.0"
matplotlib = ">=3.7.0"  # For SHAP plots
```

## Validation

Manual testing:

```python
from src.prediction.data_loader import HistoricalDataLoader
from src.prediction.feature_builder import FeatureBuilder
from src.prediction.backends.lightgbm_backend import LightGBMBackend
from src.prediction.config import PredictionConfig

# Load data
config = PredictionConfig.from_yaml()
loader = HistoricalDataLoader()
df = await loader.fetch_historical_data("USD", "EUR", days=365)

# Build features and targets
builder = FeatureBuilder(config.technical_indicators)
features = builder.build_features(df, mode="price_only")
targets = builder.build_targets(df, horizons=[1, 7, 30])

# Align
common_idx = features.index.intersection(targets.index)
X = features.loc[common_idx]
y = targets.loc[common_idx]

# Train
backend = LightGBMBackend()
metrics = backend.train(X, y, horizons=[1, 7, 30])
print("Validation metrics:", metrics)

# Predict
X_latest = X.iloc[[-1]]
predictions = backend.predict(X_latest, horizons=[1, 7, 30])
print("Predictions:", predictions)

# Explainability
from src.prediction.explainer import PredictionExplainer
explainer = PredictionExplainer(backend.models[1], backend.feature_names)
importance = explainer.get_feature_importance(top_n=5)
print("Top features:", importance)
```

## Success Criteria

- Models train successfully for multiple horizons
- Directional accuracy > 52% (better than random)
- Quantile coverage within 75-90% range
- Predictions complete in <500ms
- Feature importance correctly identifies key drivers
- SHAP plots generate without errors
- Models save and load correctly
- All unit tests pass with >80% coverage
- Confidence score correctly reflects model quality

## Next Phase

After Phase 2.3 completes, proceed to **Phase 2.4: Fallback Heuristics** to implement simple rule-based predictions for when ML models are unavailable.

### To-dos

- [ ] Create BasePredictorBackend interface in src/prediction/backends/base.py
- [ ] Implement LightGBMBackend with quantile regression and direction classification
- [ ] Create PredictionExplainer with SHAP for web UI visualizations (waterfall, force plots)
- [ ] Implement CalibrationChecker for quality gates in src/prediction/utils/calibration.py
- [ ] Add model training with early stopping and validation metrics
- [ ] Write comprehensive unit tests for LightGBM backend (>80% coverage)
- [ ] Write unit tests for SHAP explainer and calibration utilities
- [ ] Manually validate model training with real EUR/USD data