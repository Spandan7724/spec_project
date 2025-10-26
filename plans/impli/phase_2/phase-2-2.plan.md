<!-- f67714c1-a54f-4e8d-9617-16955f212afc b12a9487-8678-4e6d-ae67-7f14e93a9f0a -->
# Phase 2.2: Model Registry & Storage

## Overview

Build a lightweight model registry using JSON for metadata and Python's pickle for model serialization. This eliminates the need for MLflow while providing essential functionality: model versioning, metadata tracking, model save/load, and query capabilities. Perfect for a college project with future upgrade path to MLflow if needed.

## Implementation Steps

### Step 1: Model Registry Implementation

**File**: `src/prediction/registry.py`

Implement JSON-based registry with pickle storage:

```python
import json
import os
import pickle
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from .models import ModelMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """Simple JSON-based model registry with pickle storage"""
    
    def __init__(self, registry_path: str, storage_dir: str):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to JSON metadata file
            storage_dir: Directory to store model pickle files
        """
        self.registry_path = registry_path
        self.storage_dir = storage_dir
        
        # Ensure directories exist
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from JSON file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} models")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.registry = {}
        else:
            self.registry = {}
            logger.info("Initialized empty registry")
    
    def _save_registry(self):
        """Save registry to JSON file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
            logger.info(f"Saved registry with {len(self.registry)} models")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise
    
    def register_model(
        self, 
        metadata: ModelMetadata,
        model_obj: Any,
        scaler_obj: Optional[Any] = None
    ):
        """
        Register a new model
        
        Args:
            metadata: Model metadata
            model_obj: Trained model object (will be pickled)
            scaler_obj: Optional feature scaler (will be pickled)
        """
        logger.info(f"Registering model: {metadata.model_id}")
        
        # Save model file
        model_path = os.path.join(self.storage_dir, f"{metadata.model_id}.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_obj, f)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
        metadata.model_path = model_path
        
        # Save scaler if provided
        if scaler_obj is not None:
            scaler_path = os.path.join(self.storage_dir, f"{metadata.model_id}_scaler.pkl")
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler_obj, f)
                logger.info(f"Saved scaler to {scaler_path}")
                metadata.scaler_path = scaler_path
            except Exception as e:
                logger.error(f"Error saving scaler: {e}")
                raise
        
        # Add to registry
        self.registry[metadata.model_id] = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "currency_pair": metadata.currency_pair,
            "trained_at": metadata.trained_at.isoformat() if isinstance(metadata.trained_at, datetime) else metadata.trained_at,
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
        logger.info(f"Successfully registered model: {metadata.model_id}")
    
    def get_model(
        self, 
        currency_pair: str, 
        model_type: str = "lightgbm"
    ) -> Optional[Dict]:
        """
        Get latest model for a currency pair
        
        Args:
            currency_pair: Currency pair (e.g., "USD/EUR")
            model_type: Model type ("lightgbm" or "lstm")
        
        Returns:
            Model metadata dict or None if not found
        """
        # Find matching models
        matches = [
            m for m in self.registry.values()
            if m['currency_pair'] == currency_pair and m['model_type'] == model_type
        ]
        
        if not matches:
            logger.warning(f"No model found for {currency_pair} ({model_type})")
            return None
        
        # Return most recent
        latest = max(matches, key=lambda m: m['trained_at'])
        logger.info(f"Found model {latest['model_id']} for {currency_pair}")
        return latest
    
    def load_model_objects(self, model_metadata: Dict) -> tuple:
        """
        Load model and scaler objects from disk
        
        Args:
            model_metadata: Model metadata dict from get_model()
        
        Returns:
            Tuple of (model, scaler) where scaler may be None
        """
        model_path = model_metadata['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        scaler = None
        scaler_path = model_metadata.get('scaler_path')
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
                raise
        
        return model, scaler
    
    def list_models(
        self, 
        currency_pair: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        List all models, optionally filtered
        
        Args:
            currency_pair: Optional filter by currency pair
            model_type: Optional filter by model type
        
        Returns:
            List of model metadata dicts, sorted by trained_at (newest first)
        """
        models = list(self.registry.values())
        
        if currency_pair:
            models = [m for m in models if m['currency_pair'] == currency_pair]
        
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        # Sort by trained_at descending
        models = sorted(models, key=lambda m: m['trained_at'], reverse=True)
        
        logger.info(f"Listed {len(models)} models")
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from registry and disk
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        if model_id not in self.registry:
            logger.warning(f"Model not found: {model_id}")
            return False
        
        metadata = self.registry[model_id]
        
        # Delete model file
        if os.path.exists(metadata['model_path']):
            os.remove(metadata['model_path'])
            logger.info(f"Deleted model file: {metadata['model_path']}")
        
        # Delete scaler file if exists
        if metadata.get('scaler_path') and os.path.exists(metadata['scaler_path']):
            os.remove(metadata['scaler_path'])
            logger.info(f"Deleted scaler file: {metadata['scaler_path']}")
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
        
        logger.info(f"Deleted model: {model_id}")
        return True
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific model
        
        Args:
            model_id: Model ID
        
        Returns:
            Model metadata dict or None
        """
        return self.registry.get(model_id)
```

### Step 2: Create Model Storage Directory Structure

**Directory**: `models/prediction/`

Create the directory structure for storing models:

```bash
mkdir -p models/prediction
```

### Step 3: Initialize Empty Registry

**File**: `models/prediction_registry.json`

Create empty registry file (will be auto-populated):

```json
{}
```

### Step 4: Unit Tests

**File**: `tests/prediction/test_registry.py`

Test model registry functionality:

```python
import pytest
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.prediction.registry import ModelRegistry
from src.prediction.models import ModelMetadata


@pytest.fixture
def temp_registry_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model():
    """Create a simple mock model object"""
    class MockModel:
        def __init__(self):
            self.weights = [1, 2, 3, 4, 5]
        
        def predict(self, X):
            return [0.5] * len(X)
    
    return MockModel()


@pytest.fixture
def sample_scaler():
    """Create a simple mock scaler object"""
    class MockScaler:
        def __init__(self):
            self.mean = 0.5
            self.std = 0.1
        
        def transform(self, X):
            return (X - self.mean) / self.std
    
    return MockScaler()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata"""
    return ModelMetadata(
        model_id="test_usdeur_lightgbm_20251025",
        model_type="lightgbm",
        currency_pair="USD/EUR",
        trained_at=datetime(2025, 10, 25, 12, 0, 0),
        version="1.0",
        validation_metrics={
            "1d": {"rmse": 0.45, "directional_accuracy": 0.68, "quantile_coverage": 0.87},
            "7d": {"rmse": 0.62, "directional_accuracy": 0.65, "quantile_coverage": 0.85}
        },
        min_samples=800,
        calibration_ok=True,
        features_used=["sma_5", "sma_20", "rsi_14", "macd"],
        horizons=[1, 7, 30],
        model_path=""
    )


def test_registry_initialization(temp_registry_dir):
    """Test registry initializes correctly"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    
    assert os.path.exists(storage_dir)
    assert os.path.exists(registry_path)
    assert registry.registry == {}


def test_register_model(temp_registry_dir, sample_model, sample_metadata):
    """Test model registration"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)
    
    # Check model file was created
    assert os.path.exists(sample_metadata.model_path)
    
    # Check registry was updated
    assert sample_metadata.model_id in registry.registry
    assert registry.registry[sample_metadata.model_id]["currency_pair"] == "USD/EUR"


def test_register_model_with_scaler(temp_registry_dir, sample_model, sample_scaler, sample_metadata):
    """Test model registration with scaler"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model, sample_scaler)
    
    # Check both files were created
    assert os.path.exists(sample_metadata.model_path)
    assert sample_metadata.scaler_path is not None
    assert os.path.exists(sample_metadata.scaler_path)


def test_get_model(temp_registry_dir, sample_model, sample_metadata):
    """Test retrieving model by currency pair"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)
    
    # Retrieve model
    retrieved = registry.get_model("USD/EUR", "lightgbm")
    
    assert retrieved is not None
    assert retrieved["model_id"] == sample_metadata.model_id
    assert retrieved["currency_pair"] == "USD/EUR"


def test_get_model_not_found(temp_registry_dir):
    """Test retrieving non-existent model"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    
    retrieved = registry.get_model("GBP/JPY", "lightgbm")
    assert retrieved is None


def test_load_model_objects(temp_registry_dir, sample_model, sample_scaler, sample_metadata):
    """Test loading model and scaler from disk"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model, sample_scaler)
    
    # Load back
    metadata_dict = registry.get_model("USD/EUR", "lightgbm")
    loaded_model, loaded_scaler = registry.load_model_objects(metadata_dict)
    
    assert loaded_model is not None
    assert loaded_scaler is not None
    assert hasattr(loaded_model, 'weights')
    assert hasattr(loaded_scaler, 'mean')


def test_list_models(temp_registry_dir, sample_model):
    """Test listing models"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    
    # Register multiple models
    for i, pair in enumerate(["USD/EUR", "USD/EUR", "GBP/JPY"]):
        metadata = ModelMetadata(
            model_id=f"test_{pair.replace('/', '')}_{i}",
            model_type="lightgbm",
            currency_pair=pair,
            trained_at=datetime(2025, 10, 25, 12, i, 0),
            version="1.0",
            validation_metrics={},
            min_samples=500,
            calibration_ok=True,
            features_used=[],
            horizons=[1, 7, 30],
            model_path=""
        )
        registry.register_model(metadata, sample_model)
    
    # List all models
    all_models = registry.list_models()
    assert len(all_models) == 3
    
    # List filtered by currency pair
    usd_eur_models = registry.list_models(currency_pair="USD/EUR")
    assert len(usd_eur_models) == 2


def test_delete_model(temp_registry_dir, sample_model, sample_metadata):
    """Test deleting a model"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)
    
    model_path = sample_metadata.model_path
    assert os.path.exists(model_path)
    
    # Delete model
    success = registry.delete_model(sample_metadata.model_id)
    
    assert success
    assert not os.path.exists(model_path)
    assert sample_metadata.model_id not in registry.registry


def test_registry_persistence(temp_registry_dir, sample_model, sample_metadata):
    """Test that registry persists across instances"""
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")
    
    # Create first registry instance and register model
    registry1 = ModelRegistry(registry_path, storage_dir)
    registry1.register_model(sample_metadata, sample_model)
    
    # Create new registry instance (should load existing data)
    registry2 = ModelRegistry(registry_path, storage_dir)
    
    assert sample_metadata.model_id in registry2.registry
    assert len(registry2.registry) == 1
```

## Key Design Decisions

1. **JSON for metadata**: Human-readable, version-control friendly, no extra dependencies
2. **Pickle for models**: Python standard library, works with any ML framework
3. **Separate model and scaler storage**: Allows independent loading/updating
4. **Automatic directory creation**: Handles setup automatically
5. **Latest model selection**: Returns most recent model by default
6. **Comprehensive logging**: All operations logged for debugging
7. **Delete functionality**: Clean up old models when needed
8. **Model filtering**: Query by currency pair and model type

## Files to Create

- `src/prediction/registry.py`
- `models/prediction/` (directory)
- `models/prediction_registry.json` (empty JSON file)
- `tests/prediction/test_registry.py`

## Dependencies

No additional dependencies needed! Uses Python standard library:

- `json` - for metadata storage
- `pickle` - for model serialization
- `pathlib` - for path handling

## Validation

Manual testing:

```python
from src.prediction.registry import ModelRegistry
from src.prediction.models import ModelMetadata
from datetime import datetime

# Initialize registry
registry = ModelRegistry(
    registry_path="models/prediction_registry.json",
    storage_dir="models/prediction/"
)

# Create sample metadata
metadata = ModelMetadata(
    model_id="usdeur_lightgbm_v1",
    model_type="lightgbm",
    currency_pair="USD/EUR",
    trained_at=datetime.now(),
    version="1.0",
    validation_metrics={"1d": {"rmse": 0.45, "directional_accuracy": 0.68}},
    min_samples=800,
    calibration_ok=True,
    features_used=["sma_5", "sma_20", "rsi_14"],
    horizons=[1, 7, 30],
    model_path=""
)

# Register a dummy model
class DummyModel:
    def predict(self, X):
        return [0.5] * len(X)

model = DummyModel()
registry.register_model(metadata, model)

# Retrieve and load
model_meta = registry.get_model("USD/EUR", "lightgbm")
loaded_model, _ = registry.load_model_objects(model_meta)
print(f"Loaded model: {model_meta['model_id']}")

# List all models
all_models = registry.list_models()
print(f"Total models: {len(all_models)}")
```

## Success Criteria

- Registry initializes and creates directories automatically
- Models can be registered with metadata
- Models can be retrieved by currency pair and type
- Model and scaler objects pickle and unpickle correctly
- Registry JSON file persists across sessions
- Latest model is returned when multiple exist for same pair
- Model deletion removes files and registry entry
- All unit tests pass with >85% coverage
- Registry operations complete in <100ms

## Next Phase

After Phase 2.2 completes, proceed to **Phase 2.3: LightGBM Backend + SHAP Explainability** to implement the actual ML model training and prediction logic.

### To-dos

- [ ] Implement ModelRegistry class with JSON metadata and pickle model storage
- [ ] Create models/prediction/ directory structure and empty registry.json file
- [ ] Add model registration, retrieval, and deletion functionality
- [ ] Implement model metadata tracking (accuracy, currency_pair, version, features)
- [ ] Add model loading with scaler support for feature preprocessing
- [ ] Write comprehensive unit tests for registry operations (>85% coverage)
- [ ] Test registry persistence across multiple instances