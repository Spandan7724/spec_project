import os
import tempfile
from datetime import datetime

import pytest

from src.prediction.registry import ModelRegistry
from src.prediction.models import ModelMetadata


class MockModel:
    def __init__(self):
        self.weights = [1, 2, 3]

    def predict(self, X):
        return [0.5 for _ in range(len(X))]


class MockScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def transform(self, X):
        return [(x - self.mean) / self.std for x in X]


@pytest.fixture
def temp_registry_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model():
    return MockModel()


@pytest.fixture
def sample_scaler():
    return MockScaler()


@pytest.fixture
def sample_metadata():
    return ModelMetadata(
        model_id="test_usdeur_lightgbm_20251025",
        model_type="lightgbm",
        currency_pair="USD/EUR",
        trained_at=datetime(2025, 10, 25, 12, 0, 0),
        version="1.0",
        validation_metrics={
            "1d": {"rmse": 0.45, "directional_accuracy": 0.68, "quantile_coverage": 0.87},
            "7d": {"rmse": 0.62, "directional_accuracy": 0.65, "quantile_coverage": 0.85},
        },
        min_samples=800,
        calibration_ok=True,
        features_used=["sma_5", "sma_20", "rsi_14", "macd"],
        horizons=[1, 7, 30],
        model_path="",
    )


def test_registry_initialization(temp_registry_dir):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)

    assert os.path.exists(storage_dir)
    assert os.path.exists(registry_path)
    assert registry.registry == {}


def test_register_model(temp_registry_dir, sample_model, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)

    assert os.path.exists(sample_metadata.model_path)
    assert sample_metadata.model_id in registry.registry
    assert registry.registry[sample_metadata.model_id]["currency_pair"] == "USD/EUR"


def test_register_model_with_scaler(temp_registry_dir, sample_model, sample_scaler, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model, sample_scaler)

    assert os.path.exists(sample_metadata.model_path)
    assert sample_metadata.scaler_path is not None
    assert os.path.exists(sample_metadata.scaler_path)


def test_get_model(temp_registry_dir, sample_model, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)

    retrieved = registry.get_model("USD/EUR", "lightgbm")
    assert retrieved is not None
    assert retrieved["model_id"] == sample_metadata.model_id
    assert retrieved["currency_pair"] == "USD/EUR"


def test_get_model_not_found(temp_registry_dir):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    assert registry.get_model("GBP/JPY", "lightgbm") is None


def test_load_model_objects(temp_registry_dir, sample_model, sample_scaler, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model, sample_scaler)

    meta = registry.get_model("USD/EUR", "lightgbm")
    model_obj, scaler_obj = registry.load_model_objects(meta)
    assert hasattr(model_obj, "predict")
    assert scaler_obj is not None and hasattr(scaler_obj, "transform")


def test_list_models(temp_registry_dir, sample_model):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)

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
            model_path="",
        )
        registry.register_model(metadata, sample_model)

    all_models = registry.list_models()
    assert len(all_models) == 3

    usd_eur_models = registry.list_models(currency_pair="USD/EUR")
    assert len(usd_eur_models) == 2


def test_delete_model(temp_registry_dir, sample_model, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry = ModelRegistry(registry_path, storage_dir)
    registry.register_model(sample_metadata, sample_model)

    model_path = sample_metadata.model_path
    assert os.path.exists(model_path)

    success = registry.delete_model(sample_metadata.model_id)
    assert success
    assert not os.path.exists(model_path)
    assert sample_metadata.model_id not in registry.registry


def test_registry_persistence(temp_registry_dir, sample_model, sample_metadata):
    registry_path = os.path.join(temp_registry_dir, "registry.json")
    storage_dir = os.path.join(temp_registry_dir, "models")

    registry1 = ModelRegistry(registry_path, storage_dir)
    registry1.register_model(sample_metadata, sample_model)

    registry2 = ModelRegistry(registry_path, storage_dir)
    assert sample_metadata.model_id in registry2.registry
    assert len(registry2.registry) == 1
