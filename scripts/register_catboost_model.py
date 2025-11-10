#!/usr/bin/env python3
"""
Script to register the best CatBoost_3 model from ml_models/models/
into the prediction registry at data/models/prediction/
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.registry import ModelRegistry
from src.prediction.models import ModelMetadata
from src.prediction.config import PredictionConfig


def register_catboost_model(currency_pair: str = "USD/EUR"):
    """
    Register CatBoost_3 model from ml_models/models/ into the prediction registry.

    Args:
        currency_pair: The currency pair this model was trained on (default: USD/EUR)
    """
    # Paths
    ml_models_dir = project_root / "ml_models" / "models"
    catboost_model_path = ml_models_dir / "catboost_model3.cbm"

    if not catboost_model_path.exists():
        print(f"❌ Error: CatBoost model not found at {catboost_model_path}")
        return False

    # Load config
    config = PredictionConfig.from_yaml()

    # Initialize registry
    registry = ModelRegistry(
        registry_path=config.model_registry_path,
        storage_dir=config.model_storage_dir
    )

    # Create model metadata
    model_id = f"{currency_pair.lower().replace('/', '')}_catboost_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    metadata = ModelMetadata(
        model_id=model_id,
        model_type="catboost",
        currency_pair=currency_pair,
        trained_at=datetime.now(),
        version="1.0",
        validation_metrics={
            "test_rmse": 0.00564,
            "test_mae": 0.00424,
            "test_r2": 0.9883,
            "mape": 0.38,
            "direction_accuracy": 51.10
        },
        min_samples=2767,
        calibration_ok=True,
        features_used=174,  # From training_results.json
        horizons=[1, 7, 30],
        model_path="",  # Will be set by registry
        scaler_path=None
    )

    print(f"📦 Registering CatBoost_3 model...")
    print(f"   Model ID: {model_id}")
    print(f"   Currency Pair: {currency_pair}")
    print(f"   Test R²: 0.9883 (98.83%)")
    print(f"   Source: {catboost_model_path}")

    # Copy the model file to the registry storage directory
    dest_path = Path(registry.storage_dir) / f"{model_id}.cbm"
    shutil.copy2(catboost_model_path, dest_path)
    print(f"   ✓ Copied model to {dest_path}")

    # Copy the scaler file
    scaler_src_path = ml_models_dir / "scaler.pkl"
    scaler_dest_path = None
    if scaler_src_path.exists():
        scaler_dest_path = Path(registry.storage_dir) / f"{model_id}_scaler.pkl"
        shutil.copy2(scaler_src_path, scaler_dest_path)
        print(f"   ✓ Copied scaler to {scaler_dest_path}")
    else:
        print(f"   ⚠ Scaler not found at {scaler_src_path}")

    # Update metadata with actual path
    metadata.model_path = str(dest_path)
    metadata.scaler_path = str(scaler_dest_path) if scaler_dest_path else None

    # Register in the JSON registry (manually since we're not using pickle)
    registry.registry[model_id] = {
        "model_id": model_id,
        "model_type": "catboost",
        "currency_pair": currency_pair,
        "trained_at": metadata.trained_at.isoformat(),
        "version": metadata.version,
        "validation_metrics": metadata.validation_metrics,
        "min_samples": metadata.min_samples,
        "calibration_ok": metadata.calibration_ok,
        "features_used": metadata.features_used,
        "horizons": metadata.horizons,
        "model_path": str(dest_path),
        "scaler_path": str(scaler_dest_path) if scaler_dest_path else None,
    }
    registry._save_registry()

    print(f"   ✓ Registered in prediction registry")
    print(f"\n✅ Successfully registered CatBoost_3 model!")
    print(f"\n📊 Model will now be used for {currency_pair} predictions in the agentic loop")

    # List all catboost models in registry
    catboost_models = registry.list_models(model_type="catboost")
    print(f"\n📋 All CatBoost models in registry: {len(catboost_models)}")
    for model in catboost_models:
        print(f"   - {model['model_id']} (R²: {model['validation_metrics'].get('test_r2', 'N/A')})")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register CatBoost_3 model into prediction registry")
    parser.add_argument(
        "--pair",
        default="USD/EUR",
        help="Currency pair (default: USD/EUR)"
    )

    args = parser.parse_args()

    success = register_catboost_model(currency_pair=args.pair)
    sys.exit(0 if success else 1)
