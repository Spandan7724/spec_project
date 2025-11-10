#!/usr/bin/env python3
"""
Test script to verify CatBoost_3 model can be loaded and used for predictions
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.predictor import MLPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig


async def test_catboost_prediction():
    """Test CatBoost model loading and prediction"""
    print("=" * 80)
    print("Testing CatBoost_3 Model in Prediction System")
    print("=" * 80)

    # Initialize predictor
    config = PredictionConfig.from_yaml()
    predictor = MLPredictor(config)

    print(f"\n✓ Predictor initialized")
    print(f"  - Registry path: {config.model_registry_path}")
    print(f"  - Storage dir: {config.model_storage_dir}")

    # Create prediction request
    request = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7, 30],
        include_quantiles=True,
        include_direction_probabilities=True,
        max_age_hours=1,
        features_mode="price_only",
        correlation_id="test_catboost",
    )

    print(f"\n📊 Making prediction request for {request.currency_pair}")
    print(f"  - Horizons: {request.horizons}")

    # Make prediction
    try:
        response = await predictor.predict(request)

        print(f"\n✅ Prediction successful!")
        print(f"  - Status: {response.status}")
        print(f"  - Model ID: {response.model_id}")
        print(f"  - Confidence: {response.confidence:.2%}")
        print(f"  - Processing time: {response.processing_time_ms}ms")
        print(f"  - Latest close: ${response.latest_close:.4f}")

        if response.predictions:
            print(f"\n📈 Predictions:")
            for horizon, pred in response.predictions.items():
                print(f"\n  Horizon {horizon}d:")
                print(f"    - Mean change: {pred.mean_change_pct:+.4f}%")
                print(f"    - Direction probability: {pred.direction_probability:.2%}")
                if pred.quantiles:
                    print(f"    - Quantiles: {pred.quantiles}")

        if response.quality:
            print(f"\n📊 Quality Metrics:")
            print(f"  - Model confidence: {response.quality.model_confidence:.2%}")
            print(f"  - Calibrated: {response.quality.calibrated}")
            if response.quality.validation_metrics:
                print(f"  - Validation metrics: {response.quality.validation_metrics}")
            if response.quality.notes:
                print(f"  - Notes: {', '.join(response.quality.notes)}")

        if response.model_info:
            print(f"\n🔍 Model Info:")
            for key, value in response.model_info.items():
                print(f"  - {key}: {value}")

        return True

    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_catboost_prediction())
    sys.exit(0 if success else 1)
