#!/usr/bin/env python3
"""
Test script to verify model confidence is properly calculated
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.advanced_predictor import AdvancedMLPredictor
from src.prediction.models import PredictionRequest
from src.prediction.config import PredictionConfig
from src.prediction.backends.advanced_ensemble_backend import AdvancedEnsembleBackend


def test_backend_confidence():
    """Test the advanced ensemble backend confidence directly"""
    print("=" * 80)
    print("Testing Advanced Ensemble Backend Confidence")
    print("=" * 80)

    try:
        backend = AdvancedEnsembleBackend(ml_models_dir="ml_models/models")
        confidence = backend.get_model_confidence()
        model_info = backend.get_model_info()

        print(f"\n✅ Backend loaded successfully")
        print(f"   - Best model: {model_info['best_model']}")
        print(f"   - Test R²: {model_info['test_r2']:.6f}")
        print(f"   - Test RMSE: {model_info['test_rmse']:.6f}")
        print(f"   - Test MAE: {model_info['test_mae']:.6f}")
        print(f"   - MAPE: {model_info['mape']:.4f}%")
        print(f"\n📊 Model Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        if confidence < 0.90:
            print(f"   ⚠️  WARNING: Confidence is below 90%!")
        else:
            print(f"   ✅ Excellent confidence (>90%)!")

        return confidence

    except Exception as e:
        print(f"\n❌ Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


async def test_predictor_confidence():
    """Test the advanced predictor confidence end-to-end"""
    print("\n" + "=" * 80)
    print("Testing Advanced Predictor Confidence (End-to-End)")
    print("=" * 80)

    try:
        config = PredictionConfig.from_yaml()
        predictor = AdvancedMLPredictor(config, ml_models_dir="ml_models/models")

        if not predictor.is_available():
            print("❌ Advanced predictor not available")
            return 0.0

        print(f"\n✓ Advanced predictor initialized")

        # Create prediction request
        request = PredictionRequest(
            currency_pair="USD/EUR",
            horizons=[1, 7, 30],
            include_quantiles=True,
            include_direction_probabilities=True,
            max_age_hours=1,
            features_mode="price_only",
            correlation_id="test_confidence",
        )

        print(f"\n📊 Making prediction for {request.currency_pair}...")

        response = await predictor.predict(request)

        print(f"\n✅ Prediction successful!")
        print(f"   - Status: {response.status}")
        print(f"   - Model ID: {response.model_id}")
        print(f"   - Confidence: {response.confidence:.4f} ({response.confidence*100:.2f}%)")

        if response.confidence < 0.90:
            print(f"   ⚠️  WARNING: Confidence is below 90%!")
        elif response.confidence < 0.50:
            print(f"   ❌ ERROR: Confidence is critically low (<50%)!")
        else:
            print(f"   ✅ Excellent confidence!")

        if response.quality:
            print(f"\n📈 Quality Metrics:")
            print(f"   - Model confidence: {response.quality.model_confidence:.4f} ({response.quality.model_confidence*100:.2f}%)")
            print(f"   - Calibrated: {response.quality.calibrated}")

            if response.quality.validation_metrics:
                print(f"   - Validation metrics:")
                for key, value in response.quality.validation_metrics.items():
                    if isinstance(value, float):
                        print(f"     • {key}: {value:.6f}")
                    else:
                        print(f"     • {key}: {value}")

        if response.model_info:
            print(f"\n🔍 Model Info:")
            if 'best_model' in response.model_info:
                print(f"   - Best model: {response.model_info['best_model']}")
            if 'test_metrics' in response.model_info:
                metrics = response.model_info['test_metrics']
                if metrics:
                    print(f"   - Test metrics:")
                    for key, value in metrics.items():
                        if value is not None and isinstance(value, (int, float)):
                            print(f"     • {key}: {value:.6f}")

        return response.confidence

    except Exception as e:
        print(f"\n❌ Predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


async def main():
    print("\n" + "🔬" * 40)
    print("Model Confidence Validation Test Suite")
    print("🔬" * 40)

    # Test 1: Backend confidence
    backend_confidence = test_backend_confidence()

    # Test 2: Predictor confidence
    predictor_confidence = await test_predictor_confidence()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Backend confidence:   {backend_confidence:.4f} ({backend_confidence*100:.2f}%)")
    print(f"Predictor confidence: {predictor_confidence:.4f} ({predictor_confidence*100:.2f}%)")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    all_passed = True

    if backend_confidence >= 0.90:
        print("✅ Backend confidence >= 90% (PASS)")
    else:
        print(f"❌ Backend confidence < 90% (FAIL): {backend_confidence*100:.2f}%")
        all_passed = False

    if predictor_confidence >= 0.90:
        print("✅ Predictor confidence >= 90% (PASS)")
    else:
        print(f"❌ Predictor confidence < 90% (FAIL): {predictor_confidence*100:.2f}%")
        all_passed = False

    if abs(backend_confidence - predictor_confidence) < 0.05:
        print("✅ Backend and predictor confidences are consistent (PASS)")
    else:
        print(f"⚠️  Backend and predictor confidences differ by {abs(backend_confidence - predictor_confidence)*100:.2f}%")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
