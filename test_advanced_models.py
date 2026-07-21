#!/usr/bin/env python3
"""
Test script for advanced ensemble predictor integration.
Tests the new ml_models integration with the currency assistant system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prediction.advanced_predictor import AdvancedMLPredictor
from src.prediction.config import PredictionConfig
from src.prediction.models import PredictionRequest


async def test_model_loading():
    """Test 1: Verify models can be loaded"""
    print("=" * 80)
    print("TEST 1: Model Loading")
    print("=" * 80)

    try:
        predictor = AdvancedMLPredictor(ml_models_dir="ml_models/models")

        if predictor.is_available():
            print("✓ Advanced ensemble predictor loaded successfully")

            model_info = predictor.get_model_info()
            print(f"\nModel Information:")
            print(f"  Best Model: {model_info.get('best_model')}")
            print(f"  Test RMSE: {model_info.get('test_rmse', 'N/A')}")
            print(f"  Test MAE: {model_info.get('test_mae', 'N/A')}")
            print(f"  Test R²: {model_info.get('test_r2', 'N/A')}")
            print(f"  MAPE: {model_info.get('mape', 'N/A')}%")
            print(f"  Direction Accuracy: {model_info.get('direction_accuracy', 'N/A')}%")
            print(f"\n  Loaded Models:")
            loaded = model_info.get('loaded_models', {})
            for name, count in loaded.items():
                if isinstance(count, int):
                    print(f"    - {name}: {count}")
                else:
                    print(f"    - {name}: {'✓' if count else '✗'}")

            return True
        else:
            print("✗ Advanced ensemble predictor not available")
            return False

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prediction():
    """Test 2: Make a sample prediction"""
    print("\n" + "=" * 80)
    print("TEST 2: Sample Prediction (USD/EUR)")
    print("=" * 80)

    try:
        predictor = AdvancedMLPredictor(ml_models_dir="ml_models/models")

        if not predictor.is_available():
            print("✗ Predictor not available, skipping test")
            return False

        # Create prediction request
        request = PredictionRequest(
            currency_pair="USD/EUR",
            horizons=[1, 7, 30],
            include_quantiles=True,
            include_direction_probabilities=True,
            max_age_hours=24,
            correlation_id="test_001"
        )

        print(f"\nRequest:")
        print(f"  Currency Pair: {request.currency_pair}")
        print(f"  Horizons: {request.horizons}")
        print(f"\nFetching data and generating predictions...")

        # Make prediction
        response = await predictor.predict(request)

        print(f"\nResponse:")
        print(f"  Status: {response.status}")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Processing Time: {response.processing_time_ms}ms")
        print(f"  Latest Close: {response.latest_close:.6f}")
        print(f"  Features Used: {len(response.features_used)}")

        if response.predictions:
            print(f"\n  Predictions:")
            for horizon, pred in response.predictions.items():
                print(f"\n    Horizon {horizon} day(s):")
                print(f"      Mean Change: {pred.mean_change_pct:+.4f}")
                print(f"      Direction Prob: {pred.direction_probability:.2%}")

                if pred.quantiles:
                    print(f"      Quantiles:")
                    for q_name, q_val in pred.quantiles.items():
                        print(f"        {q_name}: {q_val:+.4f}")

        if response.quality:
            print(f"\n  Quality Metrics:")
            print(f"    Model Confidence: {response.quality.model_confidence:.2%}")
            print(f"    Calibrated: {response.quality.calibrated}")
            if response.quality.validation_metrics:
                print(f"    Validation Metrics:")
                for k, v in response.quality.validation_metrics.items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.6f}")
                    else:
                        print(f"      {k}: {v}")

        if response.model_info:
            print(f"\n  Model Info:")
            for k, v in response.model_info.items():
                if k == "test_metrics":
                    print(f"    {k}:")
                    for mk, mv in v.items():
                        print(f"      {mk}: {mv}")
                elif k == "models_loaded":
                    print(f"    {k}:")
                    for mk, mv in v.items():
                        print(f"      {mk}: {mv}")
                else:
                    print(f"    {k}: {v}")

        print("\n✓ Prediction completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_pairs():
    """Test 3: Test with different currency pairs"""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Currency Pairs")
    print("=" * 80)

    try:
        predictor = AdvancedMLPredictor(ml_models_dir="ml_models/models")

        if not predictor.is_available():
            print("✗ Predictor not available, skipping test")
            return False

        pairs = ["EUR/USD", "GBP/USD", "USD/JPY"]
        results = []

        for pair in pairs:
            print(f"\nTesting {pair}...")

            request = PredictionRequest(
                currency_pair=pair,
                horizons=[1, 7],
                include_quantiles=False,
                include_direction_probabilities=True,
                max_age_hours=24,
                correlation_id=f"test_{pair.replace('/', '_')}"
            )

            response = await predictor.predict(request)

            print(f"  Status: {response.status}")
            print(f"  Confidence: {response.confidence:.2%}")

            if response.predictions:
                for horizon, pred in response.predictions.items():
                    print(f"  {horizon}d: {pred.mean_change_pct:+.4f} (dir: {pred.direction_probability:.2%})")

            results.append(response.status == "success")

        success_rate = sum(results) / len(results) * 100
        print(f"\n✓ Success rate: {success_rate:.0f}% ({sum(results)}/{len(results)})")

        return success_rate > 0

    except Exception as e:
        print(f"\n✗ Multiple pairs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🚀 Testing Advanced Ensemble Predictor Integration\n")

    results = []

    # Test 1: Model loading
    result1 = await test_model_loading()
    results.append(("Model Loading", result1))

    if result1:
        # Test 2: Single prediction
        result2 = await test_prediction()
        results.append(("Sample Prediction", result2))

        # Test 3: Multiple pairs
        result3 = await test_multiple_pairs()
        results.append(("Multiple Pairs", result3))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    total_pass = sum(1 for _, r in results if r)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")

    return total_pass == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
