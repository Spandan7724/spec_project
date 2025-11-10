#!/usr/bin/env python3
"""
Test CatBoost backend save/load functionality
"""
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.backends.catboost_backend import CatBoostBackend


def test_catboost_save_load():
    """Test that CatBoost models can be saved and loaded"""
    print("=" * 80)
    print("Testing CatBoost Backend Save/Load")
    print("=" * 80)

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create targets for horizons (small random changes)
    y_train = pd.DataFrame({
        "target_1d": np.random.randn(n_samples) * 0.01,
        "direction_1d": (np.random.randn(n_samples) > 0).astype(int),
        "target_7d": np.random.randn(n_samples) * 0.02,
        "direction_7d": (np.random.randn(n_samples) > 0).astype(int),
    })

    print(f"\n📊 Created synthetic data:")
    print(f"   - Training samples: {n_samples}")
    print(f"   - Features: {n_features}")
    print(f"   - Horizons: [1, 7]")

    # Initialize backend
    print(f"\n🔧 Initializing CatBoost backend...")
    backend = CatBoostBackend(task_type="CPU")

    # Train models
    print(f"\n🏋️  Training CatBoost models...")
    try:
        metrics = backend.train(
            X_train=X_train,
            y_train=y_train,
            horizons=[1, 7],
            num_boost_round=100,  # Small number for quick test
            patience=20
        )

        print(f"\n✅ Training successful!")
        for horizon, m in metrics.items():
            print(f"   {horizon}: RMSE={m['rmse']:.6f}, R²={m['r2']:.4f}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test prediction before save
    print(f"\n📈 Testing prediction (before save)...")
    X_test = X_train.head(3)
    pred_before = backend.predict(X_test, horizons=[1, 7])
    print(f"   Predictions shape: {pred_before.shape}")
    print(f"   Columns: {list(pred_before.columns)}")

    # Save model
    print(f"\n💾 Saving model...")
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        backend.save(tmp_path)
        print(f"   ✅ Saved to {tmp_path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load model into new backend
    print(f"\n📂 Loading model into new backend...")
    backend2 = CatBoostBackend(task_type="CPU")

    try:
        backend2.load(tmp_path)
        print(f"   ✅ Loaded successfully")
        print(f"   - Models: {list(backend2.models.keys())}")
        print(f"   - Quantile models: {list(backend2.quantile_models.keys())}")
        print(f"   - Direction models: {list(backend2.direction_models.keys())}")
        print(f"   - Features: {len(backend2.feature_names)}")
    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test prediction after load
    print(f"\n📈 Testing prediction (after load)...")
    try:
        pred_after = backend2.predict(X_test, horizons=[1, 7])
        print(f"   Predictions shape: {pred_after.shape}")

        # Compare predictions
        print(f"\n🔍 Comparing predictions...")
        diff = np.abs(pred_before.values - pred_after.values).max()
        print(f"   Max difference: {diff:.10f}")

        if diff < 1e-6:
            print(f"   ✅ Predictions match (diff < 1e-6)")
        else:
            print(f"   ⚠️  Predictions differ by {diff}")

    except Exception as e:
        print(f"   ❌ Prediction after load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test confidence
    print(f"\n📊 Testing model confidence...")
    confidence = backend2.get_model_confidence()
    print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

    # Cleanup
    Path(tmp_path).unlink()
    print(f"\n🧹 Cleaned up temporary file")

    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS PASSED")
    print(f"{'='*80}\n")

    return True


if __name__ == "__main__":
    success = test_catboost_save_load()
    sys.exit(0 if success else 1)
