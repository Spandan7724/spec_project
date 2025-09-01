#!/usr/bin/env python3
"""
Integration test script for ML prediction system
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ml import MLPredictor, MLConfig
from src.ml.prediction.predictor import MLPredictionRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_integration():
    """Test Layer 1 data integration"""
    print("🔗 Testing Layer 1 Data Integration...")
    
    try:
        from src.ml.utils.data_loader import Layer1DataLoader
        
        loader = Layer1DataLoader()
        
        # Test historical rates
        print("  Loading historical rates...")
        prices = loader.load_historical_rates("USD/EUR", days=90)
        print(f"    ✅ Loaded {len(prices)} price records")
        
        # Test technical indicators  
        print("  Loading technical indicators...")
        indicators = loader.load_technical_indicators("USD/EUR", days=90)
        print(f"    ✅ Loaded {len(indicators.columns)} technical indicators")
        
        # Test economic events
        print("  Loading economic events...")
        events = loader.load_economic_events(days=90)
        print(f"    ✅ Loaded {len(events)} economic events")
        
        # Test combined dataset
        print("  Loading combined dataset...")
        prices, indicators, events = loader.get_combined_dataset("USD/EUR", days=90)
        print(f"    ✅ Combined dataset: {prices.shape[0]} days")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Data integration failed: {e}")
        return False


async def test_feature_engineering():
    """Test feature engineering pipeline"""
    print("\n🔧 Testing Feature Engineering...")
    
    try:
        from src.ml.features.engineering import FeatureEngineer
        from src.ml.utils.data_loader import Layer1DataLoader
        
        loader = Layer1DataLoader()
        engineer = FeatureEngineer()
        
        # Load test data
        prices, indicators, events = loader.get_combined_dataset("USD/EUR", days=120)
        
        # Engineer features
        print("  Creating features...")
        features = engineer.engineer_features(prices, indicators, events)
        print(f"    ✅ Created {features.shape[1]} features from {features.shape[0]} days")
        
        # Create targets
        print("  Creating targets...")
        targets = engineer.get_target_variables(prices, [1, 7, 30])
        print(f"    ✅ Created targets: {targets.shape}")
        
        # Check data quality
        print("  Checking data quality...")
        from src.ml.features.preprocessing import DataPreprocessor
        quality = DataPreprocessor.validate_data_quality(features, targets)
        
        missing_pct = quality['features']['missing_percentage']
        print(f"    ✅ Data quality: {missing_pct:.1f}% missing values")
        
        if quality['recommendations']:
            print(f"    ⚠️  Recommendations: {quality['recommendations']}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Feature engineering failed: {e}")
        return False


async def test_model_training():
    """Test LSTM model training"""
    print("\n🧠 Testing LSTM Model Training...")
    
    try:
        from src.ml.models.lstm_model import LSTMModel
        from src.ml.features.engineering import FeatureEngineer
        from src.ml.features.preprocessing import DataPreprocessor
        from src.ml.utils.data_loader import Layer1DataLoader
        
        # Prepare data
        loader = Layer1DataLoader()
        engineer = FeatureEngineer()
        preprocessor = DataPreprocessor()
        
        prices, indicators, events = loader.get_combined_dataset("USD/EUR", days=200)
        features = engineer.engineer_features(prices, indicators, events)
        targets = engineer.get_target_variables(prices, [1, 7])  # Shorter horizons for testing
        
        print(f"  Preparing training data: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Prepare data for training
        data_splits = preprocessor.prepare_data(features, targets, test_size=0.2, validation_size=0.2)
        
        print(f"  Data splits - Train: {data_splits['X_train'].shape[0]}, "
              f"Val: {data_splits['X_val'].shape[0]}, Test: {data_splits['X_test'].shape[0]}")
        
        # Initialize model with smaller config for testing
        model_config = {
            'input_size': features.shape[1],
            'hidden_size': 32,  # Smaller for testing
            'num_layers': 1,
            'dropout': 0.1,
            'prediction_horizons': [1, 7],
            'learning_rate': 0.01,
            'device': 'cpu'
        }
        
        model = LSTMModel(model_config)
        print(f"  Model initialized: {model.get_model_info()['parameter_count']} parameters")
        
        # Train model (reduced epochs for testing)
        print("  Training model...")
        start_time = time.time()
        
        history = model.fit(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val'],
            epochs=5,  # Very short for testing
            batch_size=16,
            verbose=False
        )
        
        train_time = time.time() - start_time
        print(f"    ✅ Training completed in {train_time:.1f}s")
        print(f"    Final loss: {history['loss'][-1]:.6f}")
        
        # Test prediction
        print("  Testing prediction...")
        prediction_result = model.predict(data_splits['X_test'][:1])
        
        print(f"    ✅ Prediction successful: {len(prediction_result.predictions)} horizons")
        print(f"    Model confidence: {prediction_result.model_confidence:.3f}")
        
        # Evaluate model
        print("  Evaluating model...")
        metrics = model.evaluate(data_splits['X_test'], data_splits['y_test'])
        
        print(f"    ✅ RMSE: {metrics['rmse']:.6f}")
        print(f"    ✅ Directional accuracy: {metrics['directional_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prediction_api():
    """Test the prediction API"""
    print("\n🔮 Testing Prediction API...")
    
    try:
        # Initialize predictor
        config = MLConfig.get_default()
        predictor = MLPredictor(config)
        
        # First train a model
        print("  Training model for API test...")
        training_result = predictor.train_model(
            currency_pair="USD/EUR",
            days=150,  # Smaller dataset for testing
            save_model=True,
            set_as_default=True
        )
        
        model_id = training_result['model_id']
        print(f"    ✅ Model trained: {model_id}")
        print(f"    Training samples: {training_result['training_samples']}")
        print(f"    Features: {training_result['features_count']}")
        
        # Test prediction
        print("  Making prediction...")
        request = MLPredictionRequest(
            currency_pair="USD/EUR",
            horizons=[1, 7, 30],
            include_confidence=True,
            include_direction_prob=True
        )
        
        response = await predictor.predict(request)
        
        print("    ✅ Prediction successful!")
        print(f"    Model: {response.model_id}")
        print(f"    Model confidence: {response.model_confidence:.3f}")
        print(f"    Processing time: {response.processing_time_ms:.1f}ms")
        print(f"    Features used: {response.features_used}")
        
        # Display predictions
        for horizon, pred_data in response.predictions.items():
            direction_prob = response.direction_probabilities.get(horizon, 0.5)
            print(f"    {horizon}: mean={pred_data['mean']:.6f}, "
                  f"p90={pred_data['p90']:.6f}, direction_up={direction_prob:.3f}")
        
        # Test caching
        print("  Testing prediction caching...")
        start_time = time.time()
        cached_response = await predictor.predict(request)
        cached_time = time.time() - start_time
        
        if cached_response.cached:
            print(f"    ✅ Cache hit in {cached_time*1000:.1f}ms")
        else:
            print("    ⚠️  Cache miss (expected for first run)")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Prediction API failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_backtesting():
    """Test backtesting framework"""
    print("\n📊 Testing Backtesting Framework...")
    
    try:
        from src.ml.backtesting import MLBacktester
        
        config = MLConfig.get_default()
        backtester = MLBacktester(config)
        
        # Run a short backtest
        print("  Running walk-forward backtest...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Very short for testing
        
        result = backtester.run_walkforward_backtest(
            currency_pair="USD/EUR",
            start_date=start_date,
            end_date=end_date,
            initial_train_days=120,  # Smaller training set
            retraining_frequency=15,  # More frequent retraining
            prediction_horizons=[1, 7]  # Fewer horizons for testing
        )
        
        print("    ✅ Backtest completed!")
        print(f"    Currency pair: {result.currency_pair}")
        print(f"    Test period: {result.test_period}")
        print(f"    Total predictions: {result.total_predictions}")
        
        if result.performance_metrics:
            # Display key metrics
            for key in ['avg_directional_accuracy', 'avg_rmse', 'avg_mae', 'total_predictions']:
                if key in result.performance_metrics:
                    value = result.performance_metrics[key]
                    print(f"    {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_storage():
    """Test model storage and management"""
    print("\n💾 Testing Model Storage...")
    
    try:
        from src.ml.utils.model_storage import ModelStorage
        
        storage = ModelStorage("test_models/")
        
        # Check existing models
        models = storage.list_models()
        print(f"  Available models: {len(models)}")
        
        # Display model info
        for model in models[:3]:  # Show first 3
            print(f"    {model['model_id']}: {model['currency_pair']} "
                  f"({model.get('model_type', 'unknown')})")
        
        # Test storage stats
        stats = storage.get_storage_stats()
        print(f"    ✅ Storage stats: {stats['total_models']} models, "
              f"{stats['storage_size_mb']:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Model storage test failed: {e}")
        return False


async def test_system_performance():
    """Test system performance and resource usage"""
    print("\n⚡ Testing System Performance...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"  Memory usage: {memory_mb:.1f}MB")
        
        # Test prediction speed
        config = MLConfig.get_default()
        predictor = MLPredictor(config)
        
        # Get system status
        status = predictor.get_system_status()
        
        print(f"  Models loaded: {status['models_loaded']}")
        print(f"  Available models: {status['available_models']}")
        print(f"  Cache size: {status['cache_size']}")
        print(f"  Device: {status['config']['device']}")
        
        print("    ✅ System performance check completed")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Performance test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Starting ML System Integration Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Run tests
    tests = [
        ("Data Integration", test_data_integration),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Prediction API", test_prediction_api),
        ("Backtesting", test_backtesting),
        ("Model Storage", test_model_storage),
        ("System Performance", test_system_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<20}: {status}")
    
    print(f"\nResult: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! ML system is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)