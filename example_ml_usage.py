#!/usr/bin/env python3
"""
Example usage of the ML prediction system
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ml import MLPredictor, MLConfig
from src.ml.prediction.predictor import MLPredictionRequest


async def basic_prediction_example():
    """Basic prediction example"""
    print("🔮 Basic ML Prediction Example")
    print("-" * 40)
    
    # Initialize predictor with configuration from YAML
    from src.ml.config import load_ml_config
    config = load_ml_config('ml_config.yaml')
    predictor = MLPredictor(config)
    
    try:
        # Create prediction request
        request = MLPredictionRequest(
            currency_pair="USD/EUR",
            horizons=[1, 7, 30],  # 1 day, 1 week, 1 month
            include_confidence=True,
            include_direction_prob=True
        )
        
        print(f"Making prediction for {request.currency_pair}...")
        
        # Make prediction
        response = await predictor.predict(request)
        
        print("✅ Prediction successful!")
        print(f"Model: {response.model_id}")
        print(f"Timestamp: {response.timestamp}")
        print(f"Model confidence: {response.model_confidence:.3f}")
        print(f"Processing time: {response.processing_time_ms:.1f}ms")
        print(f"Features used: {response.features_used}")
        print(f"Cached: {response.cached}")
        
        print("\nPredictions by horizon:")
        for horizon, pred_data in response.predictions.items():
            direction_prob = response.direction_probabilities.get(horizon, 0.5)
            direction = "📈 UP" if direction_prob > 0.5 else "📉 DOWN"
            confidence = abs(direction_prob - 0.5) * 2  # 0 to 1 scale
            
            print(f"  {horizon:>3}: mean={pred_data['mean']:+.6f} "
                  f"[{pred_data['p10']:+.6f}, {pred_data['p90']:+.6f}] "
                  f"{direction} ({confidence:.1%})")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("\n💡 This might happen if:")
        print("   - No trained model exists for USD/EUR")
        print("   - Layer 1 data is not available")
        print("   - Try running training example first")


async def training_example():
    """Model training example"""
    print("\n🧠 Model Training Example")
    print("-" * 40)
    
    from src.ml.config import load_ml_config
    config = load_ml_config('ml_config.yaml')
    predictor = MLPredictor(config)
    
    try:
        print("Training LSTM model for USD/EUR...")
        print("This may take a few minutes...")
        
        # Train model
        result = predictor.train_model(
            currency_pair="USD/EUR",
            days=200,  # Use 200 days of historical data
            save_model=True,
            set_as_default=True
        )
        
        print("✅ Training completed!")
        print(f"Model ID: {result['model_id']}")
        print(f"Training samples: {result['training_samples']}")
        print(f"Features: {result['features_count']}")
        
        # Display performance metrics
        metrics = result['performance_metrics']
        print("\nPerformance metrics:")
        print(f"  RMSE: {metrics.get('rmse', 0):.6f}")
        print(f"  MAE: {metrics.get('mae', 0):.6f}")
        print(f"  R²: {metrics.get('r2', 0):.4f}")
        print(f"  Directional accuracy: {metrics.get('directional_accuracy', 0):.1%}")
        
        # Training history
        history = result['training_history']
        final_loss = history['loss'][-1] if history['loss'] else 0
        print(f"  Final training loss: {final_loss:.6f}")
        print(f"  Training epochs: {len(history['loss'])}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 This might happen if:")
        print("   - Insufficient historical data")
        print("   - Layer 1 data providers not available")
        print("   - Memory/compute limitations")


async def model_management_example():
    """Model management example"""
    print("\n📊 Model Management Example")
    print("-" * 40)
    
    from src.ml.config import load_ml_config
    config = load_ml_config('ml_config.yaml')
    predictor = MLPredictor(config)
    
    try:
        # List available models
        models = predictor.get_available_models()
        print(f"Available models: {len(models)}")
        
        for model in models[:5]:  # Show first 5
            status = "🟢 DEFAULT" if model.get('is_default', False) else "⚪ Available"
            print(f"  {status} {model['model_id']}")
            print(f"    Currency: {model['currency_pair']}")
            print(f"    Created: {model['created']}")
            print(f"    Type: {model.get('model_type', 'unknown')}")
            print()
        
        # System status
        status = predictor.get_system_status()
        print("System status:")
        print(f"  Models loaded in memory: {status['models_loaded']}")
        print(f"  Total available models: {status['available_models']}")
        print(f"  Cache entries: {status['cache_size']}")
        print(f"  Device: {status['config']['device']}")
        
        # Storage stats
        storage_stats = status['storage_stats']
        print(f"  Storage: {storage_stats['total_models']} models, "
              f"{storage_stats['storage_size_mb']:.1f}MB")
        
    except Exception as e:
        print(f"❌ Model management failed: {e}")


async def backtesting_example():
    """Backtesting example"""
    print("\n📈 Backtesting Example")
    print("-" * 40)
    
    try:
        from src.ml.backtesting import MLBacktester
        
        from src.ml.config import load_ml_config
        config = load_ml_config('ml_config.yaml')
        backtester = MLBacktester(config)
        
        print("Running walk-forward backtest...")
        print("This tests how the model would have performed historically")
        
        # Define test period (last 2 weeks)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        print(f"Test period: {start_date.date()} to {end_date.date()}")
        
        # Run backtest
        result = backtester.run_walkforward_backtest(
            currency_pair="USD/EUR",
            start_date=start_date,
            end_date=end_date,
            initial_train_days=180,
            retraining_frequency=7,  # Retrain weekly
            prediction_horizons=[1, 7]  # 1 day and 1 week
        )
        
        print("✅ Backtest completed!")
        print(f"Total predictions: {result.total_predictions}")
        
        # Performance metrics
        metrics = result.performance_metrics
        print("\nPerformance summary:")
        
        for key in ['avg_directional_accuracy', 'avg_rmse', 'avg_mae']:
            if key in metrics:
                value = metrics[key]
                if 'accuracy' in key:
                    print(f"  {key}: {value:.1%}")
                else:
                    print(f"  {key}: {value:.6f}")
        
        # Horizon-specific metrics
        for horizon in [1, 7]:
            acc_key = f'{horizon}d_directional_accuracy'
            if acc_key in metrics:
                print(f"  {horizon}-day accuracy: {metrics[acc_key]:.1%}")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        print("Note: Backtesting requires sufficient historical data")


async def feature_importance_example():
    """Feature importance example"""
    print("\n🔍 Feature Importance Example")
    print("-" * 40)
    
    try:
        from src.ml.config import load_ml_config
        config = load_ml_config('ml_config.yaml')
        predictor = MLPredictor(config)
        
        # Get feature importance
        print("Calculating feature importance for USD/EUR model...")
        
        importance = predictor.get_feature_importance("USD/EUR")
        
        print("✅ Feature importance calculated!")
        print("Top 10 most important features:")
        
        # Display top features
        for i, (feature_name, importance_score) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {feature_name:<25} {importance_score:.6f}")
        
        # Feature categories
        from src.ml.features.engineering import FeatureEngineer
        engineer = FeatureEngineer()
        categories = engineer.get_feature_importance_names()
        
        print("\nFeature importance by category:")
        for category, feature_names in categories.items():
            if feature_names:
                category_importance = sum(importance.get(name, 0) for name in feature_names)
                print(f"  {category:<12}: {category_importance:.6f}")
        
    except Exception as e:
        print(f"❌ Feature importance failed: {e}")
        print("Note: Requires a trained model for the currency pair")


async def main():
    """Run all examples"""
    print("🚀 ML Prediction System Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Prediction", basic_prediction_example),
        ("Model Training", training_example),
        ("Model Management", model_management_example),
        ("Backtesting", backtesting_example),
        ("Feature Importance", feature_importance_example)
    ]
    
    for example_name, example_func in examples:
        print(f"\n📋 {example_name}")
        try:
            await example_func()
        except KeyboardInterrupt:
            print(f"\n⏸️  {example_name} interrupted")
            break
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
        
        # Pause between examples
        print("\n" + "="*50)
    
    print("\n✨ Examples completed!")
    print("\n💡 Tips:")
    print("   - Run training example first to create models")
    print("   - Ensure Layer 1 data providers are working")
    print("   - Check logs for detailed error information")


if __name__ == "__main__":
    asyncio.run(main())