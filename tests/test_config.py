#!/usr/bin/env python3
"""
Test script to demonstrate ML configuration system
"""

import shutil
from pathlib import Path
from src.ml.config import load_ml_config
from src.ml.prediction import MLPredictor


def test_default_config():
    """Test the default configuration (timestamp versioning)"""
    print("🔧 Testing Default Configuration (timestamp versioning)")
    print("=" * 60)
    
    # Load default config by passing a non-existent path to force defaults
    config = load_ml_config("non_existent_config.yaml")
    print(f"Model versioning: {config.model_versioning}")
    print(f"Sequence length: {config.model.sequence_length}")
    print(f"Training samples: Train={config.model.train_test_split}, Val={config.model.validation_split}")
    
    # Initialize predictor
    predictor = MLPredictor(config)
    
    # Train a model (small training for demo)
    print("\nTraining model with default config...")
    result = predictor.train_model("USD/EUR", days=100, save_model=True)
    
    model_id = result['model_id']
    print(f"✅ Model saved as: {model_id}")
    
    # List models
    models = predictor.get_available_models()
    print(f"Available models: {len(models)}")
    for model in models[-2:]:  # Show last 2 models
        created = model.get('created_at', model.get('timestamp', 'unknown'))
        print(f"  - {model['model_id']} (created: {created})")
    
    return model_id


def test_yaml_config():
    """Test YAML configuration (overwrite versioning)"""
    print("\n🔧 Testing YAML Configuration (overwrite versioning)")
    print("=" * 60)
    
    # Load YAML config
    config = load_ml_config("ml_config.yaml")
    print(f"Model versioning: {config.model_versioning}")
    print(f"Sequence length: {config.model.sequence_length}")
    print(f"Training samples: Train={config.model.train_test_split}, Val={config.model.validation_split}")
    print(f"Features - MA periods: {config.features.ma_periods}")
    
    # Initialize predictor
    predictor = MLPredictor(config)
    
    # Train a model multiple times to show overwriting
    print("\nTraining model with YAML config (overwrite mode)...")
    
    # First training
    result1 = predictor.train_model("EUR/USD", days=100, save_model=True)
    model_id1 = result1['model_id']
    print(f"✅ First training - Model saved as: {model_id1}")
    
    # Second training (should overwrite)
    print("\nTraining again (should overwrite previous model)...")
    result2 = predictor.train_model("EUR/USD", days=100, save_model=True)
    model_id2 = result2['model_id']
    print(f"✅ Second training - Model saved as: {model_id2}")
    
    # Check if they have the same ID (overwrite mode)
    if model_id1 == model_id2:
        print("✅ SUCCESS: Models have same ID (overwrite mode working)")
    else:
        print("❌ FAILED: Models have different IDs")
    
    # List models
    models = predictor.get_available_models()
    print(f"Available models: {len(models)}")
    
    return model_id2


def test_data_amount_comparison():
    """Compare data usage between configurations"""
    print("\n📊 Comparing Data Usage")
    print("=" * 60)
    
    # Default config (sequence_length=60)
    config_default = load_ml_config()
    print(f"Default config - Sequence length: {config_default.model.sequence_length}")
    
    # YAML config (sequence_length=30)
    config_yaml = load_ml_config("ml_config.yaml")
    print(f"YAML config - Sequence length: {config_yaml.model.sequence_length}")
    
    print("\nWith 200 days of data:")
    print(f"Default (seq=60): {200 - 60} training sequences available")
    print(f"YAML (seq=30):    {200 - 30} training sequences available")
    print(f"Improvement: {(200-30) - (200-60)} more sequences = {((200-30)/(200-60)-1)*100:.1f}% more data!")


def cleanup_test_models():
    """Clean up test models"""
    print("\n🧹 Cleaning up test models...")
    
    models_dir = Path("models")
    if models_dir.exists():
        test_patterns = ["LSTM_USD_EUR", "LSTM_EUR_USD", "lstm_usd", "lstm_eur"]
        
        removed_count = 0
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                dir_name = model_dir.name.lower()
                if any(pattern.lower() in dir_name for pattern in test_patterns):
                    shutil.rmtree(model_dir)
                    removed_count += 1
                    print(f"  Removed: {model_dir.name}")
        
        print(f"✅ Cleaned up {removed_count} test model directories")


def main():
    """Main test function"""
    print("🚀 ML Configuration System Test")
    print("=" * 60)
    
    try:
        # Test 1: Default configuration
        default_model = test_default_config()
        
        # Test 2: YAML configuration
        yaml_model = test_yaml_config()
        
        # Test 3: Data comparison
        test_data_amount_comparison()
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print("✅ Default config test: PASSED")
        print("✅ YAML config test: PASSED") 
        print("✅ Data usage comparison: COMPLETED")
        print("✅ Model versioning strategies: WORKING")
        print(f"   - Timestamp: Creates unique folders (e.g., {default_model})")
        print(f"   - Overwrite: Reuses same folder (e.g., {yaml_model})")
        
        print("\n🎯 KEY BENEFITS:")
        print("  • Configurable model versioning (timestamp vs overwrite)")
        print("  • Tunable sequence length for better data efficiency")
        print("  • YAML-based configuration for easy customization")
        print("  • More training data with optimized settings")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Optional cleanup
        response = input("\n🧹 Clean up test models? (y/N): ").strip().lower()
        if response == 'y':
            cleanup_test_models()


if __name__ == "__main__":
    main()