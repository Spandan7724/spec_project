#!/usr/bin/env python3
"""
Test script to capture the exact MLPredictionResponse format that agents receive
This shows the actual output structure sent to agents for verification
"""

import asyncio
import sys
import json
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ml import MLPredictor, MLConfig
from src.ml.prediction.predictor import MLPredictionRequest


async def capture_agent_output():
    """
    Capture the exact MLPredictionResponse that agents receive
    This simulates what an agent would get when requesting ML predictions
    """
    print("🤖 CAPTURING AGENT ML OUTPUT")
    print("=" * 60)
    print("This shows the EXACT format agents receive for verification\n")
    
    # Initialize ML predictor (same as agents would)
    from src.ml.config import load_ml_config
    config = load_ml_config('ml_config.yaml')
    predictor = MLPredictor(config)
    
    try:
        # Step 1: Train a model first (agents won't do this, but needed for testing)
        print("📋 Step 1: Training model (agents don't do this)...")
        training_result = predictor.train_model(
            currency_pair="USD/EUR",
            days=150,
            save_model=True,
            set_as_default=True
        )
        print(f"✅ Model trained: {training_result['model_id']}")
        
        # Step 2: Create prediction request (exactly what agents do)
        print("\n📋 Step 2: Agent creates prediction request...")
        request = MLPredictionRequest(
            currency_pair="USD/EUR",
            horizons=[1, 7, 30],  # 1 day, 1 week, 1 month
            include_confidence=True,
            include_direction_prob=True
        )
        
        print("Agent request:")
        request_dict = asdict(request)
        print(json.dumps(request_dict, indent=2))
        
        # Step 3: Make prediction (exactly what agents do)
        print("\n📋 Step 3: Agent calls ML prediction...")
        response = await predictor.predict(request)
        
        # Step 4: Show exact response format agents receive
        print("\n🎯 EXACT AGENT OUTPUT FORMAT:")
        print("=" * 60)
        
        # Convert to dictionary for JSON serialization
        response_dict = asdict(response)
        
        # Pretty print the JSON that agents would receive
        print("Raw MLPredictionResponse object as JSON:")
        print(json.dumps(response_dict, indent=2, default=str))
        
        # Step 5: Show human-readable interpretation
        print("\n📊 HUMAN-READABLE INTERPRETATION:")
        print("=" * 60)
        print(f"Currency Pair: {response.currency_pair}")
        print(f"Timestamp: {response.timestamp}")
        print(f"Model ID: {response.model_id}")
        print(f"Model Confidence: {response.model_confidence:.3f}")
        print(f"Processing Time: {response.processing_time_ms:.1f}ms")
        print(f"Features Used: {response.features_used}")
        print(f"From Cache: {response.cached}")
        
        print("\nDetailed Predictions:")
        for horizon, pred_data in response.predictions.items():
            direction_prob = response.direction_probabilities.get(horizon, 0.5)
            direction = "📈 UP" if direction_prob > 0.5 else "📉 DOWN"
            confidence = abs(direction_prob - 0.5) * 2  # 0 to 1 scale
            
            print(f"\n{horizon} prediction:")
            print(f"  Mean forecast: {pred_data['mean']:+.6f}")
            print(f"  Confidence interval: [{pred_data['p10']:+.6f}, {pred_data['p90']:+.6f}]")
            print(f"  Direction: {direction}")
            print(f"  Direction probability: {direction_prob:.3f}")
            print(f"  Direction confidence: {confidence:.1%}")
        
        # Step 6: Test caching behavior
        print("\n📋 Step 4: Testing cached response (agents benefit from this)...")
        cached_response = await predictor.predict(request)
        print(f"✅ Cache hit: {cached_response.cached}")
        print(f"Cache response time: {cached_response.processing_time_ms:.1f}ms")
        
        # Step 7: Show what agents can extract for decision making
        print("\n🧠 AGENT DECISION MAKING DATA:")
        print("=" * 60)
        print("Key data points agents can use:")
        
        for horizon, pred_data in response.predictions.items():
            direction_prob = response.direction_probabilities.get(horizon, 0.5)
            
            print(f"\n{horizon}:")
            print(f"  Expected return: {pred_data['mean']:+.6f}")
            print(f"  Worst case (p10): {pred_data['p10']:+.6f}")  
            print(f"  Best case (p90): {pred_data['p90']:+.6f}")
            print(f"  Probability of increase: {direction_prob:.1%}")
            print(f"  Risk/reward ratio: {abs(pred_data['p90']/pred_data['p10']):.2f}")
        
        print(f"\nOverall model reliability: {response.model_confidence:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to capture agent output: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_currencies():
    """Test predictions for multiple currency pairs that agents might request"""
    print("\n\n🌍 TESTING MULTIPLE CURRENCY PAIRS")
    print("=" * 60)
    
    currency_pairs = ["USD/EUR", "EUR/USD", "GBP/USD"]
    from src.ml.config import load_ml_config
    predictor = MLPredictor(load_ml_config('ml_config.yaml'))
    
    for pair in currency_pairs:
        print(f"\n📈 Testing {pair}...")
        
        try:
            # Check if model exists, if not train one
            models = predictor.get_available_models()
            pair_models = [m for m in models if m['currency_pair'] == pair]
            
            if not pair_models:
                print(f"  Training model for {pair}...")
                training_result = predictor.train_model(
                    currency_pair=pair,
                    days=120,
                    save_model=True,
                    set_as_default=True
                )
                print(f"  ✅ Trained: {training_result['model_id']}")
            else:
                print(f"  ✅ Using existing model: {pair_models[0]['model_id']}")
            
            # Make prediction
            request = MLPredictionRequest(
                currency_pair=pair,
                horizons=[1, 7],  # Shorter for testing
                include_confidence=True,
                include_direction_prob=True
            )
            
            response = await predictor.predict(request)
            
            print(f"  Prediction - Model confidence: {response.model_confidence:.3f}")
            for horizon, pred_data in response.predictions.items():
                direction_prob = response.direction_probabilities.get(horizon, 0.5)
                direction = "📈" if direction_prob > 0.5 else "📉"
                print(f"    {horizon}: {pred_data['mean']:+.6f} {direction} ({direction_prob:.1%})")
                
        except Exception as e:
            print(f"  ❌ Failed for {pair}: {e}")


async def main():
    """Main test function"""
    success = await capture_agent_output()
    
    if success:
        await test_multiple_currencies()
        
        print("\n\n✅ SUMMARY")
        print("=" * 60)
        print("✅ Successfully captured ML prediction output format")
        print("✅ Agents receive MLPredictionResponse objects with:")
        print("   - Currency pair and timestamp")
        print("   - Model ID and confidence score")
        print("   - Predictions for each horizon (mean, p10, p50, p90)")
        print("   - Direction probabilities (probability of price increase)")
        print("   - Processing metadata (time, features used, cache status)")
        print("✅ System ready for agent integration")
        
    else:
        print("\n❌ Failed to capture agent output format")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
