"""
Test ML forecasting with real historical FX data from Yahoo Finance.

This script downloads real historical data and trains the LSTM model,
providing comprehensive metrics including MAE, MSE, RMSE, and more.
"""

import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.yfinance_collector import YFinanceDataCollector, download_fx_data_for_training
from ml.trainer import ModelTrainer, TrainingConfig
from ml.models import ModelConfig, ModelEvaluator
from ml.features import FeatureConfig
from ml.forecaster import FXForecaster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_prepare_data(currency_pairs: list = None, period: str = "1y"):
    """Download historical data from Yahoo Finance."""
    
    print("📊 Downloading Historical FX Data from Yahoo Finance")
    print("=" * 60)
    
    if currency_pairs is None:
        currency_pairs = ['USD/EUR', 'USD/GBP', 'EUR/GBP']
    
    try:
        collector = YFinanceDataCollector()
        
        # Check if data already exists
        available_data = collector.get_available_data()
        if available_data:
            print("📁 Found existing data:")
            for data_info in available_data:
                print(f"   {data_info['currency_pair']}: {data_info['records']} records "
                      f"({data_info['start_date'].date()} to {data_info['end_date'].date()})")
            
            # Ask if we should use existing or download fresh
            print("\nUsing existing data for faster testing...")
            
            # Load existing data
            data = {}
            for pair in currency_pairs:
                df = collector.load_historical_data(pair, period, "1h")
                if df is not None:
                    data[pair] = df
            
            return data
        
        # Download fresh data
        print(f"🔄 Downloading {len(currency_pairs)} currency pairs...")
        print(f"   Period: {period} (hourly data)")
        
        data = collector.download_historical_data(
            currency_pairs=currency_pairs,
            period=period,
            interval="1h",
            save=True
        )
        
        if not data:
            print("❌ No data downloaded")
            return None
        
        # Print detailed summary
        print("\n📈 Data Summary:")
        for pair, df in data.items():
            summary = collector.get_data_summary(df)
            print(f"\n{pair}:")
            print(f"   Records: {summary['total_records']:,}")
            print(f"   Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            print(f"   Duration: {summary['date_range']['days']} days")
            print(f"   Rate Statistics:")
            print(f"     Min: {summary['rate_statistics']['min']:.6f}")
            print(f"     Max: {summary['rate_statistics']['max']:.6f}")
            print(f"     Mean: {summary['rate_statistics']['mean']:.6f}")
            print(f"     Std: {summary['rate_statistics']['std']:.6f}")
            print(f"   Data Quality:")
            print(f"     Missing Values: {summary['missing_values']}")
            print(f"     Avg Records/Day: {summary['data_quality']['avg_records_per_day']:.1f}")
        
        return data
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        print(f"❌ Data download failed: {e}")
        return None


def train_model_with_real_data(df: pd.DataFrame, currency_pair: str):
    """Train LSTM model with real historical data."""
    
    print(f"\n🧠 Training ML Model for {currency_pair}")
    print("=" * 50)
    
    try:
        print(f"📊 Training data: {len(df)} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Configure for real data training
        model_config = ModelConfig(
            sequence_length=168,  # 1 week of hourly data
            prediction_horizon=24,  # Predict 24 hours ahead
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        feature_config = FeatureConfig(
            rsi_period=14,
            ma_short_period=12, 
            ma_long_period=26,
            bb_period=20,
            volatility_windows=[6, 12, 24, 48],
            lag_periods=[1, 2, 3, 6, 12, 24]
        )
        
        training_config = TrainingConfig(
            model_config=model_config,
            feature_config=feature_config,
            max_epochs=50,  # More epochs for real data
            batch_size=32,
            learning_rate=0.001,
            patience=10,
            validation_split=0.2
        )
        
        print(f"🏗️  Model Configuration:")
        print(f"   Sequence Length: {model_config.sequence_length} hours")
        print(f"   Prediction Horizon: {model_config.prediction_horizon} hours") 
        print(f"   Hidden Size: {model_config.hidden_size}")
        print(f"   Layers: {model_config.num_layers}")
        print(f"   Max Epochs: {training_config.max_epochs}")
        
        # Initialize trainer
        trainer = ModelTrainer(training_config)
        
        print("🔄 Preparing training data...")
        X, y = trainer.prepare_data(df)
        
        print(f"   Training sequences: {X.shape[0]}")
        print(f"   Input shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Features per timestep: {X.shape[2]}")
        
        if X.shape[0] < 100:
            print("⚠️  Warning: Very few training sequences. Consider more data or shorter sequence length.")
        
        print("\n🚀 Starting model training...")
        training_result = trainer.train_model(X, y)
        
        print(f"\n✅ Training completed!")
        print(f"   Training time: {training_result.training_time:.2f} seconds")
        print(f"   Best epoch: {training_result.best_epoch}")
        print(f"   Converged: {training_result.converged}")
        print(f"   Final train loss: {training_result.train_losses[-1]:.6f}")
        print(f"   Final val loss: {training_result.val_losses[-1]:.6f}")
        
        # Comprehensive metrics
        print(f"\n📊 Model Performance Metrics:")
        val_metrics = training_result.val_metrics
        
        print(f"   MSE (Mean Squared Error): {val_metrics['mse']:.8f}")
        print(f"   MAE (Mean Absolute Error): {val_metrics['mae']:.6f}")
        print(f"   RMSE (Root Mean Squared Error): {val_metrics['rmse']:.6f}")
        print(f"   MAPE (Mean Absolute Percentage Error): {val_metrics['mape']:.2f}%")
        print(f"   SMAPE (Symmetric MAPE): {val_metrics.get('smape', 'N/A'):.2f}%")
        print(f"   MASE (Mean Absolute Scaled Error): {val_metrics.get('mase', 'N/A'):.4f}")
        print(f"   R² (Coefficient of Determination): {val_metrics['r2']:.4f}")
        print(f"   Directional Accuracy: {val_metrics['directional_accuracy']:.2f}%")
        print(f"   Max Error: {val_metrics.get('max_error', 'N/A'):.6f}")
        print(f"   Median Absolute Error: {val_metrics.get('median_absolute_error', 'N/A'):.6f}")
        
        # Save trained model
        model_path = f"models/{currency_pair.replace('/', '_')}_real_data.pkl"
        trainer.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        return trainer, training_result
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print(f"❌ Model training failed: {e}")
        return None, None


def test_predictions_with_real_data(trainer, df: pd.DataFrame, currency_pair: str):
    """Test predictions on recent real data."""
    
    print(f"\n🔮 Testing Predictions for {currency_pair}")
    print("=" * 40)
    
    try:
        # Use last 7 days of data for prediction testing
        test_data = df.tail(7 * 24).copy()  # Last 7 days
        
        if len(test_data) < trainer.config.model_config.sequence_length:
            print("⚠️  Insufficient test data")
            return
        
        # Prepare features for the test data
        df_features = trainer.feature_engineer.prepare_features(test_data)
        feature_columns = trainer.feature_engineer.get_feature_columns(df_features)
        features_scaled = trainer.feature_engineer.transform_features(df_features)
        
        # Create sequence for prediction
        sequence = trainer.sequence_generator.create_single_sequence(features_scaled)
        
        # Make prediction
        import torch
        trainer.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(trainer.device)
            
            # Get standard prediction
            pred_mean, pred_var = trainer.model(sequence_tensor)
            
            # Get prediction with uncertainty
            pred_mean_mc, epistemic_unc, aleatoric_unc = trainer.model.predict_with_uncertainty(
                sequence_tensor, num_samples=100
            )
        
        # Convert to numpy
        predictions = pred_mean.cpu().numpy().flatten()
        predictions_mc = pred_mean_mc.cpu().numpy().flatten()
        epistemic_uncertainty = epistemic_unc.cpu().numpy().flatten()
        aleatoric_uncertainty = aleatoric_unc.cpu().numpy().flatten()
        
        # Get current and recent rates
        current_rate = test_data['rate'].iloc[-1]
        recent_rates = test_data['rate'].tail(24).values  # Last 24 hours
        
        print(f"📈 Current Rate: {current_rate:.6f}")
        print(f"📊 Recent 24h Range: {recent_rates.min():.6f} - {recent_rates.max():.6f}")
        print(f"📊 Recent 24h Change: {((current_rate - recent_rates[0]) / recent_rates[0] * 100):+.2f}%")
        
        print(f"\n🔮 Predictions (next 24 hours):")
        print(f"   1-hour ahead: {predictions[0]:.6f} (uncertainty: ±{np.sqrt(epistemic_uncertainty[0] + aleatoric_uncertainty[0]):.6f})")
        print(f"   6-hour ahead: {predictions[5]:.6f} (uncertainty: ±{np.sqrt(epistemic_uncertainty[5] + aleatoric_uncertainty[5]):.6f})")
        print(f"   12-hour ahead: {predictions[11]:.6f} (uncertainty: ±{np.sqrt(epistemic_uncertainty[11] + aleatoric_uncertainty[11]):.6f})")
        print(f"   24-hour ahead: {predictions[23]:.6f} (uncertainty: ±{np.sqrt(epistemic_uncertainty[23] + aleatoric_uncertainty[23]):.6f})")
        
        # Calculate expected changes
        changes = ((predictions - current_rate) / current_rate) * 100
        print(f"\n📊 Expected Changes:")
        print(f"   1-hour: {changes[0]:+.3f}%")
        print(f"   6-hour: {changes[5]:+.3f}%") 
        print(f"   12-hour: {changes[11]:+.3f}%")
        print(f"   24-hour: {changes[23]:+.3f}%")
        
        # Trend analysis
        trend_24h = "UP" if changes[23] > 0.1 else "DOWN" if changes[23] < -0.1 else "STABLE"
        print(f"\n📈 24-hour Trend: {trend_24h}")
        
        # Uncertainty analysis
        avg_epistemic = np.mean(epistemic_uncertainty[:24])
        avg_aleatoric = np.mean(aleatoric_uncertainty[:24])
        total_uncertainty = np.sqrt(avg_epistemic + avg_aleatoric)
        
        print(f"\n🎯 Uncertainty Analysis:")
        print(f"   Model Uncertainty (Epistemic): {avg_epistemic:.6f}")
        print(f"   Data Uncertainty (Aleatoric): {avg_aleatoric:.6f}")
        print(f"   Total Uncertainty: {total_uncertainty:.6f}")
        
        # Confidence assessment
        confidence = max(0.0, min(1.0, 1.0 - total_uncertainty))
        print(f"   Model Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
        
        return predictions, predictions_mc, epistemic_uncertainty, aleatoric_uncertainty
        
    except Exception as e:
        logger.error(f"Prediction testing failed: {e}")
        print(f"❌ Prediction testing failed: {e}")
        return None


async def main():
    """Main test function with real data."""
    
    print("Currency Assistant - ML Training with Real Data")
    print("=" * 70)
    
    # Configuration
    currency_pairs = ['USD/EUR', 'USD/GBP', 'EUR/GBP']
    period = "1y"  # 1 year of historical data
    
    try:
        # Step 1: Download historical data
        data = download_and_prepare_data(currency_pairs, period)
        
        if not data:
            print("❌ Failed to get historical data")
            return
        
        # Step 2: Train models for each currency pair
        trained_models = {}
        
        for pair, df in data.items():
            if len(df) < 1000:  # Need sufficient data
                print(f"⚠️  Skipping {pair}: insufficient data ({len(df)} records)")
                continue
            
            print(f"\n" + "="*70)
            trainer, result = train_model_with_real_data(df, pair)
            
            if trainer and result:
                trained_models[pair] = trainer
                
                # Test predictions
                predictions = test_predictions_with_real_data(trainer, df, pair)
                
                print(f"\n✅ {pair} model training and testing completed")
            else:
                print(f"❌ {pair} model training failed")
        
        # Summary
        print(f"\n" + "="*70)
        print("🎉 Training Summary")
        print(f"   Successfully trained: {len(trained_models)}/{len(currency_pairs)} models")
        
        if trained_models:
            print("   Available for predictions:")
            for pair in trained_models:
                print(f"     ✅ {pair}")
            
            print(f"\n📁 Models saved in 'models/' directory")
            print("🔗 Ready for integration with Decision Engine")
        else:
            print("❌ No models were successfully trained")
        
    except Exception as e:
        logger.error(f"Main test failed: {e}")
        print(f"❌ Main test failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")