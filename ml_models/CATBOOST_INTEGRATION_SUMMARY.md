# CatBoost Model Integration - Complete Summary

## 🏆 Overview

The **CatBoost model** has been successfully integrated into the Currency Assistant application as a new training option. This model achieved the **best performance** in benchmarking tests with:

- **Test RMSE**: 0.005640
- **Test R²**: 0.988341 (98.83% variance explained!)
- **Test MAE**: 0.004239
- **MAPE**: ~0.38%

CatBoost outperformed all other models including LightGBM, XGBoost, and advanced neural networks (LSTM-Transformer, CNN-LSTM, etc.).

---

## 📁 Files Created/Modified

### New Files:
1. **`src/prediction/backends/catboost_backend.py`**
   - Complete CatBoost backend implementation
   - Follows the same interface as LightGBM and LSTM backends
   - Includes mean regression, quantile regression, and direction classification
   - GPU support with automatic detection
   - Conservative hyperparameters from winning CatBoost_3 configuration

### Modified Files:
1. **`src/prediction/training.py`**
   - Added `train_and_register_catboost()` function
   - Supports all hyperparameter customization

2. **`backend/routes/models.py`**
   - Updated API endpoints to accept `"catboost"` as model_type
   - Added CatBoost-specific parameters to training request
   - Integrated CatBoost training into background task handler

3. **`src/prediction/predictor.py`**
   - Added CatBoost backend initialization
   - Prioritizes CatBoost models over LightGBM (best performance)
   - Updated model loading logic
   - Updated response to show which model was used

---

## 🎯 Features

### 1. **Winning Configuration**
The CatBoost backend uses the conservative configuration that won in benchmarking:

```python
{
    "learning_rate": 0.005,        # Small learning rate
    "depth": 6,                     # Conservative tree depth
    "l2_leaf_reg": 10,             # Strong regularization
    "min_data_in_leaf": 30,        # Prevent overfitting
    "random_strength": 0.1,         # Low randomness
    "bagging_temperature": 0.1,     # Conservative bagging
    "iterations": 4000,             # Many iterations
    "early_stopping_rounds": 300    # Patient early stopping
}
```

### 2. **GPU Acceleration**
- Automatic GPU detection
- Falls back to CPU if GPU not available
- Can be controlled via `task_type` parameter

### 3. **Comprehensive Predictions**
Each model provides:
- **Mean prediction** (primary forecast)
- **Quantile predictions** (p10, p50, p90) for uncertainty estimation
- **Direction classification** (up/down movement probability)
- **Feature importance** for interpretability

### 4. **RobustScaler**
Uses `RobustScaler` instead of `StandardScaler` for better handling of financial data outliers

---

## 🚀 How to Use

### 1. **Via API (Web UI)**

Train a CatBoost model for any currency pair:

```bash
curl -X POST "http://localhost:8000/api/models/train" \
  -H "Content-Type: application/json" \
  -d '{
    "currency_pair": "USD/EUR",
    "model_type": "catboost",
    "horizons": [1, 3, 7],
    "history_days": 1000,
    "version": "1.0",
    "catboost_rounds": 4000,
    "catboost_patience": 300,
    "catboost_learning_rate": 0.005,
    "catboost_depth": 6,
    "catboost_task_type": "auto"
  }'
```

**Parameters**:
- `currency_pair`: Currency pair to train on (e.g., "USD/EUR", "GBP/USD")
- `model_type`: Use `"catboost"` for the best performance
- `horizons`: Prediction horizons in days (e.g., [1, 3, 7, 14])
- `history_days`: Historical data window (default: config value)
- `catboost_rounds`: Max iterations (default: 4000)
- `catboost_patience`: Early stopping patience (default: 300)
- `catboost_learning_rate`: Learning rate (default: 0.005)
- `catboost_depth`: Tree depth (default: 6)
- `catboost_task_type`: "CPU", "GPU", or "auto" (default: "auto")

### 2. **Via Python (Programmatic)**

```python
from src.prediction.training import train_and_register_catboost
from src.prediction.config import PredictionConfig

# Train with default configuration
metadata = await train_and_register_catboost(
    currency_pair="USD/EUR",
    horizons=[1, 3, 7],
    days=1000,
    version="1.0"
)

# Train with custom hyperparameters
metadata = await train_and_register_catboost(
    currency_pair="GBP/USD",
    horizons=[1, 3, 7, 14],
    catboost_rounds=5000,
    catboost_learning_rate=0.003,
    catboost_depth=8,
    task_type="GPU"
)
```

### 3. **Model Selection Priority**

The predictor automatically selects models in this order:
1. **CatBoost** (if available) - Best performance
2. **LightGBM** (if CatBoost unavailable)
3. **LSTM** (for intraday predictions)
4. **Fallback** (heuristics if no models available)

---

## 📊 Training Process

When you train a CatBoost model:

1. **Data Loading**: Fetches historical data for the currency pair
2. **Feature Engineering**: Creates 174 technical indicators and features
3. **Model Training**:
   - Trains mean regression model
   - Trains quantile models (10th, 50th, 90th percentiles)
   - Trains direction classification model
   - Uses early stopping to prevent overfitting
4. **Validation**: Computes metrics on validation set
5. **Registration**: Saves model and registers in model registry
6. **GPU Detection**: Automatically uses GPU if available

---

## 🎨 GUI Integration

The CatBoost model is now available in the web interface:

1. Navigate to **Model Training** section
2. Select currency pair
3. Choose **"catboost"** from model type dropdown
4. Configure horizons and hyperparameters (or use defaults)
5. Click **Train Model**
6. Monitor progress via job status endpoint
7. Model will be automatically used for predictions once trained

The UI will show:
- Training progress
- Model metrics (RMSE, MAE, R², directional accuracy)
- Which model is being used for predictions
- Model performance comparison

---

## 📈 Performance Comparison

From benchmarking on EURUSD data (1971-2025, ~14,000 daily records):

| Rank | Model | Type | Test RMSE | Test R² | Test MAE |
|------|-------|------|-----------|---------|----------|
| 🏆 1 | **CatBoost_3** | Traditional ML | 0.005640 | 0.988341 | 0.004239 |
| 🥈 2 | CatBoost_Ensemble | Traditional ML | 0.005662 | 0.988253 | 0.004265 |
| 🥉 3 | Ultimate_Ensemble | Traditional ML | 0.005667 | 0.988230 | 0.004251 |
| 4 | CatBoost_1 | Traditional ML | 0.005803 | 0.987659 | 0.004379 |
| 5 | LightGBM | Traditional ML | 0.005882 | 0.987318 | 0.004466 |
| 6 | XGBoost | Traditional ML | 0.006603 | 0.984022 | 0.004914 |
| 7 | LSTM-Transformer | Neural Network | 0.010962 | 0.956220 | 0.008399 |

---

## 🔧 Configuration Options

### Default Configuration (Optimal for Forex):
```python
{
    "iterations": 4000,
    "learning_rate": 0.005,
    "depth": 6,
    "l2_leaf_reg": 10,
    "min_data_in_leaf": 30,
    "random_strength": 0.1,
    "bagging_temperature": 0.1,
    "border_count": 254,
    "task_type": "auto",  # GPU or CPU
    "early_stopping_rounds": 300,
    "random_seed": 42
}
```

### Customization:
All parameters can be overridden when training:
- More iterations for more complex patterns
- Higher learning rate for faster training (less stable)
- Greater depth for capturing complex interactions
- Adjust regularization for your specific use case

---

## 🧪 Testing

To test the CatBoost integration:

```bash
# 1. Start the backend
cd /home/spandan/projects/spec_project_2/currency_assistant
source .venv/bin/activate
python -m backend.main

# 2. Train a CatBoost model (in another terminal)
curl -X POST "http://localhost:8000/api/models/train" \
  -H "Content-Type: application/json" \
  -d '{"currency_pair": "USD/EUR", "model_type": "catboost", "horizons": [1, 3, 7]}'

# 3. Check training status
curl "http://localhost:8000/api/models/train/status/{job_id}"

# 4. Make predictions (model will be used automatically)
curl "http://localhost:8000/api/predict?currency_pair=USD/EUR&horizons=1,3,7"
```

---

## 💡 Why CatBoost Won

1. **Conservative Hyperparameters**: Prevents overfitting on financial data
2. **RobustScaler**: Better handles outliers in forex prices
3. **Proper Regularization**: L2 regularization with high values
4. **Small Learning Rate**: Allows gradual, stable learning
5. **Many Iterations**: Sufficient training with early stopping
6. **GPU Support**: Faster training on large datasets
7. **Gradient Boosting**: Excellent for tabular time series data

---

## 📝 Notes

- CatBoost models are saved in the model registry
- Models are automatically loaded when making predictions
- The system prioritizes CatBoost over other models (best performance)
- GPU acceleration is automatic if CUDA is available
- Training takes longer than LightGBM but produces superior results
- Feature importance is available for model interpretability

---

## 🎓 Next Steps

1. Train CatBoost models for your most important currency pairs
2. Monitor model performance over time
3. Retrain periodically with new data
4. Compare CatBoost vs LightGBM for your specific use cases
5. Experiment with hyperparameters for different market conditions

---

## 📞 Support

For issues or questions:
- Check logs in `logs/` directory
- Review model metrics in model registry
- Use feature importance for debugging
- GPU issues: Check CUDA installation and task_type parameter

**CatBoost is now your best performing model for forex prediction! 🚀**
