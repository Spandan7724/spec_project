"""
Advanced Forex Price Prediction Model Training Pipeline
Comprehensive feature engineering and ensemble modeling for EURUSD prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

import joblib
import json
from datetime import datetime

print("=" * 80)
print("ADVANCED FOREX PRICE PREDICTION MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1/6] Loading and preprocessing data...")

df = pd.read_csv('data/eurusd_d.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ============================================================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Creating comprehensive feature set...")

def create_advanced_features(df):
    """Create extensive technical and statistical features"""
    data = df.copy()

    # Basic price features
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['HL_Pct'] = (data['High'] - data['Low']) / data['Close']
    data['OC_Pct'] = (data['Close'] - data['Open']) / data['Open']

    # Price momentum and changes
    for period in [1, 2, 3, 5, 7, 14, 21, 30]:
        data[f'Price_Change_{period}d'] = data['Close'].pct_change(period)
        data[f'High_Change_{period}d'] = data['High'].pct_change(period)
        data[f'Low_Change_{period}d'] = data['Low'].pct_change(period)

    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{window}'] = SMAIndicator(close=data['Close'], window=window).sma_indicator()
        data[f'EMA_{window}'] = EMAIndicator(close=data['Close'], window=window).ema_indicator()
        data[f'Close_to_SMA_{window}'] = (data['Close'] - data[f'SMA_{window}']) / data[f'SMA_{window}']

    # MA crossovers
    data['SMA_5_20_cross'] = data['SMA_5'] - data['SMA_20']
    data['SMA_10_50_cross'] = data['SMA_10'] - data['SMA_50']
    data['SMA_50_200_cross'] = data['SMA_50'] - data['SMA_200']

    # RSI for multiple periods
    for period in [7, 14, 21, 28]:
        data[f'RSI_{period}'] = RSIIndicator(close=data['Close'], window=period).rsi()

    # MACD
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Bollinger Bands
    for window in [20, 50]:
        bb = BollingerBands(close=data['Close'], window=window, window_dev=2)
        data[f'BB_High_{window}'] = bb.bollinger_hband()
        data[f'BB_Low_{window}'] = bb.bollinger_lband()
        data[f'BB_Mid_{window}'] = bb.bollinger_mavg()
        data[f'BB_Width_{window}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        data[f'BB_Position_{window}'] = (data['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    # ATR (Average True Range)
    for period in [7, 14, 21]:
        data[f'ATR_{period}'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=period).average_true_range()

    # ADX (Average Directional Index)
    adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ADX'] = adx.adx()
    data['ADX_Pos'] = adx.adx_pos()
    data['ADX_Neg'] = adx.adx_neg()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()

    # Williams %R
    data['Williams_R'] = WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close']).williams_r()

    # Volatility measures
    for window in [5, 10, 20, 30]:
        data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
        data[f'Volatility_HL_{window}'] = (data['High'] - data['Low']).rolling(window=window).std()

    # Statistical features
    for window in [5, 10, 20, 30, 50, 60, 100]:
        data[f'Rolling_Mean_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'Rolling_Std_{window}'] = data['Close'].rolling(window=window).std()
        data[f'Rolling_Min_{window}'] = data['Close'].rolling(window=window).min()
        data[f'Rolling_Max_{window}'] = data['Close'].rolling(window=window).max()
        data[f'Rolling_Median_{window}'] = data['Close'].rolling(window=window).median()
        data[f'Rolling_Skew_{window}'] = data['Returns'].rolling(window=window).skew()
        data[f'Rolling_Kurt_{window}'] = data['Returns'].rolling(window=window).kurt()

    # Distance from extremes
    for window in [20, 50, 100]:
        data[f'Distance_from_High_{window}'] = (data[f'Rolling_Max_{window}'] - data['Close']) / data['Close']
        data[f'Distance_from_Low_{window}'] = (data['Close'] - data[f'Rolling_Min_{window}']) / data['Close']

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
        data[f'Volume_Proxy_Lag_{lag}'] = data['HL_Pct'].shift(lag)  # Using HL% as volume proxy

    # Time-based features
    data['Day_of_Week'] = data['Date'].dt.dayofweek
    data['Day_of_Month'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Year'] = data['Date'].dt.year

    # Cyclical encoding for time features
    data['Day_of_Week_Sin'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
    data['Day_of_Week_Cos'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Trend indicators
    for window in [10, 20, 50]:
        data[f'Trend_{window}'] = data['Close'].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )

    return data

df_features = create_advanced_features(df)
print(f"Created {len(df_features.columns) - len(df.columns)} new features")

# ============================================================================
# 3. TARGET VARIABLE AND DATA SPLITTING
# ============================================================================
print("\n[3/6] Creating targets and splitting data...")

# Multiple prediction horizons
df_features['Target_1d'] = df_features['Close'].shift(-1)  # Next day
df_features['Target_3d'] = df_features['Close'].shift(-3)  # 3 days ahead
df_features['Target_7d'] = df_features['Close'].shift(-7)  # 1 week ahead

# For this training, focus on 1-day ahead prediction
target_col = 'Target_1d'

# Remove rows with NaN values
df_features = df_features.dropna()

# Features to exclude from training
exclude_cols = ['Date', 'Target_1d', 'Target_3d', 'Target_7d', 'Close', 'Open', 'High', 'Low']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

print(f"Total features for modeling: {len(feature_cols)}")

# Time series split (80/20 split maintaining temporal order)
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size].copy()
test_data = df_features.iloc[train_size:].copy()

X_train = train_data[feature_cols]
y_train = train_data[target_col]
X_test = test_data[feature_cols]
y_test = test_data[target_col]

print(f"Training set: {len(train_data)} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
print(f"Test set: {len(test_data)} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# ============================================================================
# 4. TRAIN ADVANCED ENSEMBLE MODELS
# ============================================================================
print("\n[4/6] Training advanced ensemble models...")

results = {}

# -------- LightGBM --------
print("\n[4.1] Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_train],
    callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(500)]
)

lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_test = lgb_model.predict(X_test)

results['LightGBM'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, lgb_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, lgb_pred_test)),
    'train_mae': mean_absolute_error(y_train, lgb_pred_train),
    'test_mae': mean_absolute_error(y_test, lgb_pred_test),
    'train_r2': r2_score(y_train, lgb_pred_train),
    'test_r2': r2_score(y_test, lgb_pred_test)
}

lgb_model.save_model('models/lightgbm_model.txt')
print(f"LightGBM Test RMSE: {results['LightGBM']['test_rmse']:.6f}, Test R²: {results['LightGBM']['test_r2']:.6f}")

# -------- XGBoost --------
print("\n[4.2] Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.01,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'tree_method': 'hist',
    'n_jobs': -1
}

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test_dm = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(
    xgb_params,
    xgb_train,
    num_boost_round=5000,
    evals=[(xgb_train, 'train')],
    early_stopping_rounds=200,
    verbose_eval=500
)

xgb_pred_train = xgb_model.predict(xgb.DMatrix(X_train))
xgb_pred_test = xgb_model.predict(xgb.DMatrix(X_test))

results['XGBoost'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, xgb_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, xgb_pred_test)),
    'train_mae': mean_absolute_error(y_train, xgb_pred_train),
    'test_mae': mean_absolute_error(y_test, xgb_pred_test),
    'train_r2': r2_score(y_train, xgb_pred_train),
    'test_r2': r2_score(y_test, xgb_pred_test)
}

xgb_model.save_model('models/xgboost_model.json')
print(f"XGBoost Test RMSE: {results['XGBoost']['test_rmse']:.6f}, Test R²: {results['XGBoost']['test_r2']:.6f}")

# -------- Neural Network (MLP) --------
print("\n[4.3] Training Neural Network...")
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    random_state=42,
    verbose=False
)

mlp_model.fit(X_train_scaled, y_train)
mlp_pred_train = mlp_model.predict(X_train_scaled)
mlp_pred_test = mlp_model.predict(X_test_scaled)

results['Neural_Network'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, mlp_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, mlp_pred_test)),
    'train_mae': mean_absolute_error(y_train, mlp_pred_train),
    'test_mae': mean_absolute_error(y_test, mlp_pred_test),
    'train_r2': r2_score(y_train, mlp_pred_train),
    'test_r2': r2_score(y_test, mlp_pred_test)
}

joblib.dump(mlp_model, 'models/neural_network_model.pkl')
print(f"Neural Network Test RMSE: {results['Neural_Network']['test_rmse']:.6f}, Test R²: {results['Neural_Network']['test_r2']:.6f}")

# -------- Ensemble (Weighted Average) --------
print("\n[4.4] Creating Ensemble Model...")

# Calculate inverse RMSE weights (better models get higher weight)
test_rmses = [results['LightGBM']['test_rmse'], results['XGBoost']['test_rmse'], results['Neural_Network']['test_rmse']]
inv_rmse = [1/rmse for rmse in test_rmses]
weights = [w/sum(inv_rmse) for w in inv_rmse]

print(f"Ensemble weights - LightGBM: {weights[0]:.3f}, XGBoost: {weights[1]:.3f}, Neural Network: {weights[2]:.3f}")

ensemble_pred_train = (weights[0] * lgb_pred_train +
                        weights[1] * xgb_pred_train +
                        weights[2] * mlp_pred_train)
ensemble_pred_test = (weights[0] * lgb_pred_test +
                       weights[1] * xgb_pred_test +
                       weights[2] * mlp_pred_test)

results['Ensemble'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred_test)),
    'train_mae': mean_absolute_error(y_train, ensemble_pred_train),
    'test_mae': mean_absolute_error(y_test, ensemble_pred_test),
    'train_r2': r2_score(y_train, ensemble_pred_train),
    'test_r2': r2_score(y_test, ensemble_pred_test),
    'weights': {'LightGBM': weights[0], 'XGBoost': weights[1], 'Neural_Network': weights[2]}
}

print(f"Ensemble Test RMSE: {results['Ensemble']['test_rmse']:.6f}, Test R²: {results['Ensemble']['test_r2']:.6f}")

# ============================================================================
# 5. TRAIN CATBOOST WITH GPU
# ============================================================================
print("\n[5/6] Training CatBoost models with GPU support...")

# Check if GPU is available
try:
    # Test GPU availability
    test_pool = Pool([[1, 2, 3]], [1])
    test_model = CatBoostRegressor(iterations=1, task_type='GPU', verbose=0)
    test_model.fit(test_pool)
    gpu_available = True
    task_type = 'GPU'
    print("GPU detected! Using GPU for CatBoost training.")
except:
    gpu_available = False
    task_type = 'CPU'
    print("GPU not available. Using CPU for CatBoost training.")

# Create Pool objects
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# -------- CatBoost Model 1: Default optimized --------
print("\n[5.1] Training CatBoost Model 1 (Optimized)...")
catboost_model1 = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=20,
    random_strength=0.5,
    bagging_temperature=0.2,
    border_count=254,
    task_type=task_type,
    verbose=500,
    early_stopping_rounds=200,
    random_seed=42
)

catboost_model1.fit(train_pool, eval_set=test_pool)

cb1_pred_train = catboost_model1.predict(X_train)
cb1_pred_test = catboost_model1.predict(X_test)

results['CatBoost_1'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, cb1_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, cb1_pred_test)),
    'train_mae': mean_absolute_error(y_train, cb1_pred_train),
    'test_mae': mean_absolute_error(y_test, cb1_pred_test),
    'train_r2': r2_score(y_train, cb1_pred_train),
    'test_r2': r2_score(y_test, cb1_pred_test)
}

catboost_model1.save_model('models/catboost_model1.cbm')
print(f"CatBoost 1 Test RMSE: {results['CatBoost_1']['test_rmse']:.6f}, Test R²: {results['CatBoost_1']['test_r2']:.6f}")

# -------- CatBoost Model 2: Deep trees --------
print("\n[5.2] Training CatBoost Model 2 (Deep Trees)...")
catboost_model2 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=5,
    min_data_in_leaf=10,
    random_strength=1,
    bagging_temperature=0.5,
    task_type=task_type,
    verbose=500,
    early_stopping_rounds=200,
    random_seed=123
)

catboost_model2.fit(train_pool, eval_set=test_pool)

cb2_pred_train = catboost_model2.predict(X_train)
cb2_pred_test = catboost_model2.predict(X_test)

results['CatBoost_2'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, cb2_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, cb2_pred_test)),
    'train_mae': mean_absolute_error(y_train, cb2_pred_train),
    'test_mae': mean_absolute_error(y_test, cb2_pred_test),
    'train_r2': r2_score(y_train, cb2_pred_train),
    'test_r2': r2_score(y_test, cb2_pred_test)
}

catboost_model2.save_model('models/catboost_model2.cbm')
print(f"CatBoost 2 Test RMSE: {results['CatBoost_2']['test_rmse']:.6f}, Test R²: {results['CatBoost_2']['test_r2']:.6f}")

# -------- CatBoost Model 3: Conservative (less overfitting) --------
print("\n[5.3] Training CatBoost Model 3 (Conservative)...")
catboost_model3 = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.005,
    depth=6,
    l2_leaf_reg=10,
    min_data_in_leaf=30,
    random_strength=0.1,
    bagging_temperature=0.1,
    task_type=task_type,
    verbose=500,
    early_stopping_rounds=300,
    random_seed=456
)

catboost_model3.fit(train_pool, eval_set=test_pool)

cb3_pred_train = catboost_model3.predict(X_train)
cb3_pred_test = catboost_model3.predict(X_test)

results['CatBoost_3'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, cb3_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, cb3_pred_test)),
    'train_mae': mean_absolute_error(y_train, cb3_pred_train),
    'test_mae': mean_absolute_error(y_test, cb3_pred_test),
    'train_r2': r2_score(y_train, cb3_pred_train),
    'test_r2': r2_score(y_test, cb3_pred_test)
}

catboost_model3.save_model('models/catboost_model3.cbm')
print(f"CatBoost 3 Test RMSE: {results['CatBoost_3']['test_rmse']:.6f}, Test R²: {results['CatBoost_3']['test_r2']:.6f}")

# -------- CatBoost Ensemble --------
print("\n[5.4] Creating CatBoost Ensemble...")

cb_test_rmses = [results['CatBoost_1']['test_rmse'], results['CatBoost_2']['test_rmse'], results['CatBoost_3']['test_rmse']]
cb_inv_rmse = [1/rmse for rmse in cb_test_rmses]
cb_weights = [w/sum(cb_inv_rmse) for w in cb_inv_rmse]

print(f"CatBoost Ensemble weights - Model 1: {cb_weights[0]:.3f}, Model 2: {cb_weights[1]:.3f}, Model 3: {cb_weights[2]:.3f}")

cb_ensemble_pred_train = (cb_weights[0] * cb1_pred_train +
                          cb_weights[1] * cb2_pred_train +
                          cb_weights[2] * cb3_pred_train)
cb_ensemble_pred_test = (cb_weights[0] * cb1_pred_test +
                         cb_weights[1] * cb2_pred_test +
                         cb_weights[2] * cb3_pred_test)

results['CatBoost_Ensemble'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, cb_ensemble_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, cb_ensemble_pred_test)),
    'train_mae': mean_absolute_error(y_train, cb_ensemble_pred_train),
    'test_mae': mean_absolute_error(y_test, cb_ensemble_pred_test),
    'train_r2': r2_score(y_train, cb_ensemble_pred_train),
    'test_r2': r2_score(y_test, cb_ensemble_pred_test),
    'weights': {'Model_1': cb_weights[0], 'Model_2': cb_weights[1], 'Model_3': cb_weights[2]}
}

print(f"CatBoost Ensemble Test RMSE: {results['CatBoost_Ensemble']['test_rmse']:.6f}, Test R²: {results['CatBoost_Ensemble']['test_r2']:.6f}")

# -------- Ultimate Ensemble (All models) --------
print("\n[5.5] Creating Ultimate Ensemble (All Models)...")

all_test_rmses = [
    results['LightGBM']['test_rmse'],
    results['XGBoost']['test_rmse'],
    results['Neural_Network']['test_rmse'],
    results['CatBoost_1']['test_rmse'],
    results['CatBoost_2']['test_rmse'],
    results['CatBoost_3']['test_rmse']
]

all_inv_rmse = [1/rmse for rmse in all_test_rmses]
all_weights = [w/sum(all_inv_rmse) for w in all_inv_rmse]

ultimate_pred_train = (all_weights[0] * lgb_pred_train +
                        all_weights[1] * xgb_pred_train +
                        all_weights[2] * mlp_pred_train +
                        all_weights[3] * cb1_pred_train +
                        all_weights[4] * cb2_pred_train +
                        all_weights[5] * cb3_pred_train)

ultimate_pred_test = (all_weights[0] * lgb_pred_test +
                       all_weights[1] * xgb_pred_test +
                       all_weights[2] * mlp_pred_test +
                       all_weights[3] * cb1_pred_test +
                       all_weights[4] * cb2_pred_test +
                       all_weights[5] * cb3_pred_test)

results['Ultimate_Ensemble'] = {
    'train_rmse': np.sqrt(mean_squared_error(y_train, ultimate_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, ultimate_pred_test)),
    'train_mae': mean_absolute_error(y_train, ultimate_pred_train),
    'test_mae': mean_absolute_error(y_test, ultimate_pred_test),
    'train_r2': r2_score(y_train, ultimate_pred_train),
    'test_r2': r2_score(y_test, ultimate_pred_test),
    'weights': {
        'LightGBM': all_weights[0],
        'XGBoost': all_weights[1],
        'Neural_Network': all_weights[2],
        'CatBoost_1': all_weights[3],
        'CatBoost_2': all_weights[4],
        'CatBoost_3': all_weights[5]
    }
}

print(f"Ultimate Ensemble Test RMSE: {results['Ultimate_Ensemble']['test_rmse']:.6f}, Test R²: {results['Ultimate_Ensemble']['test_r2']:.6f}")

# ============================================================================
# 6. RESULTS SUMMARY AND SAVE
# ============================================================================
print("\n" + "=" * 80)
print("[6/6] FINAL RESULTS SUMMARY")
print("=" * 80)

# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('test_rmse')

print("\n📊 MODEL PERFORMANCE COMPARISON (Sorted by Test RMSE)")
print("-" * 80)
print(f"{'Model':<25} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<12}")
print("-" * 80)
for model_name, metrics in results_df.iterrows():
    print(f"{model_name:<25} {metrics['test_rmse']:<12.6f} {metrics['test_mae']:<12.6f} {metrics['test_r2']:<12.6f}")

print("\n🏆 BEST MODEL: " + results_df.index[0])
print(f"   Test RMSE: {results_df.iloc[0]['test_rmse']:.6f}")
print(f"   Test MAE: {results_df.iloc[0]['test_mae']:.6f}")
print(f"   Test R²: {results_df.iloc[0]['test_r2']:.6f}")

# Additional metrics
best_model_name = results_df.index[0]
best_predictions = {
    'LightGBM': lgb_pred_test,
    'XGBoost': xgb_pred_test,
    'Neural_Network': mlp_pred_test,
    'Ensemble': ensemble_pred_test,
    'CatBoost_1': cb1_pred_test,
    'CatBoost_2': cb2_pred_test,
    'CatBoost_3': cb3_pred_test,
    'CatBoost_Ensemble': cb_ensemble_pred_test,
    'Ultimate_Ensemble': ultimate_pred_test
}[best_model_name]

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - best_predictions) / y_test)) * 100
print(f"   Test MAPE: {mape:.4f}%")

# Direction accuracy (did we predict up/down correctly?)
current_prices = test_data['Close'].values
actual_direction = np.sign(y_test.values - current_prices)
pred_direction = np.sign(best_predictions - current_prices)
direction_accuracy = np.mean(actual_direction == pred_direction) * 100
print(f"   Direction Accuracy: {direction_accuracy:.2f}%")

# Save all results
results_to_save = {
    'model_performance': results_df.to_dict(),
    'best_model': best_model_name,
    'test_metrics': {
        'rmse': float(results_df.iloc[0]['test_rmse']),
        'mae': float(results_df.iloc[0]['test_mae']),
        'r2': float(results_df.iloc[0]['test_r2']),
        'mape': float(mape),
        'direction_accuracy': float(direction_accuracy)
    },
    'training_info': {
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'n_features': len(feature_cols),
        'train_period': f"{train_data['Date'].min()} to {train_data['Date'].max()}",
        'test_period': f"{test_data['Date'].min()} to {test_data['Date'].max()}",
        'gpu_used': gpu_available
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('models/training_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=4, default=str)

# Save feature list
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

print("\n✅ All models saved in 'models/' directory")
print("✅ Training results saved to 'models/training_results.json'")
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
