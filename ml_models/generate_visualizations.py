"""
Comprehensive Visualization Script for Forex Price Prediction Models
Generates all relevant plots and graphs for model analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================
print("\n[1/8] Loading data and models...")

# Load original data
df = pd.read_csv('data/eurusd_d.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Load training results
with open('models/training_results.json', 'r') as f:
    results = json.load(f)

with open('models/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

# Recreate features
from feature_engineering import create_advanced_features

df_features = create_advanced_features(df)
df_features['Target_1d'] = df_features['Close'].shift(-1)
df_features = df_features.dropna()

# Split data
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size].copy()
test_data = df_features.iloc[train_size:].copy()

exclude_cols = ['Date', 'Target_1d', 'Target_3d', 'Target_7d', 'Close', 'Open', 'High', 'Low']
X_train = train_data[feature_cols]
y_train = train_data['Target_1d']
X_test = test_data[feature_cols]
y_test = test_data['Target_1d']

# Load models
scaler = joblib.load('models/scaler.pkl')
X_test_scaled = scaler.transform(X_test)

lgb_model = lgb.Booster(model_file='models/lightgbm_model.txt')
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_model.json')
mlp_model = joblib.load('models/neural_network_model.pkl')
catboost_model1 = CatBoostRegressor()
catboost_model1.load_model('models/catboost_model1.cbm')
catboost_model2 = CatBoostRegressor()
catboost_model2.load_model('models/catboost_model2.cbm')
catboost_model3 = CatBoostRegressor()
catboost_model3.load_model('models/catboost_model3.cbm')

# Generate predictions
lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
mlp_pred = mlp_model.predict(X_test_scaled)
cb1_pred = catboost_model1.predict(X_test)
cb2_pred = catboost_model2.predict(X_test)
cb3_pred = catboost_model3.predict(X_test)

# Calculate ensembles
test_rmses = [
    np.sqrt(mean_squared_error(y_test, lgb_pred)),
    np.sqrt(mean_squared_error(y_test, xgb_pred)),
    np.sqrt(mean_squared_error(y_test, mlp_pred))
]
inv_rmse = [1/rmse for rmse in test_rmses]
weights = [w/sum(inv_rmse) for w in inv_rmse]
ensemble_pred = weights[0] * lgb_pred + weights[1] * xgb_pred + weights[2] * mlp_pred

cb_test_rmses = [
    np.sqrt(mean_squared_error(y_test, cb1_pred)),
    np.sqrt(mean_squared_error(y_test, cb2_pred)),
    np.sqrt(mean_squared_error(y_test, cb3_pred))
]
cb_inv_rmse = [1/rmse for rmse in cb_test_rmses]
cb_weights = [w/sum(cb_inv_rmse) for w in cb_inv_rmse]
cb_ensemble_pred = cb_weights[0] * cb1_pred + cb_weights[1] * cb2_pred + cb_weights[2] * cb3_pred

all_test_rmses = test_rmses + cb_test_rmses
all_inv_rmse = [1/rmse for rmse in all_test_rmses]
all_weights = [w/sum(all_inv_rmse) for w in all_inv_rmse]
ultimate_pred = (all_weights[0] * lgb_pred + all_weights[1] * xgb_pred +
                 all_weights[2] * mlp_pred + all_weights[3] * cb1_pred +
                 all_weights[4] * cb2_pred + all_weights[5] * cb3_pred)

predictions_dict = {
    'LightGBM': lgb_pred,
    'XGBoost': xgb_pred,
    'Neural_Network': mlp_pred,
    'Ensemble': ensemble_pred,
    'CatBoost_1': cb1_pred,
    'CatBoost_2': cb2_pred,
    'CatBoost_3': cb3_pred,
    'CatBoost_Ensemble': cb_ensemble_pred,
    'Ultimate_Ensemble': ultimate_pred
}

print(f"Loaded {len(predictions_dict)} models successfully")

# ============================================================================
# 2. MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n[2/8] Creating model performance comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Extract metrics
model_names = list(results['model_performance']['test_rmse'].keys())
test_rmse = [results['model_performance']['test_rmse'][m] for m in model_names]
test_mae = [results['model_performance']['test_mae'][m] for m in model_names]
test_r2 = [results['model_performance']['test_r2'][m] for m in model_names]
train_rmse = [results['model_performance']['train_rmse'][m] for m in model_names]

# Sort by test RMSE
sorted_indices = np.argsort(test_rmse)
model_names_sorted = [model_names[i] for i in sorted_indices]
test_rmse_sorted = [test_rmse[i] for i in sorted_indices]
test_mae_sorted = [test_mae[i] for i in sorted_indices]
test_r2_sorted = [test_r2[i] for i in sorted_indices]
train_rmse_sorted = [train_rmse[i] for i in sorted_indices]

# Plot 1: Test RMSE comparison
ax1 = axes[0, 0]
bars = ax1.barh(model_names_sorted, test_rmse_sorted, color=sns.color_palette("viridis", len(model_names)))
ax1.set_xlabel('Test RMSE', fontweight='bold')
ax1.set_title('Test RMSE by Model (Lower is Better)', fontweight='bold')
ax1.axvline(x=min(test_rmse_sorted), color='red', linestyle='--', alpha=0.5, label='Best')
for i, (bar, val) in enumerate(zip(bars, test_rmse_sorted)):
    ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
             va='center', ha='left', fontsize=8, fontweight='bold')
ax1.legend()

# Plot 2: Test R² comparison
ax2 = axes[0, 1]
bars = ax2.barh(model_names_sorted, test_r2_sorted, color=sns.color_palette("rocket", len(model_names)))
ax2.set_xlabel('Test R² Score', fontweight='bold')
ax2.set_title('Test R² Score by Model (Higher is Better)', fontweight='bold')
ax2.axvline(x=max(test_r2_sorted), color='red', linestyle='--', alpha=0.5, label='Best')
for i, (bar, val) in enumerate(zip(bars, test_r2_sorted)):
    ax2.text(val, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
             va='center', ha='left', fontsize=8, fontweight='bold')
ax2.legend()

# Plot 3: Train vs Test RMSE (Overfitting check)
ax3 = axes[1, 0]
x_pos = np.arange(len(model_names_sorted))
width = 0.35
bars1 = ax3.bar(x_pos - width/2, train_rmse_sorted, width, label='Train RMSE', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, test_rmse_sorted, width, label='Test RMSE', alpha=0.8)
ax3.set_xlabel('Models', fontweight='bold')
ax3.set_ylabel('RMSE', fontweight='bold')
ax3.set_title('Train vs Test RMSE (Overfitting Check)', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names_sorted, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Test MAE comparison
ax4 = axes[1, 1]
bars = ax4.barh(model_names_sorted, test_mae_sorted, color=sns.color_palette("mako", len(model_names)))
ax4.set_xlabel('Test MAE', fontweight='bold')
ax4.set_title('Test MAE by Model (Lower is Better)', fontweight='bold')
ax4.axvline(x=min(test_mae_sorted), color='red', linestyle='--', alpha=0.5, label='Best')
for i, (bar, val) in enumerate(zip(bars, test_mae_sorted)):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
             va='center', ha='left', fontsize=8, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig('visualizations/01_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/01_model_performance_comparison.png")
plt.close()

# ============================================================================
# 3. ACTUAL VS PREDICTED PRICES
# ============================================================================
print("\n[3/8] Creating actual vs predicted price plots...")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Actual vs Predicted Prices (Test Set)', fontsize=16, fontweight='bold')

for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
    ax = axes[idx // 3, idx % 3]

    # Plot actual vs predicted
    ax.scatter(y_test, predictions, alpha=0.4, s=10)

    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    ax.set_xlabel('Actual Price', fontweight='bold')
    ax.set_ylabel('Predicted Price', fontweight='bold')
    ax.set_title(f'{model_name}\nRMSE: {rmse:.6f}, R²: {r2:.6f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/02_actual_vs_predicted.png")
plt.close()

# ============================================================================
# 4. TIME SERIES PREDICTIONS
# ============================================================================
print("\n[4/8] Creating time series prediction plots...")

# Best 3 models
best_models = ['CatBoost_3', 'CatBoost_Ensemble', 'Ultimate_Ensemble']

fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle('Time Series: Actual vs Predicted Prices (Test Period)', fontsize=16, fontweight='bold')

for idx, model_name in enumerate(best_models):
    ax = axes[idx]

    # Plot full test period
    ax.plot(test_data['Date'].values, y_test.values, label='Actual', linewidth=2, alpha=0.7)
    ax.plot(test_data['Date'].values, predictions_dict[model_name],
            label='Predicted', linewidth=2, alpha=0.7)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions_dict[model_name]))
    mae = mean_absolute_error(y_test, predictions_dict[model_name])
    r2 = r2_score(y_test, predictions_dict[model_name])
    mape = np.mean(np.abs((y_test.values - predictions_dict[model_name]) / y_test.values)) * 100

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price (EUR/USD)', fontweight='bold')
    ax.set_title(f'{model_name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}, MAPE: {mape:.4f}%',
                 fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('visualizations/03_time_series_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/03_time_series_predictions.png")
plt.close()

# ============================================================================
# 5. PREDICTION ERRORS
# ============================================================================
print("\n[5/8] Creating prediction error analysis plots...")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Prediction Errors Distribution', fontsize=16, fontweight='bold')

for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
    ax = axes[idx // 3, idx % 3]

    # Calculate errors
    errors = y_test.values - predictions

    # Plot histogram
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')

    # Add vertical line at 0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')

    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    ax.set_xlabel('Prediction Error', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{model_name}\nMean: {mean_error:.6f}, Std: {std_error:.6f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/04_prediction_errors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/04_prediction_errors.png")
plt.close()

# ============================================================================
# 6. RESIDUAL PLOTS
# ============================================================================
print("\n[6/8] Creating residual plots...")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Residual Plots (Error vs Predicted Value)', fontsize=16, fontweight='bold')

for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
    ax = axes[idx // 3, idx % 3]

    # Calculate residuals
    residuals = y_test.values - predictions

    # Plot residuals
    ax.scatter(predictions, residuals, alpha=0.4, s=10)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

    # Add confidence bands
    std_resid = np.std(residuals)
    ax.axhline(y=2*std_resid, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±2σ')
    ax.axhline(y=-2*std_resid, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Predicted Price', fontweight='bold')
    ax.set_ylabel('Residuals', fontweight='bold')
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/05_residual_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/05_residual_plots.png")
plt.close()

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n[7/8] Creating feature importance plots...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Feature Importance Analysis (Top 30 Features)', fontsize=16, fontweight='bold')

# LightGBM
ax1 = axes[0, 0]
lgb_importance = lgb_model.feature_importance(importance_type='gain')
lgb_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_importance
}).sort_values('importance', ascending=False).head(30)
ax1.barh(range(len(lgb_features)), lgb_features['importance'].values)
ax1.set_yticks(range(len(lgb_features)))
ax1.set_yticklabels(lgb_features['feature'].values, fontsize=8)
ax1.set_xlabel('Importance (Gain)', fontweight='bold')
ax1.set_title('LightGBM Feature Importance', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# XGBoost
ax2 = axes[0, 1]
xgb_importance = xgb_model.get_score(importance_type='gain')
xgb_features = pd.DataFrame({
    'feature': list(xgb_importance.keys()),
    'importance': list(xgb_importance.values())
}).sort_values('importance', ascending=False).head(30)
# Map feature names - handle both f0 format and actual names
xgb_features['feature'] = xgb_features['feature'].apply(
    lambda x: feature_cols[int(x.replace('f', ''))] if x.startswith('f') and x[1:].isdigit() else x
)
ax2.barh(range(len(xgb_features)), xgb_features['importance'].values)
ax2.set_yticks(range(len(xgb_features)))
ax2.set_yticklabels(xgb_features['feature'].values, fontsize=8)
ax2.set_xlabel('Importance (Gain)', fontweight='bold')
ax2.set_title('XGBoost Feature Importance', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# CatBoost 1
ax3 = axes[1, 0]
cb1_importance = catboost_model1.get_feature_importance()
cb1_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': cb1_importance
}).sort_values('importance', ascending=False).head(30)
ax3.barh(range(len(cb1_features)), cb1_features['importance'].values)
ax3.set_yticks(range(len(cb1_features)))
ax3.set_yticklabels(cb1_features['feature'].values, fontsize=8)
ax3.set_xlabel('Importance', fontweight='bold')
ax3.set_title('CatBoost 1 Feature Importance', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# CatBoost 3 (Best Model)
ax4 = axes[1, 1]
cb3_importance = catboost_model3.get_feature_importance()
cb3_features = pd.DataFrame({
    'feature': feature_cols,
    'importance': cb3_importance
}).sort_values('importance', ascending=False).head(30)
ax4.barh(range(len(cb3_features)), cb3_features['importance'].values)
ax4.set_yticks(range(len(cb3_features)))
ax4.set_yticklabels(cb3_features['feature'].values, fontsize=8)
ax4.set_xlabel('Importance', fontweight='bold')
ax4.set_title('CatBoost 3 (Best Model) Feature Importance', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/06_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/06_feature_importance.png")
plt.close()

# ============================================================================
# 8. RECENT PREDICTIONS (LAST 6 MONTHS)
# ============================================================================
print("\n[8/8] Creating recent predictions zoom-in plots...")

# Last 6 months of test data (approximately 180 days * 5/7 for trading days)
last_n = min(120, len(test_data))

fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle(f'Recent Predictions - Last {last_n} Trading Days', fontsize=16, fontweight='bold')

for idx, model_name in enumerate(best_models):
    ax = axes[idx]

    dates_recent = test_data['Date'].values[-last_n:]
    actual_recent = y_test.values[-last_n:]
    pred_recent = predictions_dict[model_name][-last_n:]

    # Plot
    ax.plot(dates_recent, actual_recent, label='Actual', linewidth=2.5, alpha=0.8, marker='o', markersize=3)
    ax.plot(dates_recent, pred_recent, label='Predicted', linewidth=2.5, alpha=0.8, marker='s', markersize=3)

    # Fill between
    ax.fill_between(dates_recent, actual_recent, pred_recent, alpha=0.2)

    # Calculate metrics for recent period
    rmse_recent = np.sqrt(mean_squared_error(actual_recent, pred_recent))
    mae_recent = mean_absolute_error(actual_recent, pred_recent)
    r2_recent = r2_score(actual_recent, pred_recent)

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Price (EUR/USD)', fontweight='bold')
    ax.set_title(f'{model_name} - Recent RMSE: {rmse_recent:.6f}, MAE: {mae_recent:.6f}, R²: {r2_recent:.6f}',
                 fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('visualizations/07_recent_predictions_zoom.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/07_recent_predictions_zoom.png")
plt.close()

# ============================================================================
# 9. METRICS SUMMARY TABLE
# ============================================================================
print("\n[9/9] Creating metrics summary table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Model', 'Test RMSE', 'Test MAE', 'Test R²', 'Test MAPE (%)', 'Train RMSE', 'Overfitting'])

for model_name in model_names_sorted:
    predictions = predictions_dict[model_name]
    mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

    # Extract metrics
    test_rmse_val = results['model_performance']['test_rmse'][model_name]
    test_mae_val = results['model_performance']['test_mae'][model_name]
    test_r2_val = results['model_performance']['test_r2'][model_name]
    train_rmse_val = results['model_performance']['train_rmse'][model_name]
    overfitting = ((train_rmse_val - test_rmse_val) / test_rmse_val * 100)

    table_data.append([
        model_name,
        f"{test_rmse_val:.6f}",
        f"{test_mae_val:.6f}",
        f"{test_r2_val:.6f}",
        f"{mape:.4f}",
        f"{train_rmse_val:.6f}",
        f"{overfitting:.2f}%"
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.18, 0.12, 0.12, 0.12, 0.14, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(7):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model (first data row after header)
for i in range(7):
    table[(1, i)].set_facecolor('#FFF9C4')
    table[(1, i)].set_text_props(weight='bold')

# Alternate row colors
for i in range(2, len(table_data)):
    color = '#F5F5F5' if i % 2 == 0 else 'white'
    for j in range(7):
        table[(i, j)].set_facecolor(color)

plt.title('Complete Model Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('visualizations/08_metrics_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/08_metrics_summary_table.png")
plt.close()

# ============================================================================
# 10. PRICE HISTORY WITH TRAIN/TEST SPLIT
# ============================================================================
print("\n[10/10] Creating complete price history with train/test split...")

fig, ax = plt.subplots(figsize=(20, 8))

# Plot training data
ax.plot(train_data['Date'], train_data['Close'], label='Training Data',
        linewidth=1.5, alpha=0.7, color='blue')

# Plot test data
ax.plot(test_data['Date'], test_data['Close'], label='Test Data (Actual)',
        linewidth=1.5, alpha=0.7, color='green')

# Plot best model predictions
ax.plot(test_data['Date'], predictions_dict['CatBoost_3'], label='Best Model Predictions (CatBoost_3)',
        linewidth=1.5, alpha=0.7, color='red', linestyle='--')

# Add vertical line at split point
split_date = test_data['Date'].iloc[0]
ax.axvline(x=split_date, color='black', linestyle='--', linewidth=2,
           label=f'Train/Test Split ({split_date.strftime("%Y-%m-%d")})')

ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('EUR/USD Price', fontweight='bold', fontsize=12)
ax.set_title('Complete EUR/USD Price History with Model Predictions',
             fontweight='bold', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('visualizations/09_complete_price_history.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/09_complete_price_history.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
print("\n📁 All visualizations saved in 'visualizations/' directory:")
print("   1. Model performance comparison")
print("   2. Actual vs predicted scatter plots")
print("   3. Time series predictions")
print("   4. Prediction errors distribution")
print("   5. Residual plots")
print("   6. Feature importance")
print("   7. Recent predictions zoom-in")
print("   8. Metrics summary table")
print("   9. Complete price history")
print("\n" + "=" * 80)
