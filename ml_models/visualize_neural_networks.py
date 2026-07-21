"""
Visualizations for Advanced Neural Network Models
Generates comprehensive plots for model performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("NEURAL NETWORK VISUALIZATION GENERATOR")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load predictions
predictions_df = pd.read_csv('models/nn_predictions.csv')
print(f"✓ Loaded predictions: {len(predictions_df)} samples")

# Load results
with open('models/advanced_nn_results.json', 'r') as f:
    results = json.load(f)
print(f"✓ Loaded results for {len(results)} models")

# ============================================================================
# 2. MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n[2/5] Creating model performance comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network Model Performance Comparison', fontsize=16, fontweight='bold')

# Extract metrics
models = list(results.keys())
rmse_values = [results[m]['test_rmse'] for m in models]
mae_values = [results[m]['test_mae'] for m in models]
r2_values = [results[m]['test_r2'] for m in models]

# Calculate MAPE for each model
mape_values = []
for model in models:
    if model != 'NN_Ensemble':
        pred = predictions_df[model].values
        actual = predictions_df['actual'].values
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        mape_values.append(mape)
    else:
        pred = predictions_df[model].values
        actual = predictions_df['actual'].values
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        mape_values.append(mape)

# 2.1 RMSE Comparison
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = ax.barh(models, rmse_values, color=colors)
ax.set_xlabel('RMSE', fontsize=11, fontweight='bold')
ax.set_title('Test RMSE by Model', fontsize=12, fontweight='bold')
ax.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.6f}',
            va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# 2.2 MAE Comparison
ax = axes[0, 1]
bars = ax.barh(models, mae_values, color=colors)
ax.set_xlabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('Test MAE by Model', fontsize=12, fontweight='bold')
ax.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.6f}',
            va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# 2.3 R² Comparison
ax = axes[1, 0]
bars = ax.barh(models, r2_values, color=colors)
ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Test R² by Model', fontsize=12, fontweight='bold')
ax.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}',
            va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# 2.4 MAPE Comparison
ax = axes[1, 1]
bars = ax.barh(models, mape_values, color=colors)
ax.set_xlabel('MAPE (%)', fontsize=11, fontweight='bold')
ax.set_title('Test MAPE by Model', fontsize=12, fontweight='bold')
ax.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, mape_values)):
    ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}%',
            va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/nn_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_performance_comparison.png")
plt.close()

# ============================================================================
# 3. PREDICTIONS VS ACTUAL
# ============================================================================
print("\n[3/5] Creating predictions vs actual plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Neural Network Predictions vs Actual Prices', fontsize=16, fontweight='bold')

model_names = ['ALFA', 'LSTM_Transformer', 'BiLSTM_MultiHead', 'CNN_LSTM', 'TCN', 'NN_Ensemble']
axes_flat = axes.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes_flat[idx]

    actual = predictions_df['actual'].values
    predicted = predictions_df[model_name].values

    # Create time index
    time_idx = np.arange(len(actual))

    # Plot
    ax.plot(time_idx, actual, label='Actual', alpha=0.7, linewidth=1.5)
    ax.plot(time_idx, predicted, label='Predicted', alpha=0.7, linewidth=1.5)

    # Add metrics
    rmse = results[model_name]['test_rmse']
    r2 = results[model_name]['test_r2']

    ax.set_title(f'{model_name}\nRMSE: {rmse:.6f}, R²: {r2:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/nn_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_predictions_vs_actual.png")
plt.close()

# ============================================================================
# 4. SCATTER PLOTS - PREDICTED VS ACTUAL
# ============================================================================
print("\n[4/5] Creating scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Neural Network Predictions: Scatter Plots', fontsize=16, fontweight='bold')

axes_flat = axes.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes_flat[idx]

    actual = predictions_df['actual'].values
    predicted = predictions_df[model_name].values

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Add metrics
    rmse = results[model_name]['test_rmse']
    r2 = results[model_name]['test_r2']
    mae = results[model_name]['test_mae']

    ax.set_title(f'{model_name}\nRMSE: {rmse:.6f}, R²: {r2:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Price', fontsize=10)
    ax.set_ylabel('Predicted Price', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/nn_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_scatter_plots.png")
plt.close()

# ============================================================================
# 5. PREDICTION ERRORS ANALYSIS
# ============================================================================
print("\n[5/5] Creating prediction errors analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Neural Network Prediction Errors Distribution', fontsize=16, fontweight='bold')

axes_flat = axes.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes_flat[idx]

    actual = predictions_df['actual'].values
    predicted = predictions_df[model_name].values
    errors = actual - predicted

    # Histogram
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(errors.mean(), color='g', linestyle='--', linewidth=2,
               label=f'Mean: {errors.mean():.6f}')

    # Add statistics
    std_error = np.std(errors)

    ax.set_title(f'{model_name}\nStd Dev: {std_error:.6f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Error', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/nn_error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_error_distribution.png")
plt.close()

# ============================================================================
# 6. ENSEMBLE WEIGHTS VISUALIZATION
# ============================================================================
print("\n[6/6] Creating ensemble weights visualization...")

if 'NN_Ensemble' in results and 'weights' in results['NN_Ensemble']:
    fig, ax = plt.subplots(figsize=(10, 6))

    weights_dict = results['NN_Ensemble']['weights']
    models_list = list(weights_dict.keys())
    weights_list = list(weights_dict.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(models_list)))
    bars = ax.bar(models_list, weights_list, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title('Neural Network Ensemble Weights', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(weights_list) * 1.2)

    # Add value labels
    for bar, weight in zip(bars, weights_list):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/nn_ensemble_weights.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/nn_ensemble_weights.png")
    plt.close()

# ============================================================================
# 7. RECENT PREDICTIONS ZOOM
# ============================================================================
print("\n[7/7] Creating recent predictions zoom...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Neural Network Recent Predictions (Last 100 Samples)', fontsize=16, fontweight='bold')

axes_flat = axes.flatten()

# Take last 100 samples
n_recent = min(100, len(predictions_df))
recent_data = predictions_df.tail(n_recent)

for idx, model_name in enumerate(model_names):
    ax = axes_flat[idx]

    actual = recent_data['actual'].values
    predicted = recent_data[model_name].values
    time_idx = np.arange(len(actual))

    # Plot
    ax.plot(time_idx, actual, label='Actual', alpha=0.8, linewidth=2, marker='o', markersize=3)
    ax.plot(time_idx, predicted, label='Predicted', alpha=0.8, linewidth=2, marker='s', markersize=3)

    # Add metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    ax.set_title(f'{model_name}\nRMSE: {rmse:.6f}, MAE: {mae:.6f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/nn_recent_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_recent_predictions.png")
plt.close()

# ============================================================================
# 8. PERFORMANCE METRICS HEATMAP
# ============================================================================
print("\n[8/8] Creating performance metrics heatmap...")

# Create metrics dataframe
metrics_data = {
    'RMSE': rmse_values,
    'MAE': mae_values,
    'R²': r2_values,
    'MAPE (%)': mape_values
}
metrics_df = pd.DataFrame(metrics_data, index=models)

# Normalize for heatmap (lower is better for RMSE, MAE, MAPE; higher is better for R²)
metrics_normalized = metrics_df.copy()
for col in ['RMSE', 'MAE', 'MAPE (%)']:
    metrics_normalized[col] = 1 - (metrics_normalized[col] - metrics_normalized[col].min()) / (metrics_normalized[col].max() - metrics_normalized[col].min())
metrics_normalized['R²'] = (metrics_normalized['R²'] - metrics_normalized['R²'].min()) / (metrics_normalized['R²'].max() - metrics_normalized['R²'].min())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(metrics_normalized, annot=metrics_df, fmt='.4f', cmap='RdYlGn',
            linewidths=1, cbar_kws={'label': 'Normalized Score (Higher is Better)'},
            ax=ax)
ax.set_title('Neural Network Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/nn_metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_metrics_heatmap.png")
plt.close()

# ============================================================================
# 9. MODEL RANKING
# ============================================================================
print("\n[9/9] Creating model ranking visualization...")

# Calculate overall score (normalized)
overall_scores = metrics_normalized.mean(axis=1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(overall_scores)))
bars = ax.barh(range(len(overall_scores)), overall_scores.values, color=colors, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(overall_scores)))
ax.set_yticklabels(overall_scores.index)
ax.set_xlabel('Overall Performance Score (Normalized)', fontsize=12, fontweight='bold')
ax.set_title('Neural Network Model Ranking\n(Based on Combined Metrics)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (bar, score) in enumerate(zip(bars, overall_scores.values)):
    ax.text(score, bar.get_y() + bar.get_height()/2, f' {score:.4f}',
            va='center', fontsize=11, fontweight='bold')

# Add ranking numbers
for i, bar in enumerate(bars):
    ax.text(0.01, bar.get_y() + bar.get_height()/2, f'#{i+1}',
            va='center', ha='left', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', edgecolor='white', linewidth=2))

ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, 1.1)

plt.tight_layout()
plt.savefig('visualizations/nn_model_ranking.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/nn_model_ranking.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION SUMMARY")
print("=" * 80)

# Find best model
best_model = overall_scores.index[0]
best_score = overall_scores.values[0]

print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
print(f"   Overall Score: {best_score:.4f}")
print(f"   RMSE: {results[best_model]['test_rmse']:.6f}")
print(f"   MAE: {results[best_model]['test_mae']:.6f}")
print(f"   R²: {results[best_model]['test_r2']:.6f}")

# Best per metric
print(f"\n📊 BEST MODELS BY METRIC:")
print(f"   Lowest RMSE: {models[np.argmin(rmse_values)]} ({min(rmse_values):.6f})")
print(f"   Lowest MAE: {models[np.argmin(mae_values)]} ({min(mae_values):.6f})")
print(f"   Highest R²: {models[np.argmax(r2_values)]} ({max(r2_values):.4f})")
print(f"   Lowest MAPE: {models[np.argmin(mape_values)]} ({min(mape_values):.4f}%)")

print(f"\n✓ Generated 9 visualization files in visualizations/")
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
