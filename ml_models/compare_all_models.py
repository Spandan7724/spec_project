"""
Comprehensive Model Comparison Script
Compares traditional ML models with advanced neural networks
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

# Load results from both training sessions
with open('models/training_results.json', 'r') as f:
    ml_results = json.load(f)

try:
    with open('models/advanced_nn_results.json', 'r') as f:
        nn_results = json.load(f)
    nn_available = True
except FileNotFoundError:
    print("\n⚠️  Advanced NN results not found. Run train_advanced_neural_networks.py first.")
    nn_available = False
    nn_results = {}

# Combine results
all_results = {}

# Add ML model results
for model_name, metrics in ml_results['model_performance'].items():
    # Skip if metrics is a dict containing weights
    if isinstance(metrics, dict) and 'test_rmse' in metrics:
        all_results[model_name] = {
            'test_rmse': metrics['test_rmse'],
            'test_mae': metrics['test_mae'],
            'test_r2': metrics['test_r2'],
            'model_type': 'Traditional ML'
        }

# Add NN results
if nn_available:
    for model_name, metrics in nn_results.items():
        all_results[f"NN_{model_name}"] = {
            'test_rmse': metrics['test_rmse'],
            'test_mae': metrics['test_mae'],
            'test_r2': metrics['test_r2'],
            'model_type': 'Neural Network'
        }

# Create DataFrame
df = pd.DataFrame(all_results).T
df = df.sort_values('test_rmse')

print("\n" + "=" * 100)
print("COMPLETE MODEL RANKING (All Models)")
print("=" * 100)
print(f"{'Rank':<6} {'Model':<30} {'Type':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("=" * 100)

for rank, (model_name, row) in enumerate(df.iterrows(), 1):
    print(f"{rank:<6} {model_name:<30} {row['model_type']:<20} {row['test_rmse']:<12.6f} {row['test_mae']:<12.6f} {row['test_r2']:<12.6f}")

print("\n🏆 OVERALL BEST MODEL: " + df.index[0])
print(f"   Model Type: {df.iloc[0]['model_type']}")
print(f"   Test RMSE: {df.iloc[0]['test_rmse']:.6f}")
print(f"   Test MAE: {df.iloc[0]['test_mae']:.6f}")
print(f"   Test R²: {df.iloc[0]['test_r2']:.6f}")

# Save comparison
df.to_csv('models/complete_model_comparison.csv')
print(f"\n✅ Saved comparison to: models/complete_model_comparison.csv")

# Create comparison visualization
if nn_available:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('ML Models vs Neural Networks Comparison', fontsize=16, fontweight='bold')

    # Plot 1: RMSE comparison
    ax1 = axes[0]
    colors = ['#2E86AB' if t == 'Traditional ML' else '#A23B72' for t in df['model_type']]
    bars = ax1.barh(df.index, df['test_rmse'], color=colors)
    ax1.set_xlabel('Test RMSE', fontweight='bold')
    ax1.set_title('RMSE Comparison', fontweight='bold')
    ax1.axvline(x=df['test_rmse'].min(), color='red', linestyle='--', alpha=0.5)
    ax1.invert_yaxis()

    # Plot 2: R² comparison
    ax2 = axes[1]
    bars = ax2.barh(df.index, df['test_r2'], color=colors)
    ax2.set_xlabel('Test R² Score', fontweight='bold')
    ax2.set_title('R² Score Comparison', fontweight='bold')
    ax2.axvline(x=df['test_r2'].max(), color='red', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()

    # Plot 3: Model type summary
    ax3 = axes[2]
    type_summary = df.groupby('model_type').agg({
        'test_rmse': 'mean',
        'test_r2': 'mean'
    })
    x = np.arange(len(type_summary))
    width = 0.35
    ax3.bar(x - width/2, type_summary['test_rmse'], width, label='Avg RMSE', alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, type_summary['test_r2'], width, label='Avg R²', alpha=0.8, color='orange')
    ax3.set_xlabel('Model Type', fontweight='bold')
    ax3.set_ylabel('Average RMSE', fontweight='bold')
    ax3_twin.set_ylabel('Average R²', fontweight='bold')
    ax3.set_title('Average Performance by Model Type', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(type_summary.index, rotation=15, ha='right')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('visualizations/10_complete_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: visualizations/10_complete_model_comparison.png")
    plt.close()

print("\n" + "=" * 100)
print("COMPARISON COMPLETE!")
print("=" * 100)
