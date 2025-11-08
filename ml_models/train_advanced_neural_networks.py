"""
Advanced Neural Network Models for Forex Price Prediction
Implements state-of-the-art architectures based on 2024-2025 research:
- Hybrid LSTM-Transformer with Attention
- Attention-Based LSTM (ALFA)
- Bidirectional LSTM with Multi-Head Attention
- CNN-LSTM Hybrid
- Temporal Convolutional Network (TCN)
- Ensemble of all architectures
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("=" * 80)
print("ADVANCED NEURAL NETWORK TRAINING FOR FOREX PREDICTION")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1/7] Loading and preprocessing data...")

from train_forex_models import create_advanced_features

df = pd.read_csv('data/eurusd_d.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df_features = create_advanced_features(df)
df_features['Target_1d'] = df_features['Close'].shift(-1)
df_features = df_features.dropna()

exclude_cols = ['Date', 'Target_1d', 'Target_3d', 'Target_7d', 'Close', 'Open', 'High', 'Low']
with open('models/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size].copy()
test_data = df_features.iloc[train_size:].copy()

X_train = train_data[feature_cols].values
y_train = train_data['Target_1d'].values
X_test = test_data[feature_cols].values
y_test = test_data['Target_1d'].values

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# 2. PYTORCH DATASET
# ============================================================================
class ForexDataset(Dataset):
    def __init__(self, X, y, sequence_length=30):
        """
        Create sequences for time series prediction
        sequence_length: number of past days to use for prediction
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of past data
        X_seq = self.X[idx:idx + self.sequence_length]
        # Get target (next day price)
        y_target = self.y[idx + self.sequence_length]

        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target])

# Create datasets with sequences
sequence_length = 30  # Use 30 days of history
train_dataset = ForexDataset(X_train_scaled, y_train, sequence_length)
test_dataset = ForexDataset(X_test_scaled, y_test, sequence_length)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Sequence length: {sequence_length} days")
print(f"Batch size: {batch_size}")

# ============================================================================
# 3. ADVANCED NEURAL NETWORK ARCHITECTURES
# ============================================================================
print("\n[2/7] Defining advanced neural network architectures...")

# ============================================================================
# Model 1: Attention-Based LSTM (ALFA)
# ============================================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class ALFA_Model(nn.Module):
    """Attention-Based LSTM for Forex (ALFA)"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(ALFA_Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        self.attention = AttentionLayer(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context_vector, _ = self.attention(lstm_out)

        # Fully connected layers
        output = self.fc(context_vector)
        return output

# ============================================================================
# Model 2: Hybrid LSTM-Transformer
# ============================================================================
class LSTMTransformerHybrid(nn.Module):
    """Hybrid LSTM-Transformer Model"""
    def __init__(self, input_size, hidden_size=256, num_heads=8, num_layers=2, dropout=0.3):
        super(LSTMTransformerHybrid, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,  # Bidirectional
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Transformer processing
        transformer_out = self.transformer(lstm_out)

        # Attention pooling
        attention_weights = self.attention_pool(transformer_out)
        pooled = torch.sum(attention_weights * transformer_out, dim=1)

        # Final prediction
        output = self.fc(pooled)
        return output

# ============================================================================
# Model 3: Bidirectional LSTM with Multi-Head Attention
# ============================================================================
class BiLSTMMultiHeadAttention(nn.Module):
    """Bidirectional LSTM with Multi-Head Attention"""
    def __init__(self, input_size, hidden_size=256, num_heads=8, dropout=0.3):
        super(BiLSTMMultiHeadAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Multi-head attention with residual connection
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        x1 = self.layer_norm1(lstm_out + attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x1)
        x2 = self.layer_norm2(x1 + ffn_out)

        # Global average pooling
        pooled = torch.mean(x2, dim=1)

        # Final prediction
        output = self.fc(pooled)
        return output

# ============================================================================
# Model 4: CNN-LSTM Hybrid
# ============================================================================
class CNNLSTMHybrid(nn.Module):
    """CNN-LSTM Hybrid for feature extraction and temporal modeling"""
    def __init__(self, input_size, hidden_size=256, dropout=0.3):
        super(CNNLSTMHybrid, self).__init__()

        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = AttentionLayer(hidden_size * 2)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # CNN layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Transpose back for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context_vector, _ = self.attention(lstm_out)

        # Final prediction
        output = self.fc(context_vector)
        return output

# ============================================================================
# Model 5: Temporal Convolutional Network (TCN)
# ============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.3):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Chomp padding
        out = self.dropout1(self.relu1(self.bn1(out)))

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Chomp padding
        out = self.dropout2(self.relu2(self.bn2(out)))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, num_channels=[128, 256, 256, 128], kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Transpose for conv1d
        x = x.transpose(1, 2)

        # TCN processing
        y = self.network(x)

        # Global average pooling
        y = torch.mean(y, dim=2)

        # Final prediction
        output = self.fc(y)
        return output

# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================
print("\n[3/7] Preparing training functions...")

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, patience=20):
    """Train a model with early stopping"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, train_losses, test_losses

def predict(model, data_loader):
    """Make predictions"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions).flatten()

# ============================================================================
# 5. TRAIN ALL MODELS
# ============================================================================
print("\n[4/7] Training advanced neural network models...")

input_size = X_train.shape[1]
results = {}
models_dict = {}

# Model 1: ALFA
print("\n[4.1] Training ALFA (Attention-Based LSTM)...")
alfa_model = ALFA_Model(input_size, hidden_size=256, num_layers=3, dropout=0.3)
alfa_model, _, _ = train_model(alfa_model, train_loader, test_loader, epochs=100, lr=0.001)
alfa_pred = predict(alfa_model, test_loader)

# Align predictions with actual test data
y_test_aligned = y_test[sequence_length:]
alfa_rmse = np.sqrt(mean_squared_error(y_test_aligned, alfa_pred))
alfa_mae = mean_absolute_error(y_test_aligned, alfa_pred)
alfa_r2 = r2_score(y_test_aligned, alfa_pred)

results['ALFA'] = {'test_rmse': alfa_rmse, 'test_mae': alfa_mae, 'test_r2': alfa_r2}
models_dict['ALFA'] = alfa_model
print(f"ALFA - Test RMSE: {alfa_rmse:.6f}, Test R²: {alfa_r2:.6f}")

# Model 2: LSTM-Transformer Hybrid
print("\n[4.2] Training LSTM-Transformer Hybrid...")
lstm_trans_model = LSTMTransformerHybrid(input_size, hidden_size=256, num_heads=8, dropout=0.3)
lstm_trans_model, _, _ = train_model(lstm_trans_model, train_loader, test_loader, epochs=100, lr=0.001)
lstm_trans_pred = predict(lstm_trans_model, test_loader)

lstm_trans_rmse = np.sqrt(mean_squared_error(y_test_aligned, lstm_trans_pred))
lstm_trans_mae = mean_absolute_error(y_test_aligned, lstm_trans_pred)
lstm_trans_r2 = r2_score(y_test_aligned, lstm_trans_pred)

results['LSTM_Transformer'] = {'test_rmse': lstm_trans_rmse, 'test_mae': lstm_trans_mae, 'test_r2': lstm_trans_r2}
models_dict['LSTM_Transformer'] = lstm_trans_model
print(f"LSTM-Transformer - Test RMSE: {lstm_trans_rmse:.6f}, Test R²: {lstm_trans_r2:.6f}")

# Model 3: BiLSTM with Multi-Head Attention
print("\n[4.3] Training BiLSTM with Multi-Head Attention...")
bilstm_attn_model = BiLSTMMultiHeadAttention(input_size, hidden_size=256, num_heads=8, dropout=0.3)
bilstm_attn_model, _, _ = train_model(bilstm_attn_model, train_loader, test_loader, epochs=100, lr=0.001)
bilstm_attn_pred = predict(bilstm_attn_model, test_loader)

bilstm_attn_rmse = np.sqrt(mean_squared_error(y_test_aligned, bilstm_attn_pred))
bilstm_attn_mae = mean_absolute_error(y_test_aligned, bilstm_attn_pred)
bilstm_attn_r2 = r2_score(y_test_aligned, bilstm_attn_pred)

results['BiLSTM_MultiHead'] = {'test_rmse': bilstm_attn_rmse, 'test_mae': bilstm_attn_mae, 'test_r2': bilstm_attn_r2}
models_dict['BiLSTM_MultiHead'] = bilstm_attn_model
print(f"BiLSTM-MultiHead - Test RMSE: {bilstm_attn_rmse:.6f}, Test R²: {bilstm_attn_r2:.6f}")

# Model 4: CNN-LSTM Hybrid
print("\n[4.4] Training CNN-LSTM Hybrid...")
cnn_lstm_model = CNNLSTMHybrid(input_size, hidden_size=256, dropout=0.3)
cnn_lstm_model, _, _ = train_model(cnn_lstm_model, train_loader, test_loader, epochs=100, lr=0.001)
cnn_lstm_pred = predict(cnn_lstm_model, test_loader)

cnn_lstm_rmse = np.sqrt(mean_squared_error(y_test_aligned, cnn_lstm_pred))
cnn_lstm_mae = mean_absolute_error(y_test_aligned, cnn_lstm_pred)
cnn_lstm_r2 = r2_score(y_test_aligned, cnn_lstm_pred)

results['CNN_LSTM'] = {'test_rmse': cnn_lstm_rmse, 'test_mae': cnn_lstm_mae, 'test_r2': cnn_lstm_r2}
models_dict['CNN_LSTM'] = cnn_lstm_model
print(f"CNN-LSTM - Test RMSE: {cnn_lstm_rmse:.6f}, Test R²: {cnn_lstm_r2:.6f}")

# Model 5: TCN
print("\n[4.5] Training Temporal Convolutional Network (TCN)...")
tcn_model = TCN(input_size, num_channels=[128, 256, 256, 128], dropout=0.3)
tcn_model, _, _ = train_model(tcn_model, train_loader, test_loader, epochs=100, lr=0.001)
tcn_pred = predict(tcn_model, test_loader)

tcn_rmse = np.sqrt(mean_squared_error(y_test_aligned, tcn_pred))
tcn_mae = mean_absolute_error(y_test_aligned, tcn_pred)
tcn_r2 = r2_score(y_test_aligned, tcn_pred)

results['TCN'] = {'test_rmse': tcn_rmse, 'test_mae': tcn_mae, 'test_r2': tcn_r2}
models_dict['TCN'] = tcn_model
print(f"TCN - Test RMSE: {tcn_rmse:.6f}, Test R²: {tcn_r2:.6f}")

# ============================================================================
# 6. ENSEMBLE OF NEURAL NETWORKS
# ============================================================================
print("\n[5/7] Creating Neural Network Ensemble...")

all_predictions = [alfa_pred, lstm_trans_pred, bilstm_attn_pred, cnn_lstm_pred, tcn_pred]
all_rmses = [alfa_rmse, lstm_trans_rmse, bilstm_attn_rmse, cnn_lstm_rmse, tcn_rmse]

# Weighted ensemble based on inverse RMSE
inv_rmse = [1/rmse for rmse in all_rmses]
weights = [w/sum(inv_rmse) for w in inv_rmse]

print(f"Ensemble weights:")
for name, weight in zip(['ALFA', 'LSTM_Transformer', 'BiLSTM_MultiHead', 'CNN_LSTM', 'TCN'], weights):
    print(f"   {name}: {weight:.4f}")

ensemble_pred = sum(w * pred for w, pred in zip(weights, all_predictions))

ensemble_rmse = np.sqrt(mean_squared_error(y_test_aligned, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test_aligned, ensemble_pred)
ensemble_r2 = r2_score(y_test_aligned, ensemble_pred)

results['NN_Ensemble'] = {
    'test_rmse': ensemble_rmse,
    'test_mae': ensemble_mae,
    'test_r2': ensemble_r2,
    'weights': dict(zip(['ALFA', 'LSTM_Transformer', 'BiLSTM_MultiHead', 'CNN_LSTM', 'TCN'], [float(w) for w in weights]))
}

print(f"NN Ensemble - Test RMSE: {ensemble_rmse:.6f}, Test R²: {ensemble_r2:.6f}")

# ============================================================================
# 7. SAVE MODELS AND RESULTS
# ============================================================================
print("\n[6/7] Saving models and results...")

# Save PyTorch models
for name, model in models_dict.items():
    torch.save(model.state_dict(), f'models/nn_{name.lower()}_model.pt')
    print(f"✓ Saved: models/nn_{name.lower()}_model.pt")

# Save results
with open('models/advanced_nn_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("✓ Saved: models/advanced_nn_results.json")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test_aligned,
    'ALFA': alfa_pred,
    'LSTM_Transformer': lstm_trans_pred,
    'BiLSTM_MultiHead': bilstm_attn_pred,
    'CNN_LSTM': cnn_lstm_pred,
    'TCN': tcn_pred,
    'NN_Ensemble': ensemble_pred
})
predictions_df.to_csv('models/nn_predictions.csv', index=False)
print("✓ Saved: models/nn_predictions.csv")

# ============================================================================
# 8. RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[7/7] ADVANCED NEURAL NETWORK RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results).T.sort_values('test_rmse')

print("\n📊 NEURAL NETWORK PERFORMANCE (Sorted by Test RMSE)")
print("-" * 80)
print(f"{'Model':<25} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<12}")
print("-" * 80)
for model_name, metrics in results_df.iterrows():
    print(f"{model_name:<25} {metrics['test_rmse']:<12.6f} {metrics['test_mae']:<12.6f} {metrics['test_r2']:<12.6f}")

print("\n🏆 BEST NEURAL NETWORK: " + results_df.index[0])
print(f"   Test RMSE: {results_df.iloc[0]['test_rmse']:.6f}")
print(f"   Test MAE: {results_df.iloc[0]['test_mae']:.6f}")
print(f"   Test R²: {results_df.iloc[0]['test_r2']:.6f}")

# Calculate MAPE
best_pred = all_predictions[list(results.keys()).index(results_df.index[0])]
mape = np.mean(np.abs((y_test_aligned - best_pred) / y_test_aligned)) * 100
print(f"   Test MAPE: {mape:.4f}%")

print("\n" + "=" * 80)
print("ADVANCED NEURAL NETWORK TRAINING COMPLETE!")
print("=" * 80)
