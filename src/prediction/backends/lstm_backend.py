from typing import Dict, List, Optional

import pandas as pd

from .base import BasePredictorBackend


class LSTMBackend(BasePredictorBackend):
    """LSTM-based predictor (intraday horizons).

    Implementation uses PyTorch if available; otherwise raises a clear error on train/predict.
    This backend focuses on producing mean_change predictions per horizon.
    """

    def __init__(self, seq_len: int = 64, hidden_dim: int = 64, dropout: float = 0.2, mc_samples: int = 30):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.models: Dict[int, object] = {}
        self.scaler = None
        self.validation_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

    def _require_torch(self):
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("PyTorch is required for LSTMBackend but is not installed.") from e

    def _build_sequences(self, X: pd.DataFrame, y: pd.Series, seq_len: int):
        import torch

        # Build sliding windows
        data = torch.tensor(X.values, dtype=torch.float32)
        targets = torch.tensor(y.values, dtype=torch.float32)
        seqs, tars = [], []
        for i in range(seq_len, len(X)):
            seqs.append(data[i - seq_len : i])
            tars.append(targets[i])
        return torch.stack(seqs), torch.stack(tars)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None,
    ) -> Dict:
        self._require_torch()
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        if params is None:
            params = {"epochs": 5, "lr": 1e-3}

        self.feature_names = list(X_train.columns)
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )

        metrics: Dict[str, Dict[str, float]] = {}

        class LSTMModel(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.do = nn.Dropout(p=dropout)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                # Use last hidden output
                last = out[:, -1, :]
                last = self.do(last)
                return self.head(last).squeeze(-1)

        for horizon in horizons:
            target_col = f"target_{horizon}d"
            if target_col not in y_train.columns:
                continue
            y_h = y_train[target_col].dropna()
            X_h = X_scaled.loc[y_h.index]

            if len(X_h) <= self.seq_len + 5:
                # Not enough samples
                continue

            X_seq, y_seq = self._build_sequences(X_h, y_h, self.seq_len)
            model = LSTMModel(input_dim=X_h.shape[1], hidden_dim=self.hidden_dim, dropout=self.dropout)
            opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(params["epochs"]):
                opt.zero_grad()
                pred = model(X_seq)
                loss = loss_fn(pred, y_seq)
                loss.backward()
                opt.step()

            self.models[horizon] = model
            # Minimal metrics; real validation could be added similarly
            metrics[f"{horizon}d"] = {"rmse": float(torch.sqrt(loss).item()), "n_samples": int(len(X_seq))}

        self.validation_metrics = metrics
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int],
        include_quantiles: bool = True,
    ) -> Dict[int, Dict]:
        self._require_torch()
        import torch

        if self.scaler is None:
            raise RuntimeError("Model not fitted; scaler missing")

        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        # Build last sequence for prediction
        if len(X_scaled) < self.seq_len:
            raise ValueError("Insufficient rows to build LSTM sequence for prediction")
        data = torch.tensor(X_scaled.values[-self.seq_len :], dtype=torch.float32).unsqueeze(0)

        results: Dict[int, Dict] = {}
        for horizon in horizons:
            if horizon not in self.models:
                continue
            model = self.models[horizon]
            # Base deterministic prediction
            model.eval()
            with torch.no_grad():
                mean_pred = float(model(data).item())
            out: Dict[str, float] = {"mean_change": mean_pred}
            # MC dropout for uncertainty and direction prob
            if include_quantiles:
                preds = []
                model.train()  # enable dropout
                with torch.no_grad():
                    for _ in range(self.mc_samples):
                        preds.append(float(model(data).item()))
                import numpy as np
                q10, q50, q90 = np.percentile(preds, [10, 50, 90]).tolist()
                out["quantiles"] = {"p10": float(q10), "p50": float(q50), "p90": float(q90)}
                # Direction probability as fraction of positive samples
                dir_prob = float(np.mean(np.array(preds) > 0.0))
                out["direction_prob"] = dir_prob
            results[horizon] = out
        return results

    def get_model_confidence(self) -> float:
        """
        Get model confidence based on R² or loss metrics.
        """
        if not self.validation_metrics:
            return 0.3  # Default low-moderate confidence for LSTM

        # Try to extract R² if available
        r2_scores = []
        mae_values = []

        for metrics in self.validation_metrics.values():
            if isinstance(metrics, dict):
                if 'r2' in metrics:
                    r2_scores.append(metrics['r2'])
                if 'mae' in metrics:
                    mae_values.append(metrics['mae'])

        # Use R² if available
        if r2_scores and any(r2 > 0 for r2 in r2_scores):
            avg_r2 = float(np.mean([r2 for r2 in r2_scores if r2 > 0]))
            return float(max(0.0, min(1.0, avg_r2)))

        # Fallback to MAE-based confidence
        if mae_values:
            avg_mae = float(np.mean(mae_values))
            confidence = float(np.exp(-avg_mae * 5))
            return max(0.0, min(1.0, confidence))

        return 0.3  # Default low-moderate confidence

    def save(self, path: str) -> None:
        """Save LSTM backend state to disk (per-horizon state_dicts + scaler)."""
        import pickle

        # Pack model state_dicts per horizon
        model_states: Dict[int, dict] = {}
        for h, model in self.models.items():
            # Ensure we export CPU state for portability
            model_states[h] = {k: v.cpu() for k, v in model.state_dict().items()}

        state = {
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "feature_names": self.feature_names,
            "validation_metrics": self.validation_metrics,
            "scaler": self.scaler,
            "models": model_states,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load LSTM backend state from disk."""
        import pickle
        import torch.nn as nn

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.seq_len = int(state.get("seq_len", self.seq_len))
        self.hidden_dim = int(state.get("hidden_dim", self.hidden_dim))
        self.feature_names = list(state.get("feature_names", []))
        self.validation_metrics = state.get("validation_metrics", {})
        self.scaler = state.get("scaler")

        input_dim = len(self.feature_names) if self.feature_names else None
        if input_dim is None or input_dim <= 0:
            raise RuntimeError("Invalid or missing feature_names for LSTM model load")

        class LSTMModel(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.do = nn.Dropout(p=dropout)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                last = self.do(last)
                return self.head(last).squeeze(-1)

        # Rebuild models per horizon
        self.models = {}
        for h, state_dict in (state.get("models", {}) or {}).items():
            model = LSTMModel(input_dim=input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)
            model.load_state_dict(state_dict)
            self.models[int(h)] = model
