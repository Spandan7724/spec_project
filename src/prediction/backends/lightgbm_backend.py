import logging
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BasePredictorBackend

logger = logging.getLogger(__name__)


class LightGBMBackend(BasePredictorBackend):
    """LightGBM-based predictor with quantile regression and direction classification."""

    def __init__(self):
        self.models: Dict[int, lgb.Booster] = {}
        self.quantile_models: Dict[int, Dict[str, lgb.Booster]] = {}
        self.direction_models: Dict[int, lgb.Booster] = {}
        self.scaler = StandardScaler()
        self.validation_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None,
        num_boost_round: int = 120,
        patience: int = 10,
    ) -> Dict:
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1,
            }

        logger.info("Starting LightGBM training")
        self.feature_names = list(X_train.columns)

        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)

        metrics: Dict[str, Dict[str, float]] = {}

        for horizon in horizons:
            target_col = f"target_{horizon}d"
            direction_col = f"direction_{horizon}d"
            if target_col not in y_train.columns:
                logger.warning("Target %s not found, skipping", target_col)
                continue

            valid_idx = ~y_train[target_col].isna()
            X_h = X_scaled[valid_idx]
            y_h = y_train[target_col][valid_idx]
            y_dir = y_train[direction_col][valid_idx] if direction_col in y_train.columns else None

            if len(X_h) < 100:
                logger.warning("Insufficient samples for %sd: %s", horizon, len(X_h))
                continue

            split_idx = int(len(X_h) * 0.8)
            X_tr, X_val = X_h.iloc[:split_idx], X_h.iloc[split_idx:]
            y_tr, y_val = y_h.iloc[:split_idx], y_h.iloc[split_idx:]
            if y_dir is not None:
                y_dir_tr, y_dir_val = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
            else:
                y_dir_tr, y_dir_val = None, None

            # Mean regression model
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.models[horizon] = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(0)],
            )

            # Quantile models
            self.quantile_models[horizon] = {}
            for quantile in [0.1, 0.5, 0.9]:
                q_params = params.copy()
                q_params["objective"] = "quantile"
                q_params["alpha"] = quantile
                self.quantile_models[horizon][f"p{int(quantile*100)}"] = lgb.train(
                    q_params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(0)],
                )

            # Direction classifier
            if y_dir_tr is not None:
                dir_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "verbose": -1,
                }
                dir_train = lgb.Dataset(X_tr, label=y_dir_tr)
                dir_val = lgb.Dataset(X_val, label=y_dir_val, reference=dir_train)
                self.direction_models[horizon] = lgb.train(
                    dir_params,
                    dir_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[dir_val],
                    callbacks=[lgb.early_stopping(patience), lgb.log_evaluation(0)],
                )

            # Validation metrics
            y_pred = self.models[horizon].predict(X_val)
            rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_val - y_pred)))

            # Directional accuracy
            if horizon in self.direction_models:
                y_dir_pred = self.direction_models[horizon].predict(X_val)
                dir_acc = float(np.mean((y_dir_pred > 0.5) == y_dir_val))
            else:
                # fallback estimate if direction model isn't present
                dir_acc = 0.5

            # Quantile coverage using p10 and p90
            q10_pred = self.quantile_models[horizon]["p10"].predict(X_val)
            q90_pred = self.quantile_models[horizon]["p90"].predict(X_val)
            coverage = float(np.mean((y_val >= q10_pred) & (y_val <= q90_pred)))

            metrics[f"{horizon}d"] = {
                "rmse": rmse,
                "mae": mae,
                "directional_accuracy": dir_acc,
                "quantile_coverage": coverage,
                "n_samples": int(len(X_val)),
            }

            logger.info(
                "H%s: RMSE=%.4f, DirAcc=%.3f, Coverage=%.3f",
                horizon,
                rmse,
                dir_acc,
                coverage,
            )

        self.validation_metrics = metrics
        return metrics

    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int],
        include_quantiles: bool = True,
    ) -> Dict[int, Dict]:
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        results: Dict[int, Dict] = {}
        for horizon in horizons:
            if horizon not in self.models:
                logger.warning("No model for %sd horizon", horizon)
                continue
            pred: Dict[str, float] = {}
            pred["mean_change"] = float(self.models[horizon].predict(X_scaled)[0])
            if include_quantiles and horizon in self.quantile_models:
                pred["quantiles"] = {
                    "p10": float(self.quantile_models[horizon]["p10"].predict(X_scaled)[0]),
                    "p50": float(self.quantile_models[horizon]["p50"].predict(X_scaled)[0]),
                    "p90": float(self.quantile_models[horizon]["p90"].predict(X_scaled)[0]),
                }
            if horizon in self.direction_models:
                pred["direction_prob"] = float(self.direction_models[horizon].predict(X_scaled)[0])
            results[horizon] = pred

        return results

    def get_model_confidence(self) -> float:
        """
        Get model confidence based on R² score, not directional accuracy.
        Directional accuracy ~50% is normal for forex but doesn't reflect model quality.
        R² measures how well we predict the magnitude of changes.
        """
        if not self.validation_metrics:
            return 0.5  # Default moderate confidence

        # Use R² score if available (best metric for regression quality)
        r2_scores = [m.get("r2", 0.0) for m in self.validation_metrics.values()]
        if r2_scores and any(r2 > 0 for r2 in r2_scores):
            avg_r2 = float(np.mean([r2 for r2 in r2_scores if r2 > 0]))
            # R² can be negative for bad models, clip to 0-1 range
            return float(max(0.0, min(1.0, avg_r2)))

        # Fallback: use MAE-based confidence (lower MAE = higher confidence)
        mae_values = [m.get("mae", 1.0) for m in self.validation_metrics.values()]
        if mae_values:
            avg_mae = float(np.mean(mae_values))
            # Convert MAE to confidence: smaller MAE = higher confidence
            # Typical forex MAE ranges from 0.001 (excellent) to 1.0 (poor)
            # Use exponential decay: confidence = e^(-MAE * 5)
            confidence = float(np.exp(-avg_mae * 5))
            return max(0.0, min(1.0, confidence))

        return 0.5  # Default moderate confidence

    def get_feature_importance(self, horizon: int, top_n: int = 10) -> Dict[str, float]:
        if horizon not in self.models:
            return {}
        imp = self.models[horizon].feature_importance(importance_type="gain")
        mapping = dict(zip(self.feature_names, imp))
        sorted_items = sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def save(self, path: str) -> None:
        import pickle

        state = {
            "models": {h: m.model_to_string() for h, m in self.models.items()},
            "quantile_models": {
                h: {q: m.model_to_string() for q, m in qm.items()} for h, qm in self.quantile_models.items()
            },
            "direction_models": {h: m.model_to_string() for h, m in self.direction_models.items()},
            "scaler": self.scaler,
            "validation_metrics": self.validation_metrics,
            "feature_names": self.feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.models = {h: lgb.Booster(model_str=s) for h, s in state["models"].items()}
        self.quantile_models = {
            h: {q: lgb.Booster(model_str=s) for q, s in qm.items()} for h, qm in state["quantile_models"].items()
        }
        self.direction_models = {h: lgb.Booster(model_str=s) for h, s in state["direction_models"].items()}
        self.scaler = state["scaler"]
        self.validation_metrics = state["validation_metrics"]
        self.feature_names = state.get("feature_names", [])
