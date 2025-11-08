"""
CatBoost Backend for Forex Prediction
Implements the winning CatBoost_3 configuration with conservative hyperparameters
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.preprocessing import RobustScaler

from .base import BasePredictorBackend

logger = logging.getLogger(__name__)


class CatBoostBackend(BasePredictorBackend):
    """
    CatBoost-based predictor with quantile regression and direction classification.

    Uses the winning configuration from model benchmarking:
    - Conservative hyperparameters to prevent overfitting
    - Small learning rate with many iterations
    - Strong regularization
    - GPU support if available
    """

    def __init__(self, task_type: str = "auto"):
        """
        Initialize CatBoost backend.

        Args:
            task_type: "CPU", "GPU", or "auto" (auto-detect)
        """
        self.models: Dict[int, CatBoostRegressor] = {}
        self.quantile_models: Dict[int, Dict[str, CatBoostRegressor]] = {}
        self.direction_models: Dict[int, CatBoostClassifier] = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.validation_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []

        # Determine task type
        if task_type == "auto":
            try:
                # Test GPU availability
                test_pool = Pool([[1, 2, 3]], [1])
                test_model = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
                test_model.fit(test_pool)
                self.task_type = "GPU"
                logger.info("GPU detected! CatBoost will use GPU acceleration.")
            except:
                self.task_type = "CPU"
                logger.info("GPU not available. CatBoost will use CPU.")
        else:
            self.task_type = task_type

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None,
        num_boost_round: int = 4000,
        patience: int = 300,
    ) -> Dict:
        """
        Train CatBoost models for each prediction horizon.

        Args:
            X_train: Feature DataFrame
            y_train: Target DataFrame with columns target_{horizon}d and direction_{horizon}d
            horizons: List of prediction horizons in days
            params: Optional parameter overrides
            num_boost_round: Maximum number of boosting iterations
            patience: Early stopping patience (iterations without improvement)

        Returns:
            Dictionary of validation metrics
        """
        # Use winning CatBoost_3 configuration by default
        if params is None:
            params = {
                "iterations": num_boost_round,
                "learning_rate": 0.005,  # Small learning rate
                "depth": 6,  # Conservative depth
                "l2_leaf_reg": 10,  # Strong regularization
                "min_data_in_leaf": 30,  # Prevent overfitting
                "random_strength": 0.1,  # Low randomness
                "bagging_temperature": 0.1,  # Conservative bagging
                "border_count": 254,
                "task_type": self.task_type,
                "verbose": False,
                "early_stopping_rounds": patience,
                "random_seed": 42,
            }
        else:
            # Merge with defaults
            default_params = {
                "task_type": self.task_type,
                "verbose": False,
                "early_stopping_rounds": patience,
                "random_seed": 42,
            }
            params = {**default_params, **params}

        logger.info(f"Starting CatBoost training (Task: {self.task_type})")
        self.feature_names = list(X_train.columns)

        # Scale features using RobustScaler (better for financial data)
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)

        metrics: Dict[str, Dict[str, float]] = {}

        for horizon in horizons:
            target_col = f"target_{horizon}d"
            direction_col = f"direction_{horizon}d"

            if target_col not in y_train.columns:
                logger.warning(f"Target {target_col} not found, skipping")
                continue

            # Filter valid samples
            valid_idx = ~y_train[target_col].isna()
            X_h = X_scaled[valid_idx]
            y_h = y_train[target_col][valid_idx]
            y_dir = y_train[direction_col][valid_idx] if direction_col in y_train.columns else None

            if len(X_h) < 100:
                logger.warning(f"Insufficient samples for {horizon}d: {len(X_h)}")
                continue

            # Train/validation split (80/20)
            split_idx = int(len(X_h) * 0.8)
            X_tr, X_val = X_h.iloc[:split_idx], X_h.iloc[split_idx:]
            y_tr, y_val = y_h.iloc[:split_idx], y_h.iloc[split_idx:]

            if y_dir is not None:
                y_dir_tr, y_dir_val = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
            else:
                y_dir_tr, y_dir_val = None, None

            # Create CatBoost Pools for better performance
            train_pool = Pool(X_tr, y_tr)
            val_pool = Pool(X_val, y_val)

            logger.info(f"Training CatBoost for horizon {horizon}d...")

            # ================================================================
            # 1. Mean Regression Model
            # ================================================================
            self.models[horizon] = CatBoostRegressor(**params)
            self.models[horizon].fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                verbose=False,
            )

            logger.info(
                f"  Mean model: {self.models[horizon].get_best_iteration()} iterations, "
                f"best score: {self.models[horizon].get_best_score()['validation']['RMSE']:.6f}"
            )

            # ================================================================
            # 2. Quantile Regression Models (10th, 50th, 90th percentiles)
            # ================================================================
            self.quantile_models[horizon] = {}

            for quantile in [0.1, 0.5, 0.9]:
                q_params = params.copy()
                q_params["loss_function"] = f"Quantile:alpha={quantile}"

                quantile_model = CatBoostRegressor(**q_params)
                quantile_model.fit(
                    train_pool,
                    eval_set=val_pool,
                    use_best_model=True,
                    verbose=False,
                )

                self.quantile_models[horizon][f"p{int(quantile*100)}"] = quantile_model
                logger.info(f"  Quantile {int(quantile*100)}% model: {quantile_model.get_best_iteration()} iterations")

            # ================================================================
            # 3. Direction Classification Model
            # ================================================================
            if y_dir_tr is not None:
                dir_params = {
                    "iterations": num_boost_round,
                    "learning_rate": 0.01,
                    "depth": 6,
                    "l2_leaf_reg": 5,
                    "loss_function": "Logloss",
                    "task_type": self.task_type,
                    "verbose": False,
                    "early_stopping_rounds": patience,
                    "random_seed": 42,
                }

                dir_train_pool = Pool(X_tr, y_dir_tr)
                dir_val_pool = Pool(X_val, y_dir_val)

                self.direction_models[horizon] = CatBoostClassifier(**dir_params)
                self.direction_models[horizon].fit(
                    dir_train_pool,
                    eval_set=dir_val_pool,
                    use_best_model=True,
                    verbose=False,
                )

                logger.info(f"  Direction model: {self.direction_models[horizon].get_best_iteration()} iterations")

            # ================================================================
            # 4. Compute Validation Metrics
            # ================================================================
            y_pred = self.models[horizon].predict(X_val)
            rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_val - y_pred)))

            # R² score
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Directional accuracy
            if horizon in self.direction_models:
                y_dir_pred = self.direction_models[horizon].predict(X_val)
                dir_acc = float(np.mean(y_dir_pred == y_dir_val))
            else:
                # Fallback: check if mean prediction direction matches actual
                price_changes = y_val.values
                pred_changes = y_pred
                dir_acc = float(np.mean(np.sign(price_changes) == np.sign(pred_changes)))

            # Quantile coverage (should be ~80% for 10-90 range)
            q10_pred = self.quantile_models[horizon]["p10"].predict(X_val)
            q90_pred = self.quantile_models[horizon]["p90"].predict(X_val)
            coverage = float(np.mean((y_val >= q10_pred) & (y_val <= q90_pred)))

            metrics[f"{horizon}d"] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "directional_accuracy": dir_acc,
                "quantile_coverage": coverage,
                "n_samples": int(len(X_val)),
                "best_iteration": int(self.models[horizon].get_best_iteration()),
            }

            logger.info(
                f"Horizon {horizon}d: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, "
                f"DirAcc={dir_acc:.3f}, Coverage={coverage:.3f}"
            )

        self.validation_metrics = metrics
        logger.info("CatBoost training complete!")
        return metrics

    def predict(self, X: pd.DataFrame, horizons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate predictions for given features.

        Args:
            X: Feature DataFrame
            horizons: Optional list of horizons to predict. If None, predicts all trained horizons.

        Returns:
            DataFrame with predictions for each horizon
        """
        if not self.models:
            raise RuntimeError("No models trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        horizons = horizons or list(self.models.keys())
        predictions = {}

        for horizon in horizons:
            if horizon not in self.models:
                logger.warning(f"No model for horizon {horizon}d")
                continue

            # Mean prediction
            predictions[f"pred_{horizon}d"] = self.models[horizon].predict(X_scaled)

            # Quantile predictions
            if horizon in self.quantile_models:
                predictions[f"pred_{horizon}d_p10"] = self.quantile_models[horizon]["p10"].predict(X_scaled)
                predictions[f"pred_{horizon}d_p50"] = self.quantile_models[horizon]["p50"].predict(X_scaled)
                predictions[f"pred_{horizon}d_p90"] = self.quantile_models[horizon]["p90"].predict(X_scaled)

            # Direction prediction
            if horizon in self.direction_models:
                predictions[f"direction_{horizon}d"] = self.direction_models[horizon].predict(X_scaled)
                predictions[f"direction_{horizon}d_proba"] = self.direction_models[horizon].predict_proba(X_scaled)[:, 1]

        return pd.DataFrame(predictions, index=X.index)

    def save(self, path: str) -> None:
        """Save model state to file."""
        import pickle

        state = {
            "models": {h: m.save_model(return_value=True) for h, m in self.models.items()},
            "quantile_models": {
                h: {q: m.save_model(return_value=True) for q, m in qm.items()}
                for h, qm in self.quantile_models.items()
            },
            "direction_models": {h: m.save_model(return_value=True) for h, m in self.direction_models.items()},
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "validation_metrics": self.validation_metrics,
            "task_type": self.task_type,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"CatBoost backend saved to {path}")

    def load(self, path: str) -> None:
        """Load model state from file."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Restore models
        self.models = {}
        for h, model_bytes in state["models"].items():
            self.models[h] = CatBoostRegressor()
            self.models[h].load_model(model_bytes, format="cbm")

        self.quantile_models = {}
        for h, qmodels in state["quantile_models"].items():
            self.quantile_models[h] = {}
            for q, model_bytes in qmodels.items():
                model = CatBoostRegressor()
                model.load_model(model_bytes, format="cbm")
                self.quantile_models[h][q] = model

        self.direction_models = {}
        for h, model_bytes in state["direction_models"].items():
            self.direction_models[h] = CatBoostClassifier()
            self.direction_models[h].load_model(model_bytes, format="cbm")

        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.validation_metrics = state.get("validation_metrics", {})
        self.task_type = state.get("task_type", "CPU")

        logger.info(f"CatBoost backend loaded from {path}")

    def get_feature_importance(self, horizon: int, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for a specific horizon.

        Args:
            horizon: Prediction horizon in days
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if horizon not in self.models:
            raise ValueError(f"No model trained for horizon {horizon}d")

        importances = self.models[horizon].get_feature_importance()

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)

        return importance_df
