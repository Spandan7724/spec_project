from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BasePredictorBackend(ABC):
    """Base interface for prediction backends."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        horizons: List[int],
        params: Optional[Dict] = None,
    ) -> Dict:
        """Train model and return validation metrics."""

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int],
        include_quantiles: bool = True,
    ) -> Dict[int, Dict]:
        """Predict and return mapping: horizon -> {mean_change, quantiles?, direction_prob?}."""

    @abstractmethod
    def get_model_confidence(self) -> float:
        """Return overall 0-1 confidence derived from validation metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state from disk."""

