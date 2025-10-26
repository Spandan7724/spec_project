import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.prediction.models import ModelMetadata
from src.utils.logging import get_logger
from src.utils.paths import find_project_root, resolve_project_path


logger = get_logger(__name__)


class ModelRegistry:
    """Simple JSON-based model registry with pickle storage."""

    def __init__(self, registry_path: str, storage_dir: str):
        """
        Initialize model registry.

        Args:
            registry_path: Path to JSON metadata file
            storage_dir: Directory to store model pickle files
        """
        root = find_project_root()
        self.registry_path = str(resolve_project_path(registry_path, root))
        self.storage_dir = str(resolve_project_path(storage_dir, root))

        # Ensure directories exist
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize registry file
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from JSON file. Create empty file if missing."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    self.registry: Dict[str, Dict] = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} models")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.registry = {}
        else:
            self.registry = {}
            self._save_registry()
            logger.info("Initialized empty registry")

    def _save_registry(self) -> None:
        """Persist registry to JSON file."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(
        self, metadata: ModelMetadata, model_obj: Any, scaler_obj: Optional[Any] = None
    ) -> None:
        """
        Register a new model and persist model/scaler artifacts.

        Args:
            metadata: Model metadata (will be copied into registry)
            model_obj: Trained model object (pickle-able)
            scaler_obj: Optional scaler object (pickle-able)
        """
        logger.info(f"Registering model: {metadata.model_id}")

        # Save model
        model_path = os.path.join(self.storage_dir, f"{metadata.model_id}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        metadata.model_path = model_path

        # Save scaler
        scaler_path: Optional[str] = None
        if scaler_obj is not None:
            scaler_path = os.path.join(
                self.storage_dir, f"{metadata.model_id}_scaler.pkl"
            )
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler_obj, f)
            metadata.scaler_path = scaler_path

        # Store metadata (serialize datetimes)
        self.registry[metadata.model_id] = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "currency_pair": metadata.currency_pair,
            "trained_at": metadata.trained_at.isoformat()
            if isinstance(metadata.trained_at, datetime)
            else str(metadata.trained_at),
            "version": metadata.version,
            "validation_metrics": metadata.validation_metrics,
            "min_samples": metadata.min_samples,
            "calibration_ok": metadata.calibration_ok,
            "features_used": metadata.features_used,
            "horizons": metadata.horizons,
            "model_path": metadata.model_path,
            "scaler_path": scaler_path,
        }

        self._save_registry()
        logger.info(f"Successfully registered model: {metadata.model_id}")

    def get_model(self, currency_pair: str, model_type: str = "lightgbm") -> Optional[Dict]:
        """
        Get latest model for a currency pair and type.
        Returns None if not found.
        """
        matches = [
            m
            for m in self.registry.values()
            if m.get("currency_pair") == currency_pair and m.get("model_type") == model_type
        ]
        if not matches:
            return None

        def _dt(val: str) -> datetime:
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return datetime.min

        latest = max(matches, key=lambda m: _dt(m.get("trained_at", "")))
        return latest

    def load_model_objects(self, model_metadata: Dict) -> Tuple[Any, Optional[Any]]:
        """Load model (and optional scaler) objects from disk."""
        model_path = model_metadata["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)

        scaler_obj = None
        scaler_path = model_metadata.get("scaler_path")
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler_obj = pickle.load(f)

        return model_obj, scaler_obj

    def list_models(
        self, currency_pair: Optional[str] = None, model_type: Optional[str] = None
    ) -> List[Dict]:
        """List all models, optionally filtered, sorted by trained_at desc."""
        models = list(self.registry.values())
        if currency_pair:
            models = [m for m in models if m.get("currency_pair") == currency_pair]
        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        def _dt(val: str) -> datetime:
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return datetime.min

        return sorted(models, key=lambda m: _dt(m.get("trained_at", "")), reverse=True)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its artifacts."""
        if model_id not in self.registry:
            return False
        meta = self.registry[model_id]
        # Delete files
        try:
            if meta.get("model_path") and os.path.exists(meta["model_path"]):
                os.remove(meta["model_path"])
            if meta.get("scaler_path") and os.path.exists(meta["scaler_path"]):
                os.remove(meta["scaler_path"])
        finally:
            # Remove from registry regardless of file deletion success
            self.registry.pop(model_id, None)
            self._save_registry()
        return True

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get metadata for a specific model."""
        return self.registry.get(model_id)
