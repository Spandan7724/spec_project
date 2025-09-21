"""
Model storage and management utilities
"""

import json
import joblib
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class ModelStorage:
    """
    Centralized model storage and versioning system
    """
    
    def __init__(self, storage_path: str = "models/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_path / "model_registry.json"
        
        # Load or create registry
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'models': {},
                'default_models': {},
                'created': datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save model registry to disk"""
        self.registry['last_updated'] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   currency_pair: str,
                   model_type: str = "lstm",
                   metadata: Dict[str, Any] = None,
                   set_as_default: bool = False) -> str:
        """
        Save a model with metadata and versioning
        
        Args:
            model: The model object to save
            model_name: Name for this model
            currency_pair: Currency pair this model is for
            model_type: Type of model (lstm, arima, etc.)
            metadata: Additional metadata
            set_as_default: Whether to set as default for this currency pair
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        # Use model_name directly as model_id
        # The predictor should pass the correctly formatted name based on versioning strategy
        model_id = model_name.lower().replace('/', '')
        
        # Create model directory
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model file
        model_file = model_dir / "model.pth"
        if hasattr(model, 'save_model'):
            model.save_model(str(model_file))
        else:
            # Fallback for other model types
            joblib.dump(model, model_file)
        
        # Prepare metadata
        full_metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'currency_pair': currency_pair,
            'model_id': model_id,
            'created': datetime.now().isoformat(),
            'file_path': str(model_file),
            'directory': str(model_dir)
        }
        
        # Add model-specific info if available
        if hasattr(model, 'get_model_info'):
            full_metadata['model_info'] = model.get_model_info()
        
        # Add custom metadata
        if metadata:
            full_metadata.update(metadata)
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Update registry
        self.registry['models'][model_id] = full_metadata
        
        # Set as default if requested
        if set_as_default:
            if currency_pair not in self.registry['default_models']:
                self.registry['default_models'][currency_pair] = {}
            self.registry['default_models'][currency_pair][model_type] = model_id
        
        self._save_registry()
        
        logger.info(f"Saved model {model_id} for {currency_pair}")
        return model_id
    
    def load_model(self, 
                   model_id: str = None,
                   currency_pair: str = None,
                   model_type: str = "lstm") -> Any:
        """
        Load a model by ID or get default model for currency pair
        
        Args:
            model_id: Specific model ID to load
            currency_pair: Currency pair to get default model for
            model_type: Type of model to load
            
        Returns:
            Loaded model object
        """
        if model_id is None:
            # Get default model
            if currency_pair is None:
                raise ValueError("Must specify either model_id or currency_pair")
            
            default_models = self.registry['default_models'].get(currency_pair, {})
            model_id = default_models.get(model_type)
            
            if model_id is None:
                raise ValueError(f"No default {model_type} model found for {currency_pair}")
        
        # Check if model exists
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registry['models'][model_id]
        
        # Handle both absolute and relative paths
        model_file_path = metadata['file_path']
        if Path(model_file_path).is_absolute():
            model_file = Path(model_file_path)
        else:
            # For relative paths, resolve relative to storage_path
            # Extract just the model directory name from the stored path
            model_dir_name = Path(model_file_path).parts[-2]  # e.g., "lstm_usdeur_20250901_184913"
            model_file = self.storage_path / model_dir_name / "model.pth"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model based on type
        if model_type == "lstm":
            from ..models.lstm_model import LSTMModel
            
            # Get model config from metadata
            model_info = metadata.get('model_info', {})
            config = model_info.get('config', {})
            config = dict(config)
            target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config['device'] = target_device

            model = LSTMModel(config)
            model.load_model(str(model_file))
            model.device = target_device
            model.to(target_device)
            
        else:
            # Generic loading
            model = joblib.load(model_file)
        
        logger.info(f"Loaded model {model_id}")
        return model
    
    def list_models(self, 
                   currency_pair: str = None,
                   model_type: str = None) -> List[Dict[str, Any]]:
        """
        List available models with optional filtering
        
        Args:
            currency_pair: Filter by currency pair
            model_type: Filter by model type
            
        Returns:
            List of model metadata
        """
        models = []
        
        for model_id, metadata in self.registry['models'].items():
            # Apply filters
            if currency_pair and metadata.get('currency_pair') != currency_pair:
                continue
            if model_type and metadata.get('model_type') != model_type:
                continue
            
            # Add default flag
            metadata_copy = metadata.copy()
            is_default = False
            
            pair = metadata.get('currency_pair')
            mtype = metadata.get('model_type')
            
            if pair in self.registry['default_models']:
                default_for_type = self.registry['default_models'][pair].get(mtype)
                is_default = (default_for_type == model_id)
            
            metadata_copy['is_default'] = is_default
            models.append(metadata_copy)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created', ''), reverse=True)
        
        return models
    
    def delete_model(self, model_id: str, confirm: bool = False):
        """
        Delete a model and its files
        
        Args:
            model_id: Model ID to delete
            confirm: Must be True to actually delete
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete model")
        
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.registry['models'][model_id]
        model_dir = Path(metadata['directory'])
        
        # Remove files
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.registry['models'][model_id]
        
        # Remove from defaults if it was default
        for pair, defaults in self.registry['default_models'].items():
            for mtype, default_id in list(defaults.items()):
                if default_id == model_id:
                    del defaults[mtype]
        
        self._save_registry()
        
        logger.info(f"Deleted model {model_id}")
    
    def set_default_model(self, 
                         model_id: str,
                         currency_pair: str = None,
                         model_type: str = None):
        """
        Set a model as default for a currency pair and model type
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.registry['models'][model_id]
        
        # Use metadata values if not provided
        currency_pair = currency_pair or metadata['currency_pair']
        model_type = model_type or metadata['model_type']
        
        # Set as default
        if currency_pair not in self.registry['default_models']:
            self.registry['default_models'][currency_pair] = {}
        
        self.registry['default_models'][currency_pair][model_type] = model_id
        self._save_registry()
        
        logger.info(f"Set {model_id} as default {model_type} model for {currency_pair}")
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.registry['models'][model_id]
        
        # Extract performance info from metadata
        performance = {}
        
        if 'performance_metrics' in metadata:
            performance = metadata['performance_metrics']
        
        # Add training history if available
        model_info = metadata.get('model_info', {})
        if 'training_history_length' in model_info:
            performance['training_epochs'] = model_info['training_history_length']
        
        return performance
    
    def cleanup_old_models(self, keep_per_pair: int = 5):
        """
        Clean up old models, keeping only the most recent ones
        
        Args:
            keep_per_pair: Number of models to keep per currency pair
        """
        # Group models by currency pair and type
        grouped_models = {}
        
        for model_id, metadata in self.registry['models'].items():
            pair = metadata.get('currency_pair', 'unknown')
            mtype = metadata.get('model_type', 'unknown')
            key = f"{pair}_{mtype}"
            
            if key not in grouped_models:
                grouped_models[key] = []
            
            grouped_models[key].append((model_id, metadata.get('created', '')))
        
        # Delete old models
        deleted_count = 0
        
        for key, models in grouped_models.items():
            # Sort by creation date
            models.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the most recent ones
            to_delete = models[keep_per_pair:]
            
            for model_id, _ in to_delete:
                # Don't delete if it's a default model
                is_default = False
                for pair_defaults in self.registry['default_models'].values():
                    if model_id in pair_defaults.values():
                        is_default = True
                        break
                
                if not is_default:
                    self.delete_model(model_id, confirm=True)
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_models = len(self.registry['models'])
        
        # Count by type and pair
        type_counts = {}
        pair_counts = {}
        
        for metadata in self.registry['models'].values():
            mtype = metadata.get('model_type', 'unknown')
            pair = metadata.get('currency_pair', 'unknown')
            
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # Calculate storage size
        total_size = 0
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        return {
            'total_models': total_models,
            'models_by_type': type_counts,
            'models_by_pair': pair_counts,
            'storage_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.storage_path)
        }
