"""Model training and management endpoints."""
from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from src.prediction.training import train_and_register_lightgbm, train_and_register_lstm
from src.prediction.registry import ModelRegistry
from src.prediction.config import PredictionConfig


router = APIRouter()

# In-memory training job tracking (simple for MVP)
training_jobs: Dict[str, Dict[str, Any]] = {}


class TrainModelRequest(BaseModel):
    currency_pair: str = Field(..., description="Currency pair (e.g., USD/EUR)")
    model_type: str = Field(..., description="Model type: lightgbm or lstm")
    horizons: Optional[List[int]] = Field(None, description="Prediction horizons in days (for lightgbm) or hours (for lstm)")
    version: str = Field(default="1.0", description="Model version string")
    history_days: Optional[int] = Field(None, description="Historical window in days to use for training")
    # LightGBM specific
    gbm_rounds: Optional[int] = Field(None, description="Number of boosting rounds (default: 120)")
    gbm_patience: Optional[int] = Field(None, description="Early stopping patience (default: 10)")
    gbm_learning_rate: Optional[float] = Field(None, description="Learning rate (default: 0.05)")
    gbm_num_leaves: Optional[int] = Field(None, description="Max num leaves (default: 31)")
    # LSTM specific
    lstm_epochs: Optional[int] = Field(5, description="Training epochs (default: 5)")
    lstm_hidden_dim: Optional[int] = Field(64, description="Hidden dimension (default: 64)")
    lstm_seq_len: Optional[int] = Field(64, description="Sequence length (default: 64)")
    lstm_lr: Optional[float] = Field(0.001, description="Learning rate (default: 0.001)")
    lstm_interval: Optional[str] = Field(None, description="Intraday interval for LSTM data (default: 1h)")


class TrainModelResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, training, completed, error
    progress: int
    message: str
    model_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/train", response_model=TrainModelResponse)
def train_model(request: TrainModelRequest, background: BackgroundTasks):
    """Start training a new model for a currency pair."""
    # Validate model type
    if request.model_type not in ["lightgbm", "lstm"]:
        raise HTTPException(status_code=400, detail="model_type must be 'lightgbm' or 'lstm'")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Store initial job status
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Training queued",
        "currency_pair": request.currency_pair,
        "model_type": request.model_type,
    }

    # Start background training task
    background.add_task(_train_model_task, job_id, request)

    return TrainModelResponse(
        job_id=job_id,
        status="pending",
        message=f"Training job {job_id} started for {request.currency_pair} ({request.model_type})",
    )


def _train_model_task(job_id: str, request: TrainModelRequest):
    """Background task to train model."""
    try:
        # Update status
        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["progress"] = 10
        training_jobs[job_id]["message"] = "Loading data and building features..."

        # Train based on model type
        if request.model_type == "lightgbm":
            # Training LightGBM
            training_jobs[job_id]["progress"] = 30
            training_jobs[job_id]["message"] = "Training LightGBM model..."

            async def _train_gbm():
                return await train_and_register_lightgbm(
                    currency_pair=request.currency_pair,
                    horizons=request.horizons,
                    days=request.history_days,
                    version=request.version,
                    gbm_rounds=request.gbm_rounds,
                    gbm_patience=request.gbm_patience,
                    gbm_learning_rate=request.gbm_learning_rate,
                    gbm_num_leaves=request.gbm_num_leaves,
                )

            metadata = asyncio.run(_train_gbm())

        else:  # lstm
            # Training LSTM
            training_jobs[job_id]["progress"] = 30
            training_jobs[job_id]["message"] = "Training LSTM model..."

            async def _train_lstm():
                return await train_and_register_lstm(
                    currency_pair=request.currency_pair,
                    days=request.history_days or 180,
                    interval=request.lstm_interval or "1h",
                    horizons_hours=request.horizons or [1, 4, 24],
                    version=request.version,
                    lstm_epochs=request.lstm_epochs or 5,
                    lstm_hidden_dim=request.lstm_hidden_dim or 64,
                    lstm_seq_len=request.lstm_seq_len or 64,
                    lstm_lr=request.lstm_lr or 0.001,
                )

            metadata = asyncio.run(_train_lstm())

        # Update with success
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["message"] = "Model training completed"
        training_jobs[job_id]["model_id"] = metadata.model_id
        training_jobs[job_id]["metadata"] = {
            "model_id": metadata.model_id,
            "currency_pair": metadata.currency_pair,
            "trained_at": metadata.trained_at.isoformat() if metadata.trained_at else None,
            "validation_metrics": metadata.validation_metrics,
            "horizons": metadata.horizons,
        }

    except Exception as e:
        # Update with error
        training_jobs[job_id]["status"] = "error"
        training_jobs[job_id]["progress"] = 0
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        training_jobs[job_id]["error"] = str(e)


@router.get("/train/status/{job_id}", response_model=TrainingStatusResponse)
def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]
    return TrainingStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        model_id=job.get("model_id"),
        error=job.get("error"),
    )


@router.get("/")
def list_models(
    currency_pair: str = Query(None, description="Filter by currency pair"),
    model_type: str = Query(None, description="Filter by model type (lightgbm|lstm)"),
):
    """List all trained models with optional filters."""
    config = PredictionConfig.from_yaml()
    registry = ModelRegistry(config.model_registry_path, config.model_storage_dir)

    models = registry.list_models(currency_pair=currency_pair, model_type=model_type)

    return {
        "total": len(models),
        "models": [
            {
                "model_id": m.get("model_id"),
                "model_type": m.get("model_type"),
                "currency_pair": m.get("currency_pair"),
                "trained_at": m.get("trained_at"),
                "version": m.get("version"),
                "horizons": m.get("horizons"),
                "calibration_ok": m.get("calibration_ok"),
                "min_samples": m.get("min_samples"),
            }
            for m in models
        ],
    }


@router.get("/{model_id}")
def get_model_details(model_id: str):
    """Get detailed information about a specific model."""
    config = PredictionConfig.from_yaml()
    registry = ModelRegistry(config.model_registry_path, config.model_storage_dir)

    model_info = registry.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")

    return model_info


@router.delete("/{model_id}")
def delete_model(model_id: str):
    """Delete a trained model."""
    config = PredictionConfig.from_yaml()
    registry = ModelRegistry(config.model_registry_path, config.model_storage_dir)

    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "status": "deleted",
        "model_id": model_id,
    }


@router.get("/registry/info")
def get_registry_info():
    """Get registry metadata and statistics."""
    config = PredictionConfig.from_yaml()
    registry = ModelRegistry(config.model_registry_path, config.model_storage_dir)

    all_models = registry.list_models()

    # Compute stats
    by_type = {}
    by_pair = {}
    for model in all_models:
        model_type = model.get("model_type", "unknown")
        pair = model.get("currency_pair", "unknown")

        by_type[model_type] = by_type.get(model_type, 0) + 1
        by_pair[pair] = by_pair.get(pair, 0) + 1

    return {
        "registry_path": registry.registry_path,
        "storage_dir": registry.storage_dir,
        "total_models": len(all_models),
        "models_by_type": by_type,
        "models_by_pair": by_pair,
    }
