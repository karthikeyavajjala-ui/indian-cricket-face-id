"""Health + public configuration endpoints."""
from __future__ import annotations

import platform
import sys

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import config as ml_config
from ml.face import detector_status
from ml.features import available_extractors

from ..config import settings
from ..database import get_session
from ..models_db import Prediction, Player
from ..services import model_service, training_service

router = APIRouter(prefix="/api", tags=["system"])

DISCLAIMER = ("This system is a demonstration of facial identification technology and "
              "should not be used for surveillance, law enforcement, access control, or "
              "high-impact decisions without appropriate safeguards. It recognises only "
              "the people present in its training dataset - it cannot identify every "
              "Indian cricketer.")


@router.get("/health", summary="Liveness + subsystem snapshot")
def health(session: Session = Depends(get_session)) -> dict:
    path, mode = model_service.resolve_model_path()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "database": settings.database_url.split("://", 1)[0],
        "model": {"active": str(path) if path else None, "mode": mode,
                  "ready": model_service.get_identifier().ready},
        "engines": {"detector": detector_status(),
                    "extractors": available_extractors()},
        "dataset": ml_config.dataset_summary_safe(settings.dataset_dir),
        "demo_dataset": ml_config.dataset_summary_safe(settings.demo_dataset_dir),
        "database_counts": {
            "players": session.scalar(select(func.count(Player.id))) or 0,
            "predictions": session.scalar(select(func.count(Prediction.id))) or 0,
        },
        "training": {"in_progress": training_service.is_training(),
                     "job": training_service.active_job_id()},
    }


@router.get("/config", summary="Public configuration consumed by the frontend")
def public_config() -> dict:
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "confidence_threshold": settings.confidence_threshold,
        "min_margin": settings.min_margin,
        "demo_enabled": settings.demo_enabled,
        "max_upload_mb": settings.max_upload_mb,
        "model_ready": model_service.get_identifier().ready,
        "dataset_configured": ml_config.dataset_summary_safe(
            settings.dataset_dir)["configured"],
        "endpoints": [
            "GET  /api/players", "GET  /api/players/{id}", "POST /api/predict",
            "POST /api/predict/image", "GET  /api/model/status",
            "GET  /api/model/metrics", "POST /api/train (protected)",
            "GET  /api/dashboard/summary", "GET  /docs",
        ],
        "disclaimer": DISCLAIMER,
        "responsible_ai": [
            "Only people in the trained dataset can be recognised - everyone else is "
            "reported as Unknown.",
            "Below the confidence threshold the app refuses to name anyone.",
            "Metrics are measured on your dataset (with augmentation leakage caveats) "
            "and are not a real-world accuracy guarantee.",
            "Images you upload are stored for auditing; delete data/uploads and the "
            "predictions table if you do not want that.",
        ],
    }
