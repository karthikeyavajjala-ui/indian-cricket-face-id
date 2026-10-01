"""
Training endpoint - protected.

Auth: send the key configured in ``IC_TRAIN_API_KEY`` as

    X-API-Key: <key>

If ``IC_ALLOW_UNSIGNED_TRAINING=false`` the key becomes mandatory (recommended
whenever the API is reachable from anywhere but localhost). With the default
local setup the endpoint is open but loudly reports that it is unprotected.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_session
from ..schemas import (TrainAcceptedResponse, TrainJobResponse, TrainRequest)
from ..services import model_service, training_service

log = logging.getLogger("icfid.api.train")
router = APIRouter(prefix="/api/train", tags=["training"])


def require_key(x_api_key: str | None = Header(default=None)) -> bool:
    key = settings.training_api_key
    if settings.allow_unsigned_training and (key in ("", "change-me-in-production")):
        return False                      # documented local-only convenience
    if not x_api_key or x_api_key != key:
        raise HTTPException(status_code=401,
                            detail="A valid X-API-Key header is required to start training.")
    return True


@router.post("", response_model=TrainAcceptedResponse, status_code=202,
             summary="Start a training job on a dataset folder")
def start_training(request: TrainRequest, session: Session = Depends(get_session),
                   authenticated: bool = Depends(require_key)) -> TrainAcceptedResponse:
    if training_service.is_training():
        raise HTTPException(status_code=409,
                            detail=f"A training job is already running "
                                   f"({training_service.active_job_id()}).")
    try:
        dataset = training_service.resolve_dataset(request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not dataset.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset folder not found: {dataset}")

    options = request.model_dump(exclude_none=True)
    job = training_service.create_job(session, dataset, options)
    if not training_service.start_job(job.job_id):
        raise HTTPException(status_code=409, detail="Another training job just started.")
    log.info("Training job %s started on %s", job.job_id, dataset)
    return TrainAcceptedResponse(
        job_id=job.job_id, status="queued",
        message=("Training started. Poll /api/train/status/{job_id} for progress. "
                 "Training runs in the API process; keep this machine awake."),
        dataset=str(dataset),
    )


@router.get("/status/{job_id}", response_model=TrainJobResponse,
            summary="Progress of a training job")
def job_status(job_id: str, session: Session = Depends(get_session)) -> TrainJobResponse:
    payload = training_service.job_payload(session, job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return TrainJobResponse(**payload)


@router.get("/jobs", summary="Recent training jobs")
def jobs(session: Session = Depends(get_session)) -> dict:
    return {"running": training_service.active_job_id(),
            "jobs": training_service.list_jobs(session)}


@router.get("/capabilities", summary="Available classifiers / extractors / detectors")
def capabilities() -> dict:
    from ml.face import detector_status
    from ml.features import available_extractors
    from ml.model import CLASSIFIERS
    return {
        "classifiers": list(CLASSIFIERS),
        "extractors": available_extractors(),
        "detector": detector_status(),
        "training_protected": not (settings.allow_unsigned_training and
                                   settings.training_api_key == "change-me-in-production"),
        "note": ("Extractors marked unavailable need python scripts/fetch_models.py "
                 "(sface) or a different OpenCV build."),
    }
