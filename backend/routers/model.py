"""Model status, metrics, versions and hot-reload."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ml import config as ml_config

from ..config import settings
from ..database import get_session
from ..schemas import ModelMetricsResponse, ModelStatusResponse, ReloadResponse
from ..services import model_service, training_service
from .train import require_key

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status", response_model=ModelStatusResponse,
            summary="Is a model loaded, what engine is it using, is the dataset ready?")
def status(session: Session = Depends(get_session)) -> ModelStatusResponse:
    payload = model_service.status_payload(training_service.is_training())
    try:
        model_service.sync_active_versions(session)
        session.commit()
    except Exception:                                     # pragma: no cover
        session.rollback()
    payload["disclaimer"] = (
        "Demonstration of facial identification technology. Not for surveillance, law "
        "enforcement, access control or any high-impact decision without appropriate "
        "safeguards. Only people present in the training dataset can be recognised."
    )
    return ModelStatusResponse(**payload)


@router.get("/metrics", response_model=ModelMetricsResponse,
            summary="Honest evaluation report for the active model")
def metrics() -> ModelMetricsResponse:
    return ModelMetricsResponse(**model_service.metrics_payload())


@router.get("/versions", summary="Every trained model recorded in the database")
def versions(limit: int = Query(20, ge=1, le=100),
             session: Session = Depends(get_session)) -> dict:
    try:
        model_service.sync_active_versions(session)
        session.commit()
    except Exception:                                     # pragma: no cover
        session.rollback()
    return {"versions": model_service.list_model_versions(session, limit)}


@router.post("/reload", response_model=ReloadResponse, summary="Reload the model from disk")
def reload_model() -> ReloadResponse:
    identifier = model_service.reload_identifier()
    return ReloadResponse(ok=identifier.ready,
                          message=("Model reloaded." if identifier.ready
                                   else identifier.error or "No model available."),
                          engine=identifier.engine_info())


@router.get("/available", summary="Model files present on disk")
def available_models() -> dict:
    return {"models": model_service.available_models(),
            "active": model_service.resolve_model_path()[0] and
                      str(model_service.resolve_model_path()[0])}


@router.post("/activate", response_model=ReloadResponse,
             summary="Switch the active model (protected)")
def activate_model(mode: str = Query(..., pattern="^(trained|demo)$"),
                   authenticated: bool = Depends(require_key)) -> ReloadResponse:
    from ..services import training_service                    # noqa: F401 (circular guard)
    target = settings.model_path if mode == "trained" else settings.demo_model_path
    if not target.is_file():
        raise HTTPException(status_code=404,
                            detail=f"No {mode} model at {target}. Train one first.")
    model_service.set_active_model(target)
    identifier = model_service.reload_identifier()
    return ReloadResponse(ok=identifier.ready, message=f"Switched to the {mode} model.",
                          engine=identifier.engine_info())


@router.get("/dataset", summary="Dataset inventory used for the active model")
def dataset() -> dict:
    return {
        "dataset": ml_config.dataset_summary_safe(settings.dataset_dir),
        "demo_dataset": ml_config.dataset_summary_safe(settings.demo_dataset_dir),
        "active": ("dataset" if ml_config.dataset_summary_safe(settings.dataset_dir)["configured"]
                   else "dataset_demo"),
    }
