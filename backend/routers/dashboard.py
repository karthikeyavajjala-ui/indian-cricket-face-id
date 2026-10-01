"""Dashboard + prediction history endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models_db import Prediction
from ..schemas import DashboardSummary, PredictionRow
from ..services import dashboard_service

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard/summary", response_model=DashboardSummary,
            summary="Cards, charts and recent predictions")
def dashboard_summary(session: Session = Depends(get_session)) -> DashboardSummary:
    payload = dashboard_service.summary(session)
    payload["disclaimer"] = (
        "Statistics describe this deployment only. Accuracy figures come from a "
        "held-out split of your own dataset - never treat them as real-world "
        "performance for people, lighting or cameras outside that dataset."
    )
    return DashboardSummary(**payload)


@router.get("/predictions", response_model=list[PredictionRow],
            summary="Prediction audit trail")
def predictions(limit: int = Query(25, ge=1, le=200),
                status: str | None = Query(None, description="identified | unknown | no_face"),
                session: Session = Depends(get_session)) -> list[PredictionRow]:
    query = select(Prediction).order_by(Prediction.created_at.desc(),
                                        Prediction.id.desc()).limit(limit)
    if status:
        query = query.where(Prediction.status == status)
    rows = session.scalars(query).all()
    return [PredictionRow(**dashboard_service.prediction_row(r)) for r in rows]


@router.get("/predictions/{prediction_id}/detail", summary="Stored top-k for one attempt")
def prediction_detail(prediction_id: int, session: Session = Depends(get_session)) -> dict:
    row = session.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown prediction")
    data = dashboard_service.prediction_row(row)
    try:
        data["top_predictions"] = json.loads(row.top_predictions or "[]")
    except json.JSONDecodeError:
        data["top_predictions"] = []
    return data


@router.delete("/predictions", summary="Clear the prediction history (admin)")
def clear_predictions(session: Session = Depends(get_session)) -> dict:
    rows = session.scalars(select(Prediction)).all()
    for row in rows:
        session.delete(row)
    session.commit()
    return {"deleted": len(rows)}
