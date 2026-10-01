"""Identification endpoints: multipart upload + base64 (camera) input."""
from __future__ import annotations

import base64
import binascii
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_session
from ..schemas import PredictBase64Request, PredictionResponse
from ..services import model_service, prediction_service

log = logging.getLogger("icfid.api.predict")
router = APIRouter(prefix="/api", tags=["predict"])

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@router.post("/predict", response_model=PredictionResponse,
             summary="Identify a player from an uploaded image file")
async def predict_upload(file: UploadFile = File(...),
                         source: str = Query("upload", pattern="^(upload|camera|api)$"),
                         save: bool = Query(True, description="store the attempt in the database"),
                         session: Session = Depends(get_session)) -> PredictionResponse:
    suffix = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in (file.filename or "") else ".jpg"
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type '{suffix}'. "
                                   f"Allowed: JPG, JPEG, PNG, WEBP, BMP.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File larger than {settings.max_upload_mb:.0f} MB.")
    identifier = model_service.get_identifier()
    result = prediction_service.run_and_record(session, identifier, data, suffix=suffix,
                                               source=source, save=save)
    return PredictionResponse(**result)


@router.post("/predict/image", response_model=PredictionResponse,
             summary="Identify a player from a base64 image (browser camera captures)")
def predict_base64(payload: PredictBase64Request,
                   session: Session = Depends(get_session)) -> PredictionResponse:
    raw = payload.image_base64.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"Image larger than {settings.max_upload_mb:.0f} MB.")

    identifier = model_service.get_identifier()
    result = prediction_service.run_and_record(session, identifier, data,
                                               suffix=".jpg",
                                               source=payload.source or "camera",
                                               save=payload.save)
    result["top_predictions"] = result.get("top_predictions", [])[:max(1, payload.top_k)]
    return PredictionResponse(**result)
