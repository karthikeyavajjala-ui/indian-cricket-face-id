"""
Turns an inference result into a persisted, auditable record.

Every attempt is stored - including rejections ("unknown") - because a system
that silently hides its failures cannot be evaluated. The detected face crop is
saved next to the upload so the dashboard can show exactly what the model saw.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.predictor import encode_thumbnail

from ..config import settings
from ..models_db import Player, Prediction
from . import model_service

log = logging.getLogger("icfid.predictions")

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def save_upload(data: bytes, suffix: str = ".jpg") -> Optional[str]:
    """Persist the uploaded frame; returns a path relative to the data dir."""
    if not settings.store_uploads:
        return None
    suffix = suffix.lower() if suffix.lower() in ALLOWED_SUFFIX else ".jpg"
    name = f"{uuid.uuid4().hex}{suffix}"
    path = settings.upload_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"uploads/{name}"


def save_face_crop(data_uri: str) -> Optional[str]:
    """Store the base64 face thumbnail as a real file for the dashboard."""
    if not data_uri or not settings.store_uploads or "," not in data_uri:
        return None
    try:
        raw = base64.b64decode(data_uri.split(",", 1)[1])
    except (binascii.Error, ValueError):
        return None
    name = f"{uuid.uuid4().hex}.jpg"
    path = settings.face_crop_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return f"uploads/faces/{name}"


def record_prediction(session: Session, result: dict, source: str = "upload",
                      image_reference: Optional[str] = None,
                      face_image: Optional[str] = None,
                      model_version: Optional[str] = None) -> Prediction:
    """Persist one inference result and return the row."""
    player_row = None
    player_folder = None
    predicted_name = None
    if result.get("status") == "identified" and result.get("player"):
        predicted_name = result["player"].get("name")
        predicted_id = result["player"].get("id")
        player_row = session.scalars(
            select(Player).where(Player.slug == predicted_id)).first() if predicted_id else None
        player_folder = player_row.folder if player_row else None
    elif result.get("top_predictions"):
        predicted_name = result["top_predictions"][0].get("display_name")
        player_folder = result["top_predictions"][0].get("class_name")

    row = Prediction(
        player_id=player_row.id if player_row else None,
        predicted_folder=player_folder,
        predicted_name=predicted_name,
        status=result.get("status", "error"),
        reason=result.get("reason"),
        confidence=result.get("confidence"),
        margin=result.get("margin"),
        similarity=result.get("similarity"),
        top_predictions=json.dumps(result.get("top_predictions", [])),
        faces_detected=int(result.get("faces_detected") or 0),
        source=source,
        image_reference=f"/{image_reference}" if image_reference else None,
        face_image=f"/{face_image}" if face_image else None,
        model_version=model_version or model_service.model_version_of(
            model_service.current_bundle()),
        latency_ms=(result.get("timings_ms") or {}).get("total"),
    )
    session.add(row)
    session.flush()
    return row


def run_and_record(session: Session, identifier, image_bytes: bytes,
                   suffix: str = ".jpg", source: str = "upload",
                   save: bool = True) -> dict:
    """Identify + optionally persist. Returns the API payload (with prediction_id)."""
    result = identifier.identify_image_bytes(image_bytes, top_k=identifier.cfg.top_k)
    # Nothing to audit when there is no model or the bytes were not an image.
    if not save or result.get("status") in ("invalid_image", "model_missing"):
        return result

    image_reference = save_upload(image_bytes, suffix) if result.get("status") != "no_face" else None
    face_image = save_face_crop((result.get("detected_face") or {}).get("image"))
    try:
        row = record_prediction(session, result, source=source,
                                image_reference=image_reference, face_image=face_image)
        session.commit()
        result["prediction_id"] = row.id
    except Exception:                                  # pragma: no cover
        session.rollback()
        log.exception("Could not persist prediction")
    return result
