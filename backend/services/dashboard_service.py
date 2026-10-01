"""Dashboard aggregates + recent-prediction feed and chart series."""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter, defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ml import config as ml_config

from ..config import settings
from ..models_db import Player, Prediction, TrainingRun
from . import model_service


def _start_of_today() -> dt.datetime:
    now = dt.datetime.now(dt.timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def prediction_row(row: Prediction) -> dict:
    return {
        "id": row.id,
        "player_id": row.predicted_folder,
        "predicted_name": row.predicted_name,
        "status": row.status,
        "reason": row.reason,
        "confidence": row.confidence,
        "source": row.source,
        "image_reference": row.image_reference,
        "face_image": row.face_image,
        "model_version": row.model_version,
        "latency_ms": row.latency_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def recent_predictions(session: Session, limit: int = 12) -> list[dict]:
    rows = session.scalars(
        select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)
    ).all()
    return [prediction_row(r) for r in rows]


def summary(session: Session) -> dict:
    dataset = ml_config.dataset_summary_safe(settings.dataset_dir)
    demo_dataset = ml_config.dataset_summary_safe(settings.demo_dataset_dir)
    active_dataset = dataset if dataset["configured"] else demo_dataset

    total_players = session.scalar(select(func.count(Player.id))) or 0
    supported_players = session.scalar(
        select(func.count(Player.id)).where(Player.supported.is_(True))) or 0

    total_predictions = session.scalar(select(func.count(Prediction.id))) or 0
    today = _start_of_today()
    today_rows = session.scalars(
        select(Prediction).where(Prediction.created_at >= today)).all()
    predictions_today = len(today_rows)
    identified_today = sum(1 for r in today_rows if r.status == "identified")
    unknown_today = sum(1 for r in today_rows if r.status == "unknown")

    confidences = [r.confidence for r in today_rows if r.confidence is not None]
    latencies = [r.latency_ms for r in today_rows if r.latency_ms]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    # ---- charts ---------------------------------------------------------
    all_rows = session.scalars(select(Prediction)).all()
    by_player = Counter(r.predicted_name or "Unknown" for r in all_rows)
    prediction_distribution = [
        {"label": name, "value": count} for name, count in by_player.most_common(8)
    ]

    buckets = [0, 0, 0, 0, 0]      # <0.4, .4-.55, .55-.7, .7-.85, .85-1
    for row in all_rows:
        if row.confidence is None:
            continue
        c = row.confidence
        idx = 0 if c < 0.4 else 1 if c < 0.55 else 2 if c < 0.7 else 3 if c < 0.85 else 4
        buckets[idx] += 1
    confidence_distribution = [
        {"label": label, "value": value} for label, value in zip(
            ["<40%", "40-55%", "55-70%", "70-85%", "85-100%"], buckets)
    ]

    status_distribution = [
        {"label": "Identified", "value": sum(1 for r in all_rows if r.status == "identified")},
        {"label": "Unknown (low confidence)",
         "value": sum(1 for r in all_rows if r.status == "unknown")},
        {"label": "No face", "value": sum(1 for r in all_rows if r.status == "no_face")},
    ]

    # ---- 14-day trend ---------------------------------------------------
    days = [(_start_of_today() - dt.timedelta(days=i)) for i in range(13, -1, -1)]
    trend = []
    for day in days:
        nxt = day + dt.timedelta(days=1)
        rows = [r for r in all_rows
                if r.created_at and day <= r.created_at < nxt]
        trend.append({
            "label": day.strftime("%d %b"),
            "total": len(rows),
            "identified": sum(1 for r in rows if r.status == "identified"),
            "unknown": sum(1 for r in rows if r.status == "unknown"),
        })

    # ---- model performance ---------------------------------------------
    bundle = model_service.current_bundle()
    metrics = (bundle.metrics if bundle else {}) or {}
    ev = metrics.get("evaluation", {}) or {}
    cv = metrics.get("cross_validation", {}) or {}
    per_class = ev.get("per_class", [])
    variants = None
    if bundle is not None:
        variants = [c for c in bundle.classes]
    performance = {
        "ready": bundle is not None,
        "accuracy": ev.get("accuracy"),
        "precision_macro": ev.get("precision_macro"),
        "recall_macro": ev.get("recall_macro"),
        "f1_macro": ev.get("f1_macro"),
        "cross_val_accuracy": cv.get("accuracy_mean") if cv.get("available") else None,
        "training_samples": metrics.get("training_samples"),
        "validation_samples": metrics.get("validation_samples"),
        "classes": variants or [],
        "per_class": per_class,
        "confusion": {"labels": ev.get("confusion_matrix_labels", []),
                      "matrix": ev.get("confusion_matrix", [])},
    }

    last_job = session.scalars(
        select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(1)).first()

    accuracy_note = (
        model_service.DEMO_CAVEAT if model_service.resolve_model_path()[1] == "demo" else
        "Held-out split of your own images; augmented variants of the same photo can "
        "appear on both sides of the split, so this overstates accuracy on unseen photos."
        if performance["accuracy"] is not None else
        "No model trained yet - accuracy is not available."
    )

    return {
        "total_players": total_players,
        "supported_players": supported_players,
        "total_images": active_dataset.get("total_images", 0),
        "dataset_configured": active_dataset.get("configured", False),
        "model_accuracy": performance["accuracy"],
        "model_accuracy_note": accuracy_note,
        "predictions_total": total_predictions,
        "predictions_today": predictions_today,
        "identified_today": identified_today,
        "unknown_today": unknown_today,
        "avg_confidence": avg_conf,
        "avg_latency_ms": avg_latency,
        "recent": recent_predictions(session, settings.dashboard_recent_limit),
        "confidence_distribution": confidence_distribution,
        "prediction_distribution": prediction_distribution,
        "status_distribution": status_distribution,
        "daily_trend": trend,
        "performance": performance,
        "model": {
            "ready": bundle is not None,
            "mode": model_service.resolve_model_path()[1],
            "version": (bundle.metadata or {}).get("model_version") if bundle else None,
            "trained_at": metrics.get("trained_at"),
            "classifier": (bundle.cfg or {}).get("classifier") if bundle else None,
            "feature_extractor": (bundle.cfg or {}).get("feature_extractor") if bundle else None,
            "detector": (bundle.cfg or {}).get("detector") if bundle else None,
            "confidence_threshold": (bundle.cfg or {}).get("confidence_threshold") if bundle else None,
        },
        "training_job": {
            "job_id": last_job.job_id, "status": last_job.status,
            "stage": last_job.stage, "progress": last_job.progress,
            "finished_at": last_job.finished_at.isoformat() if last_job and last_job.finished_at else None,
        } if last_job else None,
        "dataset": dataset,
        "demo_dataset": demo_dataset,
    }


def top_predictions_for(session: Session, prediction_id: int) -> list[dict]:
    row = session.get(Prediction, prediction_id)
    if not row or not row.top_predictions:
        return []
    try:
        return json.loads(row.top_predictions)
    except json.JSONDecodeError:
        return []
