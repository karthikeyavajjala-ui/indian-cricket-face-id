"""
Model lifecycle: which model file is active, how it is loaded, and how each
trained model is recorded in the ``model_versions`` table.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import config as ml_config
from ml.model import ModelBundle, load_metrics_json
from ml.predictor import FaceIdentifier, reset_identifier

from .. import catalog
from ..config import settings
from ..models_db import ModelVersion

log = logging.getLogger("icfid.model_service")

ACTIVE_POINTER = settings.data_dir / "active_model.json"

DEMO_CAVEAT = (
    "The bundled demo personas are visually very distinct AI-generated faces, so a model "
    "trained only on them scores near-perfectly by construction. Those numbers say nothing "
    "about identifying real players from real photographs."
)

TRANSPARENCY = [
    "Metrics come from a held-out split of your own dataset - they are NOT a "
    "real-world accuracy guarantee.",
    "When images per player are few, augmented variants of the same photograph can "
    "appear in both training and validation, which inflates the score.",
    "The model only knows the classes it was trained on; anyone else must be "
    "reported as Unknown, which is why a confidence threshold is enforced.",
    "Different lighting, age, angle, occlusion (helmet, cap, beard) and image "
    "quality change predictions. Treat a single result as low evidence.",
]


# --------------------------------------------------------------------------
# Active model bookkeeping
# --------------------------------------------------------------------------
def _read_pointer() -> Optional[Path]:
    if ACTIVE_POINTER.is_file():
        try:
            data = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8"))
            path = Path(data.get("path", ""))
            if path.is_file():
                return path
        except (OSError, json.JSONDecodeError):
            log.warning("Could not read %s", ACTIVE_POINTER)
    return None


def set_active_model(path: Path) -> None:
    ACTIVE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_POINTER.write_text(json.dumps({
        "path": str(Path(path).resolve()),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def resolve_model_path() -> tuple[Optional[Path], str]:
    """
    Return ``(path, mode)`` where mode is ``'trained'`` | ``'demo'`` | ``'none'``.

    Priority
    --------
    1. the pointer written by the API when a model was activated,
    2. a model trained on your own ``dataset/`` (``models/model.pkl``),
    3. the bundled demo model (fictional personas), when the demo is enabled.

    A pointer to the *demo* model is ignored as soon as a real model exists, so
    training on your own images always takes over automatically.
    """
    pointed = _read_pointer()
    if pointed is not None:
        is_demo_pointer = "demo" in pointed.name
        demo_blocked = is_demo_pointer and not settings.demo_enabled
        real_available = settings.model_path.is_file() and pointed != settings.model_path
        if not demo_blocked and not (is_demo_pointer and real_available):
            return pointed, "demo" if is_demo_pointer else "trained"
    if settings.model_path.is_file():
        return settings.model_path, "trained"
    if settings.demo_enabled and settings.demo_model_path.is_file():
        return settings.demo_model_path, "demo"
    return None, "none"


def available_models() -> list[dict]:
    pointer = _read_pointer()
    options = []
    for path, mode, label in (
        (settings.model_path, "trained", "Trained on your dataset/"),
        (settings.demo_model_path, "demo", "Bundled demo model (fictional personas)"),
    ):
        options.append({
            "path": str(path), "mode": mode, "label": label,
            "exists": path.is_file(),
            "active": bool(pointer == path or (pointer is None and resolve_model_path()[0] == path)),
        })
    return options


# --------------------------------------------------------------------------
# Identifier singleton
# --------------------------------------------------------------------------
_identifier: Optional[FaceIdentifier] = None
_identifier_path: Optional[Path] = None


def get_identifier(force_path: Optional[Path] = None) -> FaceIdentifier:
    """Lazily build (and cache) the inference engine for the active model."""
    global _identifier, _identifier_path
    path, _mode = resolve_model_path()
    if force_path is not None:
        path = force_path
    overrides = {
        "confidence_threshold": settings.confidence_threshold,
        "min_margin": settings.min_margin,
    }
    if _identifier is None or _identifier_path != path:
        _identifier = FaceIdentifier(model_path=path, cfg_overrides=overrides,
                                     player_catalog=catalog.catalog_keyed_by_slug())
        _identifier_path = path
        reset_identifier()
        log.info("Inference engine ready: %s (%s)", path, _identifier.ready)
    return _identifier


def reload_identifier() -> FaceIdentifier:
    global _identifier, _identifier_path
    _identifier = None
    _identifier_path = None
    return get_identifier()


def current_bundle() -> Optional[ModelBundle]:
    identifier = get_identifier()
    return identifier.bundle if identifier.ready else None


def model_version_of(bundle: Optional[ModelBundle], path: Optional[Path] = None) -> str:
    if bundle is None:
        return "none"
    version = (bundle.metadata or {}).get("model_version")
    if version:
        return str(version)
    if path is not None:
        return f"file:{Path(path).name}"
    return "unknown"


# --------------------------------------------------------------------------
# Status / metrics payloads
# --------------------------------------------------------------------------
def status_payload(training_in_progress: bool = False) -> dict:
    path, mode = resolve_model_path()
    identifier = get_identifier()
    engine = identifier.engine_info() if identifier.ready else {
        "ready": False,
        "error": identifier.error,
    }
    return {
        "ready": identifier.ready,
        "active_model": str(path) if path else None,
        "mode": mode,
        "error": None if identifier.ready else identifier.error,
        "engine": engine,
        "dataset": ml_config.dataset_summary_safe(settings.dataset_dir),
        "demo_dataset": ml_config.dataset_summary_safe(settings.demo_dataset_dir),
        "training_in_progress": training_in_progress,
        "viewer_note": ("No model is trained yet. Add images to dataset/ and run "
                        "python -m ml.train, or POST /api/train."),
    }


def metrics_payload() -> dict:
    identifier = get_identifier()
    if not identifier.ready:
        return {
            "ready": False,
            "message": identifier.error or "No trained model available.",
            "transparency": TRANSPARENCY,
            "notes": [],
        }
    bundle = identifier.bundle
    metrics = bundle.metrics or {}
    ev = metrics.get("evaluation", {}) or {}
    notes = list(metrics.get("notes", []))
    if resolve_model_path()[1] == "demo":
        notes.insert(0, DEMO_CAVEAT)
    payload = {
        "ready": True,
        "model_version": (bundle.metadata or {}).get("model_version"),
        "trained_at": metrics.get("trained_at"),
        "training": {
            k: metrics.get(k) for k in
            ("training_samples", "validation_samples", "total_samples", "feature_dim",
             "n_classes", "classifier", "classifier_params", "feature_extractor",
             "detector", "training_seconds", "trained_at", "dataset", "class_samples")
            if k in metrics
        },
        "evaluation": ev,
        "cross_validation": metrics.get("cross_validation", {}),
        "per_class": ev.get("per_class", []),
        "confusion": None,
        "notes": notes,
        "transparency": TRANSPARENCY,
        "metadata": bundle.metadata,
    }
    if ev.get("confusion_matrix"):
        payload["confusion"] = {
            "labels": ev.get("confusion_matrix_labels", bundle.classes),
            "matrix": ev["confusion_matrix"],
        }
    return payload


# --------------------------------------------------------------------------
# Persisting model versions
# --------------------------------------------------------------------------
def register_model_version(session: Session, bundle: ModelBundle, model_path: Path,
                           dataset_path: Optional[str] = None,
                           activate: bool = True) -> ModelVersion:
    metrics = bundle.metrics or {}
    ev = metrics.get("evaluation", {}) or {}
    cv = metrics.get("cross_validation", {}) or {}
    trained_at = None
    try:
        trained_at = dt.datetime.fromisoformat(
            str(metrics.get("trained_at")).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass

    if activate:
        for row in session.scalars(select(ModelVersion)).all():
            row.is_active = False

    version = ModelVersion(
        version=str((bundle.metadata or {}).get("model_version")
                    or f"v{bundle.format_version}-{int(dt.datetime.now().timestamp())}"),
        model_path=str(model_path),
        dataset_path=dataset_path or (metrics.get("dataset") or {}).get("path"),
        dataset_fingerprint=(bundle.metadata or {}).get("dataset_fingerprint"),
        classifier=bundle.cfg.get("classifier"),
        feature_extractor=bundle.cfg.get("feature_extractor"),
        detector=bundle.cfg.get("detector"),
        n_classes=bundle.n_classes,
        training_samples=int(metrics.get("training_samples") or 0),
        validation_samples=int(metrics.get("validation_samples") or 0),
        accuracy=ev.get("accuracy"),
        precision_macro=ev.get("precision_macro"),
        recall_macro=ev.get("recall_macro"),
        f1_macro=ev.get("f1_macro"),
        cross_val_accuracy=cv.get("accuracy_mean") if cv.get("available") else None,
        trained_at=trained_at,
        metrics_json=json.dumps(metrics),
        is_active=activate,
    )
    session.add(version)
    session.flush()
    return version


def list_model_versions(session: Session, limit: int = 20) -> list[dict]:
    rows = session.scalars(
        select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
    ).all()
    return [{
        "id": r.id, "version": r.version, "is_active": r.is_active,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "trained_at": r.trained_at.isoformat() if r.trained_at else None,
        "classifier": r.classifier, "feature_extractor": r.feature_extractor,
        "detector": r.detector, "n_classes": r.n_classes,
        "training_samples": r.training_samples, "validation_samples": r.validation_samples,
        "accuracy": r.accuracy, "f1_macro": r.f1_macro,
        "cross_val_accuracy": r.cross_val_accuracy,
        "dataset_path": r.dataset_path, "model_path": r.model_path,
    } for r in rows]


def sync_active_versions(session: Session) -> None:
    """Ensure the DB has a row for the model currently on disk (e.g. after a CLI
    training run that the API never saw)."""
    bundle = current_bundle()
    if bundle is None:
        return
    path, _mode = resolve_model_path()
    version = model_version_of(bundle, path)
    existing = session.scalars(
        select(ModelVersion).where(ModelVersion.version == version)
    ).first()
    if existing:
        return
    # A metrics.json next to the model means it was produced by ml.train
    if path and not load_metrics_json(path):
        pass
    register_model_version(session, bundle, path or settings.model_path,
                           activate=True)
    session.commit()
    log.info("Registered externally trained model %s in the database", version)
