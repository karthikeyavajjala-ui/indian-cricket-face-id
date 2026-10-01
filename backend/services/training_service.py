"""
Background training jobs (POST /api/train).

Training runs in a worker thread inside the API process so the demo needs no
Celery/Redis. The job row in ``training_runs`` is the source of truth, so
progress survives page reloads. Only one job may run at a time.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml import config as ml_config
from ml.model import save_model, train_model
from ml.pipeline import build_feature_matrix

from ..config import settings
from ..models_db import TrainingRun, utcnow
from . import model_service

log = logging.getLogger("icfid.training")

_lock = threading.Lock()
_active_job: Optional[str] = None


# --------------------------------------------------------------------------
def is_training() -> bool:
    return _active_job is not None


def active_job_id() -> Optional[str]:
    return _active_job


def resolve_dataset(requested: str) -> Path:
    """Map an API string to a dataset directory inside the project (no traversal)."""
    key = (requested or "dataset").strip()
    mapping = {
        "dataset": settings.dataset_dir,
        "demo": settings.demo_dataset_dir,
        "dataset_demo": settings.demo_dataset_dir,
        "default": settings.dataset_dir,
    }
    if key in mapping:
        return mapping[key]
    candidate = Path(key).expanduser()
    if not candidate.is_absolute():
        candidate = (settings.project_root / candidate)
    candidate = candidate.resolve()
    if candidate.exists() and not str(candidate).startswith(str(settings.project_root)):
        raise ValueError("Absolute dataset paths must live inside the project directory.")
    return candidate


def create_job(session: Session, dataset: Path, options: dict) -> TrainingRun:
    job = TrainingRun(
        job_id=uuid.uuid4().hex[:12],
        status="queued",
        stage="queued",
        progress=0.0,
        dataset_path=str(dataset),
        options_json=json.dumps(options),
    )
    session.add(job)
    session.commit()
    return job


def start_job(job_id: str) -> bool:
    """Launch the worker thread for an existing job row."""
    global _active_job
    with _lock:
        if _active_job is not None:
            return False
        _active_job = job_id
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True,
                              name=f"train-{job_id}")
    thread.start()
    return True


class JobWriter:
    """
    Single-session writer for one job.

    The worker used to open a session per log line, which fought with API
    requests over SQLite's write lock. One persistent session with throttled
    commits keeps the job row fresh without blocking readers.
    """

    def __init__(self, job_id: str):
        from ..database import SessionLocal
        self.job_id = job_id
        self.session = SessionLocal()
        self.job = self.session.scalars(
            select(TrainingRun).where(TrainingRun.job_id == job_id)).first()
        self._last_commit = 0.0
        self._dirty = False

    def update(self, *, stage: str | None = None, progress: float | None = None,
               status: str | None = None, error: str | None = None,
               finished: bool = False, log: str | None = None,
               model_version_id: int | None = None, force: bool = False) -> None:
        import time as _time
        if self.job is None:
            return
        if stage is not None:
            self.job.stage = stage[:120]
        if progress is not None:
            self.job.progress = round(float(progress), 1)
        if status is not None:
            self.job.status = status
        if error is not None:
            self.job.error = error
        if model_version_id is not None:
            self.job.model_version_id = model_version_id
        if finished:
            self.job.finished_at = utcnow()
        if log:
            self.job.log = ((self.job.log or "") + log.rstrip() + "\n")[-20000:]
        self._dirty = True
        now = _time.monotonic()
        if force or now - self._last_commit > 1.5:
            self.commit()

    def commit(self) -> None:
        if not self._dirty:
            return
        import time as _time
        try:
            self.session.add(self.job)
            self.session.commit()
            self._dirty = False
            self._last_commit = _time.monotonic()
        except Exception:                                   # pragma: no cover
            self.session.rollback()
            log.warning("Could not persist training progress (continuing)", exc_info=True)

    def close(self) -> None:
        self.commit()
        self.session.close()


def _run_job(job_id: str) -> None:
    """Worker thread body: build features, train, save, register, activate."""
    global _active_job
    from ..database import session_scope
    from .player_service import sync_dataset, sync_supported_flags

    writer = JobWriter(job_id)
    try:
        if writer.job is None:
            return
        options = json.loads(writer.job.options_json or "{}")
        dataset_dir = Path(writer.job.dataset_path)

        writer.update(status="running", stage="scanning dataset", progress=2.0,
                      log=f"dataset: {dataset_dir}\noptions: {json.dumps(options)}",
                      force=True)
        _set_started(job_id)

        cfg = ml_config.PipelineConfig(
            detector=options.get("detector") or ml_config.PipelineConfig().detector,
            feature_extractor=options.get("extractor") or "auto",
            classifier=options.get("classifier") or "svc",
            augment_per_image=int(options.get("augment", 6)),
            pca_components=int(options.get("pca_components", 0)),
            test_size=float(options.get("test_size", 0.25)),
            seed=int(options.get("seed", 2026)),
            confidence_threshold=float(options.get("confidence_threshold",
                                                  settings.confidence_threshold)),
            min_margin=float(options.get("min_margin", settings.min_margin)),
        )

        def progress(message: str, percent: float) -> None:
            mapped = 5.0 + min(percent, 100.0) * 0.5          # 5% .. 55%
            writer.update(stage=message, progress=mapped,
                          log=f"[{percent:5.1f}%] {message}")

        writer.update(stage="extracting features", progress=5.0, force=True)
        build = build_feature_matrix(dataset_dir, cfg, progress=progress)
        writer.update(stage="training classifier", progress=58.0, force=True,
                      log=(f"images scanned: {build.stats['images_seen']}, "
                           f"frames encoded: {build.stats['faces_used']}, "
                           f"classes: {len(build.classes)}, dim: {build.X.shape[1]}"))
        writer.update(progress=60.0, stage="training classifier")

        def train_progress(message: str, percent: float) -> None:
            if percent >= 80:                                  # cross-validation stage
                writer.update(stage="cross-validating", progress=58 + percent * 0.3)

        bundle = train_model(build, cfg, dataset_dir=dataset_dir,
                            progress=train_progress)

        is_demo = dataset_dir.resolve() == settings.demo_dataset_dir.resolve()
        out_path = settings.demo_model_path if is_demo else settings.model_path
        writer.update(stage="saving model", progress=93.0, force=True)
        path = save_model(bundle, out_path)

        ev = (bundle.metrics or {}).get("evaluation", {})
        writer.update(stage="recording model version", progress=96.0, force=True,
                      log=(f"saved model: {path}\naccuracy: {ev.get('accuracy')} | "
                           f"f1_macro: {ev.get('f1_macro')} | classes: {bundle.n_classes}"))

        with session_scope() as session:
            version_row = model_service.register_model_version(
                session, bundle, path, str(dataset_dir),
                activate=options.get("activate", True))
            session.flush()
            version_id = version_row.id

        if options.get("activate", True):
            model_service.set_active_model(path)
            model_service.reload_identifier()

        with session_scope() as session:
            sync_dataset(session)
            sync_supported_flags(session, bundle.classes)

        writer.update(status="succeeded", stage="done", progress=100.0,
                      model_version_id=version_id, finished=True, force=True,
                      log="training finished")
        log.info("Training job %s finished (%s)", job_id, path)

    except Exception as exc:                                   # pragma: no cover
        log.exception("Training job %s failed", job_id)
        writer.update(status="failed", stage="failed", error=str(exc), finished=True,
                      force=True,
                      log="ERROR: " + str(exc) + "\n" + traceback.format_exc()[-3000:])
    finally:
        writer.close()
        with _lock:
            _active_job = None


def _set_started(job_id: str) -> None:
    from ..database import session_scope
    with session_scope() as session:
        job = session.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).first()
        if job is not None:
            job.started_at = utcnow()
            session.add(job)


def job_payload(session: Session, job_id: str) -> Optional[dict]:
    job = session.scalars(select(TrainingRun).where(TrainingRun.job_id == job_id)).first()
    if not job:
        return None
    result = None
    if job.model_version_id:
        from ..models_db import ModelVersion
        version = session.get(ModelVersion, job.model_version_id)
        if version:
            result = {
                "model_version": version.version, "model_path": version.model_path,
                "accuracy": version.accuracy, "f1_macro": version.f1_macro,
                "cross_val_accuracy": version.cross_val_accuracy,
                "n_classes": version.n_classes,
                "training_samples": version.training_samples,
                "validation_samples": version.validation_samples,
            }
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "dataset": job.dataset_path,
        "error": job.error,
        "log": job.log,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "result": result,
    }


def list_jobs(session: Session, limit: int = 10) -> list[dict]:
    rows = session.scalars(
        select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(limit)
    ).all()
    return [{
        "job_id": r.job_id, "status": r.status, "stage": r.stage,
        "progress": r.progress, "dataset": r.dataset_path,
        "error": r.error,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    } for r in rows]
