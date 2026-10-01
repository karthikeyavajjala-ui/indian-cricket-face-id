"""
INDIAN CRICKET FACE ID - FastAPI application.

Run:
    python -m uvicorn backend.main:app --reload --port 8000
or
    python run.py

Routes
------
  /                       premium dashboard UI (static frontend)
  /docs                   interactive OpenAPI documentation
  /api/health             liveness + subsystem snapshot
  /api/players            catalog (facts + dataset + model info)
  /api/predict            identify from an uploaded file
  /api/predict/image      identify from a base64 camera frame
  /api/model/status       engine + dataset status
  /api/model/metrics      honest evaluation report
  /api/model/reload       reload model.pkl from disk
  /api/train              start a training job (protected)
  /api/dashboard/summary  cards + charts + recent attempts
  /uploads/*              stored uploads + detected face crops
  /media/{source}/*       read-only dataset images for profile galleries
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db, session_scope
from .routers import dashboard, health, model, players, predict, train
from .services import model_service, player_service, training_service

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("icfid")

FRONTEND_DIR = settings.project_root / "frontend"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup / shutdown: DB, catalog seed, dataset inventory, demo bootstrap."""
    settings.ensure_dirs()
    init_db()
    with session_scope() as session:
        player_service.sync_catalog(session)
        player_service.sync_dataset(session)
    log.info("%s v%s ready · db=%s", settings.app_name, settings.version,
             settings.database_url)
    log.info("UI: http://localhost:8000   API docs: http://localhost:8000/docs")
    bootstrap_demo_if_needed()
    yield
    log.info("shutting down")


app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.version,
    lifespan=lifespan,
    description=(
        "Facial identification demo for Indian cricket players. The ML pipeline "
        "(OpenCV face detection + pluggable feature extraction + scikit-learn "
        "classification) trains only on images you provide in `dataset/`.\n\n"
        "**Responsible use:** this is a technology demonstration. It must not be used "
        "for surveillance, law enforcement, access control or high-impact decisions. "
        "It recognises only the people present in its training dataset."
    ),
    contact={"name": "INDIAN CRICKET FACE ID"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(players.router)
app.include_router(predict.router)
app.include_router(model.router)
app.include_router(train.router)
app.include_router(dashboard.router)


# --------------------------------------------------------------------------
# Static assets
# --------------------------------------------------------------------------
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

settings.ensure_dirs()
app.mount("/uploads", StaticFiles(directory=str(settings.data_dir / "uploads")),
          name="uploads")


def _safe_media_path(source: str, relative: str) -> Path | None:
    root = (settings.demo_dataset_dir if source == "demo" else settings.dataset_dir).resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents and candidate != root:
        return None
    return candidate if candidate.is_file() else None


@app.get("/media/{source}/{relative:path}", include_in_schema=False)
def dataset_media(source: str, relative: str):
    """Serve dataset images (read-only) for profile galleries."""
    if source not in ("local", "demo"):
        return JSONResponse({"detail": "unknown media source"}, status_code=404)
    path = _safe_media_path(source, relative)
    if path is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(str(path))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse({"detail": "frontend/ not found"}, status_code=404)
    return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    path = FRONTEND_DIR / "assets" / "img" / "favicon.svg"
    if path.is_file():
        return FileResponse(str(path), media_type="image/svg+xml")
    return JSONResponse({"detail": "no favicon"}, status_code=404)


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    """Deep links (e.g. /players/virat-kohli) fall through to the SPA shell."""
    if path.startswith(("api/", "uploads/", "media/", "static/", "docs", "openapi")):
        return JSONResponse({"detail": "not found"}, status_code=404)
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})
    return JSONResponse({"detail": "frontend/ not found"}, status_code=404)


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
_BOOTSTRAP_LOCK = threading.Lock()


def bootstrap_demo_if_needed() -> None:
    """
    First run experience:

    * no model + no personal dataset + demo enabled  -> generate the fictional
      demo personas' dataset and train a demo model in the background;
    * a model exists                                 -> nothing happens.

    Never invents predictions: it only trains on real files on disk.
    """
    path, mode = model_service.resolve_model_path()
    if path is not None:
        log.info("Active model: %s (%s)", path, mode)
        return
    if not (settings.auto_bootstrap and settings.demo_enabled):
        log.warning("No trained model. Add images to %s and run: python -m ml.train",
                    settings.dataset_dir)
        return

    real_dataset_ready = (settings.dataset_dir / "README.md").exists() and any(
        child.is_dir() and any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                               for p in child.rglob("*") if p.is_file())
        for child in settings.dataset_dir.iterdir()
        if child.is_dir() and not child.name.startswith((".", "_"))) \
        if settings.dataset_dir.is_dir() else False
    if real_dataset_ready:
        log.warning("Your dataset/ has images but no model is trained. "
                    "Run: python -m ml.train --dataset dataset")
        return

    demo_sources = settings.demo_dataset_dir / "_sources"
    has_demo_images = any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                          for p in settings.demo_dataset_dir.rglob("*")
                          if p.is_file()) if settings.demo_dataset_dir.is_dir() else False

    def _worker() -> None:
        time.sleep(1.0)                        # let the API finish starting
        try:
            if not has_demo_images or not (settings.demo_dataset_dir / "Demo_Persona_A").is_dir():
                log.info("Bootstrapping the demo dataset from dataset_demo/_sources …")
                from scripts.make_demo_dataset import build_dataset
                if not demo_sources.is_dir():
                    log.warning("Demo sources missing - skipping bootstrap.")
                    return
                build_dataset(demo_sources, settings.demo_dataset_dir, variants=9, seed=7)
            log.info("Training the demo model (fictional, AI-generated personas) …")
            with session_scope() as session:
                dataset = settings.demo_dataset_dir
                job = training_service.create_job(session, dataset, {
                    "dataset": "dataset_demo", "classifier": "svc",
                    "extractor": "auto", "detector": "auto", "augment": 4,
                    "activate": True, "auto_bootstrap": True,
                })
                job_id = job.job_id
            training_service.start_job(job_id)
            log.info("Demo training job %s started - see /api/train/status/%s",
                     job_id, job_id)
        except Exception:                                     # pragma: no cover
            log.exception("Demo bootstrap failed")

    with _BOOTSTRAP_LOCK:
        threading.Thread(target=_worker, daemon=True, name="demo-bootstrap").start()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error", "error": str(exc)})
