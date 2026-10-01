"""Application settings (env-driven, with sane local defaults)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:                                    # optional: load .env if python-dotenv exists
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:                       # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


class Settings:
    """All runtime configuration in one place."""

    app_name = "INDIAN CRICKET FACE ID"
    version = "1.0.0"
    project_root = PROJECT_ROOT

    # --- storage ---------------------------------------------------------
    data_dir = _path("IC_DATA_DIR", PROJECT_ROOT / "data")
    database_url = os.getenv("IC_DATABASE_URL",
                             f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}")
    upload_dir = _path("IC_UPLOAD_DIR", PROJECT_ROOT / "data" / "uploads")
    face_crop_dir = _path("IC_FACE_CROP_DIR", PROJECT_ROOT / "data" / "uploads" / "faces")

    # --- model -----------------------------------------------------------
    model_path = _path("IC_MODEL_PATH", PROJECT_ROOT / "models" / "model.pkl")
    demo_model_path = _path("IC_DEMO_MODEL_PATH", PROJECT_ROOT / "models" / "demo_model.pkl")
    confidence_threshold = float(os.getenv("IC_CONFIDENCE_THRESHOLD", "0.55"))
    min_margin = float(os.getenv("IC_MIN_MARGIN", "0.12"))

    # --- datasets --------------------------------------------------------
    dataset_dir = _path("IC_DATASET_DIR", PROJECT_ROOT / "dataset")
    demo_dataset_dir = _path("IC_DEMO_DATASET_DIR", PROJECT_ROOT / "dataset_demo")
    demo_enabled = _bool("IC_DEMO_ENABLED", True)
    auto_bootstrap = _bool("IC_AUTO_BOOTSTRAP", True)
    max_upload_mb = float(os.getenv("IC_MAX_UPLOAD_MB", "12"))

    # --- training endpoint protection -----------------------------------
    training_api_key = os.getenv("IC_TRAIN_API_KEY", "change-me-in-production")
    admin_api_key = os.getenv("IC_ADMIN_API_KEY", training_api_key)
    allow_unsigned_training = _bool("IC_ALLOW_UNSIGNED_TRAINING", True)

    # --- misc ------------------------------------------------------------
    cors_origins = [o.strip() for o in os.getenv("IC_CORS_ORIGINS", "*").split(",") if o.strip()]
    log_level = os.getenv("IC_LOG_LEVEL", "INFO")
    store_uploads = _bool("IC_STORE_UPLOADS", True)
    dashboard_recent_limit = int(os.getenv("IC_DASHBOARD_RECENT_LIMIT", "12"))

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.face_crop_dir,
                     self.model_path.parent):
            Path(path).mkdir(parents=True, exist_ok=True)

    def as_public_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "demo_model_path": str(self.demo_model_path),
            "dataset_dir": str(self.dataset_dir),
            "demo_dataset_dir": str(self.demo_dataset_dir),
            "demo_enabled": self.demo_enabled,
            "confidence_threshold": self.confidence_threshold,
            "min_margin": self.min_margin,
            "training_protected": not self.allow_unsigned_training
            or self.training_api_key != "change-me-in-production",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
