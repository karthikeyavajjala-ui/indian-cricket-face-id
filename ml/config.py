"""
Central configuration for the INDIAN CRICKET FACE ID machine-learning pipeline.

Every path/value can be overridden with an environment variable so the same
codebase can run locally, in CI, or against a different dataset mount.

Nothing in this module downloads data.  Copyrighted player photographs are
*never* fetched automatically - see scripts/prepare_dataset.py and
docs/DATASET.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# ml/config.py -> ml/ -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


DATASET_DIR = _env_path("IC_DATASET_DIR", PROJECT_ROOT / "dataset")
DEMO_DATASET_DIR = _env_path("IC_DEMO_DATASET_DIR", PROJECT_ROOT / "dataset_demo")
MODELS_DIR = _env_path("IC_MODELS_DIR", PROJECT_ROOT / "models")
DATA_DIR = _env_path("IC_DATA_DIR", PROJECT_ROOT / "data")
UPLOAD_DIR = _env_path("IC_UPLOAD_DIR", DATA_DIR / "uploads")
CACHE_DIR = _env_path("IC_CACHE_DIR", DATA_DIR / "cache")

DEFAULT_MODEL_PATH = _env_path("IC_MODEL_PATH", MODELS_DIR / "model.pkl")
DEMO_MODEL_PATH = MODELS_DIR / "demo_model.pkl"

# Optional OpenCV-Zoo ONNX models (see scripts/fetch_models.py)
YUNET_MODEL_PATH = MODELS_DIR / "face_detector" / "face_detection_yunet_2023mar.onnx"
YUNET_DYN_MODEL_PATH = MODELS_DIR / "face_detector" / "face_detection_yunet_2026may.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_models" / "face_recognition_sface_2021dec.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}
MIN_IMAGE_SIDE = 64          # px - smaller files are rejected during validation
MIN_FACE_SIDE = 40           # px - smaller detections are ignored
BLUR_VARIANCE_THRESHOLD = 12.0   # Laplacian variance below this = "very blurry"


# --------------------------------------------------------------------------
# Pipeline configuration
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Configuration persisted inside model.pkl and reused at inference time."""

    # --- detection -------------------------------------------------------
    detector: str = os.getenv("IC_DETECTOR", "auto")   # auto | yunet | haar
    align_size: int = 112                               # aligned face side (px)
    max_faces: int = 10

    # --- feature extraction ---------------------------------------------
    # auto -> sface if the ONNX model is available, otherwise hog_lbp
    feature_extractor: str = os.getenv("IC_EXTRACTOR", "auto")
    # free-form extractor kwargs (e.g. {"pca_components": 200})
    extractor_params: dict = field(default_factory=dict)

    # --- classifier ------------------------------------------------------
    classifier: str = os.getenv("IC_CLASSIFIER", "svc")   # svc | logistic | rf | knn
    classifier_params: dict = field(default_factory=dict)
    pca_components: int = int(os.getenv("IC_PCA_COMPONENTS", "0"))  # 0 = no PCA

    # --- decision policy (responsible-AI guards) -------------------------
    # Below `confidence_threshold` the system answers "Unknown", never a guess.
    confidence_threshold: float = float(os.getenv("IC_CONFIDENCE_THRESHOLD", "0.55"))
    # If the top-2 classes are closer than this margin the answer is ambiguous.
    min_margin: float = float(os.getenv("IC_MIN_MARGIN", "0.12"))
    # Minimum cosine similarity for the deep (embedding) engine to accept a match.
    min_similarity: float = float(os.getenv("IC_MIN_SIMILARITY", "0.28"))
    top_k: int = 3

    # --- training --------------------------------------------------------
    test_size: float = float(os.getenv("IC_TEST_SIZE", "0.25"))
    seed: int = int(os.getenv("IC_SEED", "2026"))
    augment_per_image: int = int(os.getenv("IC_AUGMENT_PER_IMAGE", "6"))
    cross_validate: bool = os.getenv("IC_CROSS_VALIDATE", "1") not in ("0", "false", "False")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    for path in (DATASET_DIR, MODELS_DIR, DATA_DIR, UPLOAD_DIR, CACHE_DIR,
                 MODELS_DIR / "face_detector", MODELS_DIR / "face_models"):
        Path(path).mkdir(parents=True, exist_ok=True)


def resolve_dataset_dir(prefer_demo: bool = False) -> Path:
    """Return the dataset directory to use (demo only when explicitly asked)."""
    if prefer_demo:
        return DEMO_DATASET_DIR
    return DATASET_DIR


def dataset_is_configured(dataset_dir: Path | None = None) -> bool:
    """True when at least 2 class folders contain at least one usable image."""
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    if not dataset_dir.is_dir():
        return False
    classes = 0
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir() if p.is_file()):
            classes += 1
    return classes >= 2


def human_name_from_folder(folder_name: str) -> str:
    """`Virat_Kohli` -> `Virat Kohli`"""
    return folder_name.replace("_", " ").strip()


def dataset_summary_safe(dataset_dir: Path | None = None) -> dict:
    """Lightweight dataset status for API responses (never raises)."""
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    counts: dict = {}
    try:
        if dataset_dir.is_dir():
            for child in sorted(dataset_dir.iterdir()):
                if child.is_dir() and not child.name.startswith((".", "_")):
                    n = sum(1 for p in child.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                    if n:
                        counts[child.name] = n
    except OSError:
        pass
    return {
        "path": str(dataset_dir),
        "configured": len(counts) >= 2,
        "num_classes": len(counts),
        "images": counts,
        "total_images": sum(counts.values()),
        "hint": ("Place legally usable images in dataset/<Player_Name>/ folders, then "
                 "run: python scripts/prepare_dataset.py && python -m ml.train"),
    }
