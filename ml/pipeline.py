"""
Dataset scanning, validation and feature-matrix construction.

    dataset/
        Virat_Kohli/ * .jpg .png .webp ...
        Rohit_Sharma/
        Jasprit_Bumrah/

The folder name is the class label. Nothing is downloaded by this module; the
developer supplies legally usable images (see docs/DATASET.md).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .face import FaceBox, align_face, make_detector
from .features import FeatureExtractor, get_extractor

log = logging.getLogger("icfid.pipeline")


# --------------------------------------------------------------------------
# Scanning / validation
# --------------------------------------------------------------------------
@dataclass
class DatasetSample:
    path: Path
    label: str            # class folder name, e.g. "Virat_Kohli"
    shape: Tuple[int, int] = (0, 0)
    ok: bool = True
    reason: str = ""
    faces_found: int = 0
    blur_variance: float = 0.0


def scan_dataset(dataset_dir: Optional[Path] = None,
                 verbose: bool = False) -> Tuple[List[DatasetSample], Dict[str, int]]:
    """
    Walk the dataset and validate every image file.

    Returns ``(samples, class_counts)``. Validation never raises: broken files are
    returned with ``ok=False`` and a human-readable ``reason``.
    """
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    samples: List[DatasetSample] = []
    if not dataset_dir.is_dir():
        return samples, {}

    for class_dir in sorted(p for p in dataset_dir.iterdir()
                            if p.is_dir() and not p.name.startswith((".", "_"))):
        for img_path in sorted(class_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in config.IMAGE_EXTENSIONS:
                continue
            sample = DatasetSample(path=img_path, label=class_dir.name)
            img = cv2.imread(str(img_path))
            if img is None:
                sample.ok, sample.reason = False, "unreadable or unsupported image"
            else:
                h, w = img.shape[:2]
                sample.shape = (w, h)
                if min(h, w) < config.MIN_IMAGE_SIDE:
                    sample.ok = False
                    sample.reason = f"image too small ({w}x{h})"
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sample.blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sample.blur_variance < 1.5:
                        sample.ok = False
                        sample.reason = "image is blank or unreadable"
            samples.append(sample)
            if verbose:
                status = "OK " if sample.ok else "BAD"
                print(f"  [{status}] {sample.path.relative_to(dataset_dir)} "
                      f"{sample.shape[0]}x{sample.shape[1]} {sample.reason}")

    class_counts: Dict[str, int] = {}
    for s in samples:
        if s.ok:
            class_counts[s.label] = class_counts.get(s.label, 0) + 1
    return samples, class_counts


def dataset_summary(dataset_dir: Optional[Path] = None) -> dict:
    samples, counts = scan_dataset(dataset_dir)
    return {
        "dataset_dir": str(dataset_dir or config.DATASET_DIR),
        "configured": config.dataset_is_configured(dataset_dir),
        "total_files": len(samples),
        "valid_images": sum(1 for s in samples if s.ok),
        "invalid_images": sum(1 for s in samples if not s.ok),
        "classes": counts,
        "num_classes": len(counts),
        "problems": [{"file": s.path.name, "reason": s.reason}
                     for s in samples if not s.ok][:25],
    }


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def augment_face(face_bgr: np.ndarray, rng: random.Random, allow_flip: bool = True,
                 index: int = 0) -> np.ndarray:
    """
    Mild, label-preserving augmentation of an *aligned* face patch.

    Kept deliberately conservative: this is face identification, so anything that
    changes identity cues (large rotations, extreme warps) is avoided.
    """
    h, w = face_bgr.shape[:2]
    img = face_bgr

    if allow_flip and index % 4 == 3:
        img = cv2.flip(img, 1)

    # small rotation + scale + translation
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.94, 1.07)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-0.035, 0.035) * w
    M[1, 2] += rng.uniform(-0.035, 0.035) * h
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    # photometric jitter: brightness / contrast, colour temperature
    alpha = rng.uniform(0.88, 1.14)          # contrast
    beta = rng.uniform(-18, 18)              # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < 0.4:
        b, g, r = cv2.split(img.astype(np.float32))
        b *= rng.uniform(0.96, 1.04)
        r *= rng.uniform(0.96, 1.04)
        img = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

    # occasional soft blur / noise to mimic broadcast compression
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.20:
        noise = np.random.normal(0, rng.uniform(2.0, 6.0), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------
@dataclass
class BuildResult:
    X: np.ndarray
    y: np.ndarray
    classes: List[str]
    records: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def build_feature_matrix(
    dataset_dir: Optional[Path] = None,
    cfg: Optional[config.PipelineConfig] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> BuildResult:
    """
    Full offline pipeline: load -> validate -> detect -> align -> augment -> encode.
    """
    cfg = cfg or config.PipelineConfig()
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    detector = make_detector(cfg.detector)
    extractor = get_extractor(cfg.feature_extractor, **cfg.extractor_params)
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    classes = sorted(d for d in (p.name for p in dataset_dir.iterdir() if p.is_dir())
                     if not d.startswith((".", "_"))) if dataset_dir.is_dir() else []
    classes = [c for c in classes
               if any(p.suffix.lower() in config.IMAGE_EXTENSIONS
                      for p in (dataset_dir / c).rglob("*") if p.is_file())]
    if len(classes) < 2:
        raise ValueError(
            f"Dataset at {dataset_dir} needs at least 2 player folders with images. "
            f"Found: {classes or 'nothing'}. See docs/DATASET.md."
        )
    label_to_idx = {name: i for i, name in enumerate(classes)}

    allow_flip = extractor.name != "sface"   # embeddings are not flip-invariant
    stats = {
        "detector": detector.name,
        "feature_extractor": extractor.name,
        "images_seen": 0, "images_valid": 0, "images_unreadable": 0,
        "images_without_face": 0, "faces_used": 0, "augmented": 0,
        "skipped": [],
    }

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    records: List[dict] = []

    samples, _ = scan_dataset(dataset_dir)
    samples = [s for s in samples if s.ok]
    total = max(1, len(samples))

    for idx, sample in enumerate(samples):
        stats["images_seen"] += 1
        if progress and idx % 5 == 0:
            progress(f"Processing {sample.path.name}",
                     100.0 * idx / total)
        img = cv2.imread(str(sample.path))
        if img is None:
            stats["images_unreadable"] += 1
            stats["skipped"].append({"file": str(sample.path), "reason": "unreadable"})
            continue
        stats["images_valid"] += 1
        faces = detector.detect(img, max_faces=cfg.max_faces)
        if not faces:
            stats["images_without_face"] += 1
            stats["skipped"].append({"file": str(sample.path), "reason": "no face detected"})
            continue
        face: FaceBox = faces[0]              # largest face = the subject
        sample.faces_found = len(faces)
        aligned = align_face(img, face, cfg.align_size)

        variants = [aligned]
        for k in range(max(0, cfg.augment_per_image)):
            variants.append(augment_face(aligned, rng, allow_flip=allow_flip, index=k))
            stats["augmented"] += 1

        for vi, patch in enumerate(variants):
            try:
                vec = extractor.extract(patch)
            except Exception as exc:            # pragma: no cover
                log.warning("Extraction failed for %s: %s", sample.path.name, exc)
                continue
            X_rows.append(vec)
            y_rows.append(label_to_idx[sample.label])
            stats["faces_used"] += 1
            records.append({
                "file": str(sample.path.relative_to(dataset_dir)),
                "label": sample.label,
                "variant": "original" if vi == 0 else f"aug{k + 1}",
                "face_box": face.box,
                "detector": detector.name,
            })

    if not X_rows:
        raise ValueError("No usable faces were extracted. Check that the dataset "
                         "contains clear, front-facing photographs.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return BuildResult(X=X, y=y, classes=classes, records=records, stats=stats)
