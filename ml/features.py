"""
Feature extractors - the "architecture allows swapping in a stronger embedding
model later" part of the pipeline.

An extractor turns an *aligned* face patch into a fixed-length float vector.
Because the trained model records which extractor produced its features, you can
retrain with a different encoder and the inference path picks it up
automatically.

Built-in extractors
-------------------
``hog_lbp``  (default, zero downloads)
    HOG texture descriptor + uniform-LBP spatial histograms + a coarse intensity
    map. 3428-D. Works surprisingly well for small identities-per-class datasets
    and needs no GPU / no model download.

``sface``    (deep, optional - ~37 MB ONNX via scripts/fetch_models.py)
    OpenCV-Zoo SFace (MobileFaceNet trained with the Sigmoid-Constrained
    Hypersphere loss) producing a 128-D L2-normalised face embedding. This is
    the recommended extractor when you have enough images per player.

``pixel``    (baseline)
    Tiny greyscale intensities, normalised. Useful as a floor to compare against.

Adding your own (e.g. ArcFace, FaceNet, InsightFace, a fine-tuned CNN) means
subclassing :class:`FeatureExtractor` and registering it with ``@register``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type

import cv2
import numpy as np

from . import config

log = logging.getLogger("icfid.features")

# --------------------------------------------------------------------------
# LBP helpers (implemented with NumPy/SciPy-free code to keep deps light)
# --------------------------------------------------------------------------
def _uniform_lbp_codes(p: int = 8) -> np.ndarray:
    """Lookup table mapping a raw 2^p LBP code to 0..p(p-1)+2 (uniform + noise)."""
    table = np.zeros(1 << p, dtype=np.int32)
    assigned = np.zeros(1 << p, dtype=bool)
    next_bin = 0
    for code in range(1 << p):
        bits = [(code >> i) & 1 for i in range(p)]
        transitions = sum(bits[i] != bits[(i + 1) % p] for i in range(p))
        if transitions <= 2:           # uniform pattern
            table[code] = next_bin
            assigned[code] = True
            next_bin += 1
    table[~assigned] = next_bin        # single "non-uniform" catch-all bin
    return table


_LBP_TABLE = _uniform_lbp_codes(8)
_LBP_BINS = int(_LBP_TABLE.max()) + 1


def lbp_image(gray: np.ndarray) -> np.ndarray:
    """Compute the 8-neighbour LBP code image (uint8)."""
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(neighbours):
        shifted = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((shifted >= center).astype(np.uint8) << bit)
    return code


class FeatureExtractor(ABC):
    """Base class for all feature extractors."""

    name: str = "base"
    requires: tuple = ()      # python modules that must be importable
    description: str = ""

    @abstractmethod
    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return a 1-D float32 vector."""

    def extract_many(self, faces: List[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self.extract(f) for f in faces]).astype(np.float32)

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description}


_REGISTRY: Dict[str, Type[FeatureExtractor]] = {}


def register(cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:
    _REGISTRY[cls.name] = cls
    return cls


def get_extractor(name: str = "auto", **params) -> FeatureExtractor:
    """Instantiate an extractor by name. ``auto`` prefers SFace when available."""
    key = (name or "auto").lower()
    if key == "auto":
        key = "sface" if SFaceExtractor.available() else "hog_lbp"
        log.info("Feature extractor auto-selected: %s", key)
    if key not in _REGISTRY:
        raise KeyError(f"Unknown feature extractor '{name}'. "
                       f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**params)


def available_extractors() -> List[dict]:
    out = []
    for key, cls in sorted(_REGISTRY.items()):
        try:
            inst = cls()
            out.append({"name": key, "available": True,
                        "description": cls.description,
                        "dim": int(inst.dim_hint)})
        except Exception as exc:
            out.append({"name": key, "available": False,
                        "description": cls.description,
                        "reason": str(exc)})
    return out


# --------------------------------------------------------------------------
# 1. Classic HOG + LBP (default)
# --------------------------------------------------------------------------
@register
class HogLbpExtractor(FeatureExtractor):
    name = "hog_lbp"
    description = ("HOG texture + uniform-LBP spatial histograms + intensity map "
                   "(classic, no downloads required)")

    def __init__(self, size: int = 128, lbp_grid: int = 6, gray_grid: int = 32):
        self.size = size
        self.lbp_grid = lbp_grid
        self.gray_grid = gray_grid
        self._hog = cv2.HOGDescriptor(
            _winSize=(size, size), _blockSize=(32, 32), _blockStride=(16, 16),
            _cellSize=(16, 16), _nbins=9,
        )
        hog_dim = ((size - 32) // 16 + 1) ** 2 * 4 * 9          # 49 blocks * 4 cells * 9
        self.dim_hint = hog_dim + lbp_grid * lbp_grid * _LBP_BINS + gray_grid * gray_grid

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = self._prepare(face_bgr)
        hog = self._hog.compute(gray).ravel()
        lbp = self._lbp_hist(self._lbp_of(gray))
        coarse = cv2.resize(gray, (self.gray_grid, self.gray_grid),
                            interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        vec = np.concatenate([hog, lbp, coarse.ravel()])
        return vec.astype(np.float32)

    # -- internals --------------------------------------------------------
    def _prepare(self, face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr.ndim == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr
        if gray.shape[0] != self.size or gray.shape[1] != self.size:
            interp = cv2.INTER_AREA if gray.shape[0] > self.size else cv2.INTER_CUBIC
            gray = cv2.resize(gray, (self.size, self.size), interpolation=interp)
        # CLAHE keeps the descriptor robust to flash / floodlight differences
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _lbp_of(gray: np.ndarray) -> np.ndarray:
        codes = lbp_image(gray)
        return _LBP_TABLE[codes]

    def _lbp_hist(self, mapped: np.ndarray) -> np.ndarray:
        h, w = mapped.shape
        gh = h // self.lbp_grid
        gw = w // self.lbp_grid
        hists = []
        for i in range(self.lbp_grid):
            for j in range(self.lbp_grid):
                cell = mapped[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                hist = np.bincount(cell.ravel(), minlength=_LBP_BINS).astype(np.float32)
                hist /= (hist.sum() + 1e-8)
                hists.append(hist)
        return np.concatenate(hists)


# --------------------------------------------------------------------------
# 2. Deep embedding: OpenCV-Zoo SFace
# --------------------------------------------------------------------------
@register
class SFaceExtractor(FeatureExtractor):
    name = "sface"
    description = ("OpenCV-Zoo SFace (MobileFaceNet) 128-D face embedding - "
                   "run scripts/fetch_models.py to enable")

    def __init__(self, model_path=None):
        self.model_path = str(model_path or config.SFACE_MODEL_PATH)
        if not SFaceExtractor.available(model_path):
            raise FileNotFoundError(
                "SFace ONNX model not found. Run: python scripts/fetch_models.py"
            )
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "This OpenCV build has no cv2.FaceRecognizerSF (needs OpenCV >= 4.5.4)."
            )
        self._recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")
        self.dim_hint = 128

    @staticmethod
    def available(model_path=None) -> bool:
        import pathlib
        return pathlib.Path(model_path or config.SFACE_MODEL_PATH).is_file() and \
            hasattr(cv2, "FaceRecognizerSF")

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        patch = face_bgr
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        if patch.shape[0] != 112 or patch.shape[1] != 112:
            patch = cv2.resize(patch, (112, 112), interpolation=cv2.INTER_CUBIC)
        emb = self._recognizer.feature(patch).ravel().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 1e-8 else emb


# --------------------------------------------------------------------------
# 3. Baseline pixel descriptor
# --------------------------------------------------------------------------
@register
class PixelExtractor(FeatureExtractor):
    name = "pixel"
    description = "Normalised 24x24 greyscale intensities (baseline / debugging)"

    def __init__(self, size: int = 24):
        self.size = size
        self.dim_hint = size * size

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        small = cv2.equalizeHist(small).astype(np.float32) / 255.0
        return small.ravel()


def extractor_matrix_dim(extractor: FeatureExtractor, align_size: int) -> int:
    """Dimension of the vector the extractor produces for a given align size."""
    probe = np.zeros((align_size, align_size, 3), dtype=np.uint8)
    probe[:] = 128
    return int(extractor.extract(probe).shape[0])
