"""
Face detection + geometric alignment.

Two interchangeable detectors are provided behind one interface:

* ``haar``  - OpenCV's bundled Viola-Jones cascade. Zero downloads, always
              available, good enough for clear frontal photos.
* ``yunet`` - OpenCV-Zoo YuNet CNN detector (``cv2.FaceDetectorYN``) with
              5 landmarks. Much stronger on pose/lighting variation. Enabled
              automatically when the ONNX file was fetched with
              ``python scripts/fetch_models.py``.

``auto`` picks YuNet when the model file exists, otherwise Haar.

Every detector returns :class:`FaceBox` objects; :func:`align_face` maps the
detected face to a canonical pose so that *the same pixel geometry* is fed to
the feature extractor during training and during inference. That consistency is
what makes the classifier reproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import cv2
import numpy as np

from . import config

log = logging.getLogger("icfid.face")

# Canonical landmark positions (fractions of the aligned crop) used by align_face.
# These match the reference geometry of OpenCV-Zoo's SFace alignCrop, so the same
# alignment can feed both the classic (HOG/LBP) and the deep (embedding) extractor.
CANON_RIGHT_EYE = (0.3419, 0.4616)     # subject's right eye == image-left eye
CANON_LEFT_EYE = (0.6565, 0.4598)
CANON_NOSE = (0.5002, 0.6405)
CANON_MOUTH = (0.5013, 0.8247)         # mouth centre


@dataclass
class FaceBox:
    """A single detected face."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    detector: str = "haar"
    landmarks: Optional[np.ndarray] = None   # (5, 2) float array, if available

    @property
    def box(self) -> list[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

    def clip(self, shape: Sequence[int]) -> "FaceBox":
        H, W = shape[0], shape[1]
        x1 = max(0, min(int(self.x), W - 1))
        y1 = max(0, min(int(self.y), H - 1))
        x2 = max(x1 + 1, min(int(self.x + self.w), W))
        y2 = max(y1 + 1, min(int(self.y + self.h), H))
        return FaceBox(x1, y1, x2 - x1, y2 - y1, self.score, self.detector, self.landmarks)


# --------------------------------------------------------------------------
# Detector interface
# --------------------------------------------------------------------------
class FaceDetector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray, max_faces: int = 10) -> List[FaceBox]:
        ...


class HaarFaceDetector:
    """Viola-Jones cascade detector (bundled with OpenCV)."""

    name = "haar"

    def __init__(self, min_neighbors: int = 6, scale_factor: float = 1.1):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor

    def detect(self, image_bgr: np.ndarray, max_faces: int = 10) -> List[FaceBox]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        H, W = gray.shape[:2]
        # Scale the minimum accepted face with the image so that newspapers
        # full of texture do not produce dozens of tiny false positives.
        min_side = max(config.MIN_FACE_SIDE, min(H, W) // 14)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(min_side, min_side), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(rects) == 0:
            # second, more permissive pass
            rects = self._cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5,
                minSize=(min_side, min_side),
            )
        faces: List[FaceBox] = []
        for (x, y, w, h) in rects:
            if not (0.62 <= (w / max(h, 1)) <= 1.60):     # faces are roughly square
                continue
            faces.append(FaceBox(int(x), int(y), int(w), int(h), 1.0, self.name))
        faces = _non_max_suppression(faces, iou_threshold=0.35)
        faces.sort(key=lambda f: f.area, reverse=True)
        faces = faces[:max_faces]
        for face in faces:
            face.landmarks = self._estimate_landmarks(gray, face)
        return faces

    def _estimate_landmarks(self, gray: np.ndarray, face: FaceBox) -> Optional[np.ndarray]:
        """Best-effort eye localisation inside the upper face region."""
        y1, y2 = face.y, face.y + int(face.h * 0.62)
        x1, x2 = face.x, face.x + face.w
        roi = gray[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if roi.size == 0:
            return None
        eyes = self._eye_cascade.detectMultiScale(roi, 1.1, 6, minSize=(12, 12))
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[0])
        left, right = eyes[0], eyes[-1]
        lm = np.zeros((5, 2), dtype=np.float32)
        lm[0] = [x1 + left[0] + left[2] / 2, y1 + left[1] + left[3] / 2]      # right eye (image left)
        lm[1] = [x1 + right[0] + right[2] / 2, y1 + right[1] + right[3] / 2]  # left eye
        lm[2] = [face.x + face.w * 0.5, face.y + face.h * 0.62]               # nose (approx)
        lm[3] = [face.x + face.w * 0.38, face.y + face.h * 0.80]              # mouth left
        lm[4] = [face.x + face.w * 0.62, face.y + face.h * 0.80]              # mouth right
        return lm


class YuNetFaceDetector:
    """YuNet CNN detector (OpenCV Zoo) - requires the ONNX file."""

    name = "yunet"

    def __init__(self, model_path=None, score_threshold: float = 0.6, nms: float = 0.3):
        model_path = model_path or self._find_model()
        if model_path is None:
            raise FileNotFoundError("YuNet ONNX model not found")
        self._model_path = str(model_path)
        self._detector = cv2.FaceDetectorYN.create(
            self._model_path, "", (320, 320), score_threshold, nms, 5000
        )
        self._size = (320, 320)

    @staticmethod
    def _find_model():
        for candidate in (config.YUNET_DYN_MODEL_PATH, config.YUNET_MODEL_PATH):
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def available(cls) -> bool:
        return cls._find_model() is not None

    def detect(self, image_bgr: np.ndarray, max_faces: int = 10) -> List[FaceBox]:
        H, W = image_bgr.shape[:2]
        if (W, H) != self._size:
            self._detector.setInputSize((W, H))
            self._size = (W, H)
        try:
            _, raw = self._detector.detect(image_bgr)
        except cv2.error as exc:              # pragma: no cover - engine edge cases
            log.warning("YuNet failed (%s); falling back to Haar for this frame", exc)
            return HaarFaceDetector().detect(image_bgr, max_faces)
        faces: List[FaceBox] = []
        if raw is None:
            return faces
        for row in raw:
            x, y, w, h = (float(v) for v in row[:4])
            score = float(row[14]) if len(row) > 14 else 1.0
            if w < config.MIN_FACE_SIDE or h < config.MIN_FACE_SIDE:
                continue
            lm = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(FaceBox(int(x), int(y), int(w), int(h), score, self.name, lm))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces[:max_faces]


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def _non_max_suppression(faces: List[FaceBox], iou_threshold: float = 0.35) -> List[FaceBox]:
    kept: List[FaceBox] = []
    for face in sorted(faces, key=lambda f: f.area, reverse=True):
        if all(_iou(face, k) < iou_threshold for k in kept):
            kept.append(face)
    return kept


def make_detector(name: str = "auto", **kwargs) -> FaceDetector:
    """Factory. ``auto`` prefers YuNet, degrades to Haar."""
    name = (name or "auto").lower()
    if name in ("auto", "yunet", "dnn", "cnn"):
        if YuNetFaceDetector.available():
            try:
                return YuNetFaceDetector(**kwargs)
            except Exception as exc:  # pragma: no cover
                log.warning("YuNet unavailable (%s) - using Haar detector", exc)
        elif name != "auto":
            log.warning("YuNet model file missing - using Haar detector instead")
    return HaarFaceDetector()


def detector_status() -> dict:
    return {
        "yunet_available": YuNetFaceDetector.available(),
        "yunet_path": str(YuNetFaceDetector._find_model() or ""),
        "haar_available": True,
        "active": "yunet" if YuNetFaceDetector.available() else "haar",
    }


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------
def _similarity_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation+uniform scale+translation)."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    denom = float((src_c ** 2).sum())
    if denom <= 1e-8:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    a = float((src_c * dst_c).sum()) / denom
    b = float((src_c[:, 0] * dst_c[:, 1] - src_c[:, 1] * dst_c[:, 0]).sum()) / denom
    M = np.array([[a, -b, dst_mean[0] - (a * src_mean[0] - b * src_mean[1])],
                  [b,  a, dst_mean[1] - (b * src_mean[0] + a * src_mean[1])]],
                 dtype=np.float32)
    return M


def align_face(image_bgr: np.ndarray, face: FaceBox, size: int = 112) -> np.ndarray:
    """
    Crop + geometrically normalise a detected face to a ``size`` x ``size`` patch.

    Uses landmarks when available (eyes + mouth) for a full similarity fit,
    otherwise a scale-normalised crop derived from the bounding box.
    """
    img = face.clip(image_bgr.shape)
    if face.landmarks is not None and len(face.landmarks) >= 5:
        lm = np.asarray(face.landmarks, dtype=np.float32)
        # YuNet order: right-eye, left-eye, nose, right-mouth, left-mouth (image coords)
        right_eye, left_eye = lm[0], lm[1]
        mouth = (lm[3] + lm[4]) / 2.0
        nose = lm[2]
        src = np.array([right_eye, left_eye, mouth], dtype=np.float32)
        dst = np.array([[CANON_RIGHT_EYE[0] * size, CANON_RIGHT_EYE[1] * size],
                        [CANON_LEFT_EYE[0] * size, CANON_LEFT_EYE[1] * size],
                        [CANON_MOUTH[0] * size, CANON_MOUTH[1] * size]], dtype=np.float32)
        try:
            M = _similarity_from_points(src, dst)
            aligned = cv2.warpAffine(image_bgr, M, (size, size), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            if _looks_like_face(aligned):
                return aligned
            # keep the fitted nose where it should be if the fit produced garbage
            _ = nose
        except Exception:  # pragma: no cover
            pass

    # Fallback: padded bounding-box crop, eyes-in-upper-third heuristic.
    pad_x, pad_y = int(face.w * 0.10), int(face.h * 0.14)
    x1 = max(0, face.x - pad_x)
    y1 = max(0, face.y - pad_y)
    x2 = min(image_bgr.shape[1], face.x + face.w + pad_x)
    y2 = min(image_bgr.shape[0], face.y + face.h + int(face.h * 0.22))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr[img.y:img.y + img.h, img.x:img.x + img.w]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _looks_like_face(patch: np.ndarray) -> bool:
    """Sanity check that a warp did not collapse to a flat/black patch."""
    if patch is None or patch.size == 0:
        return False
    return float(patch.std()) > 4.0


def crop_face_pixels(image_bgr: np.ndarray, face: FaceBox, pad: float = 0.28) -> np.ndarray:
    """Loose RGB crop used for the 'detected face' thumbnail in the UI."""
    h, w = image_bgr.shape[:2]
    px, py = int(face.w * pad), int(face.h * pad)
    x1, y1 = max(0, face.x - px), max(0, face.y - py)
    x2, y2 = min(w, face.x + face.w + px), min(h, face.y + face.h + py)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    return crop


def draw_faces(image_bgr: np.ndarray, faces: Sequence[FaceBox],
               labels: Optional[Sequence[str]] = None) -> np.ndarray:
    """Annotated copy of the image (used by the CLI tools and for debugging)."""
    canvas = image_bgr.copy()
    for idx, face in enumerate(faces):
        colour = (90, 220, 170) if idx == 0 else (255, 170, 60)
        cv2.rectangle(canvas, (face.x, face.y), (face.x + face.w, face.y + face.h),
                      colour, max(2, canvas.shape[0] // 300))
        if labels and idx < len(labels):
            cv2.putText(canvas, labels[idx], (face.x, max(20, face.y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return canvas
