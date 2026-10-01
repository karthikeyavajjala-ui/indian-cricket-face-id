"""
Inference engine - the single place where images become decisions.

Responsibilities
----------------
1. decode + detect + align + extract (identical to the training path)
2. classify
3. apply the responsible-AI decision policy:
      * probability below ``confidence_threshold``  -> "unknown"
      * top-2 margin below ``min_margin``           -> "unknown" (ambiguous)
      * deep-embedding mode may additionally require ``min_similarity``
4. never invent a name: if the model or dataset is missing the API returns
   ``status="model_missing"`` / ``status="dataset_missing"``
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .face import FaceBox, align_face, crop_face_pixels, make_detector
from .features import get_extractor
from .model import ModelBundle

log = logging.getLogger("icfid.predictor")


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def encode_thumbnail(bgr: np.ndarray, max_side: int = 320, quality: int = 88) -> str:
    """JPEG -> base64 data URI for the 'detected face' panel."""
    if bgr is None or bgr.size == 0:
        return ""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class FaceIdentifier:
    """Loads a :class:`ModelBundle` and answers identification queries."""

    def __init__(self, model_path: Optional[Path] = None,
                 cfg_overrides: Optional[dict] = None,
                 player_catalog: Optional[dict] = None):
        self.model_path = Path(model_path or config.DEFAULT_MODEL_PATH)
        self.bundle: Optional[ModelBundle] = None
        self.error: Optional[str] = None
        self.loaded_at: Optional[str] = None
        self.cfg_overrides = cfg_overrides or {}
        self.player_catalog = player_catalog or {}
        self._detector = None
        self._extractor = None
        # Always have a usable config so callers can read thresholds/top_k even when
        # no model file exists yet (the API then answers "model_missing").
        self.cfg = config.PipelineConfig.from_dict(self.cfg_overrides)
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.model_path.is_file():
            self.error = (f"No trained model found at {self.model_path}. "
                          f"Add images to dataset/ and run: python -m ml.train")
            log.warning(self.error)
            return
        try:
            self.bundle = ModelBundle.load(self.model_path)
            self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            cfg = dict(self.bundle.cfg or {})
            cfg.update(self.cfg_overrides)
            self.cfg = config.PipelineConfig.from_dict(cfg)
            self._detector = make_detector(self.cfg.detector)
            self._extractor = get_extractor(
                self.cfg.feature_extractor, **(self.cfg.extractor_params or {}))
            self.error = None
            log.info("Model loaded: %s classes, %s (%s + %s)",
                     self.bundle.n_classes, self.bundle.metrics.get("trained_at"),
                     self._detector.name, self._extractor.name)
        except Exception as exc:                       # pragma: no cover
            self.bundle = None
            self.error = f"Failed to load model: {exc}"
            log.exception("Model load failed")

    def reload(self) -> None:
        self._load()

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def engine_info(self) -> dict:
        if not self.ready:
            return {"ready": False, "error": self.error,
                    "model_path": str(self.model_path)}
        return {
            "ready": True,
            "model_path": str(self.model_path),
            "model_version": self.bundle.metadata.get("model_version")
            or self.bundle.metadata.get("created_at"),
            "trained_at": self.bundle.metrics.get("trained_at"),
            "detector": self._detector.name,
            "feature_extractor": self._extractor.name,
            "classifier": self.bundle.cfg.get("classifier"),
            "classes": self.bundle.classes,
            "n_classes": self.bundle.n_classes,
            "confidence_threshold": self.cfg.confidence_threshold,
            "min_margin": self.cfg.min_margin,
            "metrics": self.bundle.metrics.get("evaluation", {}),
        }

    # ------------------------------------------------------------------
    def _transform(self, features: np.ndarray) -> np.ndarray:
        out = self.bundle.scaler.transform(features) if self.bundle.scaler is not None else features
        if self.bundle.pca is not None:
            out = self.bundle.pca.transform(out)
        return out

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        clf = self.bundle.classifier
        if hasattr(clf, "predict_proba"):
            try:
                return np.asarray(clf.predict_proba(X), dtype=np.float64)
            except Exception as exc:               # pragma: no cover
                log.debug("predict_proba failed (%s); falling back to scores", exc)
        if hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(X), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            return np.vstack([softmax(row) for row in scores])
        pred = clf.predict(X)
        out = np.zeros((len(pred), self.bundle.n_classes), dtype=np.float64)
        out[np.arange(len(pred)), pred.astype(int)] = 1.0
        return out

    def _prototype_similarity(self, feature: np.ndarray) -> Optional[np.ndarray]:
        centroids = getattr(self.bundle, "centroids", None)
        if not centroids:
            return None
        try:
            mat = np.vstack([np.asarray(centroids[c], dtype=np.float32)
                             for c in self.bundle.classes])
            v = feature.astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
            return mat @ v
        except Exception:                            # pragma: no cover
            return None

    # ------------------------------------------------------------------
    def identify_image_bytes(self, data: bytes, top_k: Optional[int] = None,
                             max_faces: int = 5) -> dict:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "invalid_image",
                    "message": "The uploaded file could not be decoded as an image."}
        return self.identify(img, top_k=top_k, max_faces=max_faces)

    def identify(self, image_bgr: np.ndarray, top_k: Optional[int] = None,
                 max_faces: int = 5) -> dict:
        """Full inference pipeline for one image."""
        started = time.perf_counter()
        top_k = int(top_k or self.cfg.top_k if self.ready else top_k or 3)

        if not self.ready:
            return {
                "status": "model_missing",
                "message": self.error or "Model not trained yet.",
                "engine": self.engine_info(),
                "dataset_status": config.dataset_summary_safe(),
            }
        if image_bgr is None or image_bgr.size == 0:
            return {"status": "invalid_image", "message": "Empty image."}

        # ---- detect ------------------------------------------------------
        t0 = time.perf_counter()
        faces: List[FaceBox] = self._detector.detect(image_bgr, max_faces=max_faces)
        detect_ms = (time.perf_counter() - t0) * 1000
        if not faces:
            return {
                "status": "no_face",
                "message": "No face detected. Try a clearer, front-facing photo.",
                "engine": self.engine_info(),
                "top_predictions": [],
                "confidence": None,
                "threshold": self.cfg.confidence_threshold,
                "timings_ms": {"detect": round(detect_ms, 1),
                               "total": round((time.perf_counter() - started) * 1000, 1)},
            }

        # ---- encode + classify ------------------------------------------
        t0 = time.perf_counter()
        patches = [align_face(image_bgr, f, self.cfg.align_size) for f in faces]
        features = self._extractor.extract_many(patches)
        encode_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        X = self._transform(features)
        proba = self._probabilities(X)
        classify_ms = (time.perf_counter() - t0) * 1000

        results = []
        for i, (face, patch, row) in enumerate(zip(faces, patches, proba)):
            sims = self._prototype_similarity(features[i])
            results.append(self._decide(face, patch, row, sims, top_k))

        # Primary face = largest detection (typically the subject of the photo)
        primary_idx = max(range(len(results)), key=lambda i: faces[i].area)
        primary = dict(results[primary_idx])
        secondary = [r for i, r in enumerate(results) if i != primary_idx]
        crop = crop_face_pixels(image_bgr, faces[primary_idx])
        thumbnail = encode_thumbnail(crop)

        out = {
            "status": primary["status"],
            "reason": primary.get("reason"),
            "message": primary.get("message"),
            "player": primary.get("player"),
            "confidence": primary.get("confidence"),
            "threshold": primary.get("threshold"),
            "margin": primary.get("margin"),
            "top_predictions": primary["top_predictions"],
            "similarity": primary.get("similarity"),
            "detected_face": {
                "image": thumbnail,
                "box": faces[primary_idx].box,
                "detection_score": round(float(faces[primary_idx].score), 4),
                "image_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            },
            "faces_detected": len(faces),
            "other_faces": [{"box": f.box, "status": r["status"],
                             "player": (r.get("player") or {}).get("name"),
                             "confidence": r.get("confidence")}
                            for f, r in zip(faces, results) if r is not primary],
            "engine": self.engine_info(),
            "dataset_status": config.dataset_summary_safe(),
            "timings_ms": {
                "detect": round(detect_ms, 1),
                "encode": round(encode_ms, 1),
                "classify": round(classify_ms, 1),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
            "disclaimer": DISCLAIMER,
        }
        return out

    # ------------------------------------------------------------------
    def _decide(self, face: FaceBox, patch: np.ndarray, probs: np.ndarray,
                sims: Optional[np.ndarray], top_k: int) -> dict:
        order = np.argsort(probs)[::-1]
        ranked = []
        for idx in order[:max(1, top_k)]:
            name = self.bundle.classes[int(idx)]
            player = self.player_catalog.get(slugify(name)) or {}
            entry = {
                "class_name": name,
                "player_id": slugify(name),
                "display_name": player.get("name") or name.replace("_", " "),
                "confidence": round(float(probs[int(idx)]), 4),
            }
            if sims is not None:
                entry["similarity"] = round(float(sims[int(idx)]), 4)
            if player.get("role"):
                entry["role"] = player.get("role")
            ranked.append(entry)

        best = ranked[0]
        second_conf = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
        margin = float(best["confidence"] - second_conf)
        threshold = float(self.cfg.confidence_threshold)

        status, reason, message = "identified", None, None
        if best["confidence"] < threshold:
            status = "unknown"
            reason = "low_confidence"
            message = (f"Best match '{best['display_name']}' scored "
                       f"{best['confidence'] * 100:.1f}%, below the "
                       f"{threshold * 100:.0f}% threshold. Not identified.")
        elif len(ranked) > 1 and margin < self.cfg.min_margin:
            status = "unknown"
            reason = "ambiguous"
            message = (f"Top matches are too close ({best['display_name']} "
                       f"{best['confidence'] * 100:.1f}% vs "
                       f"{ranked[1]['display_name']} {second_conf * 100:.1f}%). "
                       f"Not identified.")

        player = None
        if status == "identified":
            player = {
                "id": best["player_id"],
                "name": best["display_name"],
                "confidence": best["confidence"],
            }
            catalog_entry = self.player_catalog.get(best["player_id"])
            if catalog_entry:
                player.update({k: catalog_entry.get(k) for k in
                               ("role", "batting_style", "bowling_style", "jersey_number")})

        return {
            "status": status,
            "reason": reason,
            "message": message,
            "player": player,
            "confidence": best["confidence"],
            "threshold": threshold,
            "margin": round(margin, 4),
            "similarity": best.get("similarity"),
            "top_predictions": ranked,
            "face_box": face.box,
            "detection_score": round(float(face.score), 4),
        }


DISCLAIMER = ("This system is a demonstration of facial identification technology and "
              "should not be used for surveillance, law enforcement, access control, or "
              "high-impact decisions without appropriate safeguards. It recognises only "
              "the people present in its training dataset - it cannot identify every "
              "Indian cricketer.")


# --------------------------------------------------------------------------
# Process-wide singleton (used by the FastAPI app)
# --------------------------------------------------------------------------
_IDENTIFIER: Optional[FaceIdentifier] = None


def get_identifier(model_path: Optional[Path] = None,
                   cfg_overrides: Optional[dict] = None,
                   player_catalog: Optional[dict] = None) -> FaceIdentifier:
    global _IDENTIFIER
    if _IDENTIFIER is None:
        _IDENTIFIER = FaceIdentifier(model_path=model_path,
                                     cfg_overrides=cfg_overrides,
                                     player_catalog=player_catalog)
    return _IDENTIFIER


def reset_identifier() -> None:
    global _IDENTIFIER
    _IDENTIFIER = None
