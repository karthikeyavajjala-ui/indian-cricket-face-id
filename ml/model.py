"""
Model definition, training and evaluation.

Design notes
------------
* ``ModelBundle`` is the single artefact written to ``models/model.pkl``. It
  contains the classifier, the scaler/PCA transform, the class order, the
  pipeline configuration that produced it, and the metrics - so the inference
  path only needs the pickle file.
* Scaling always happens (HOG/LBP dimensions differ by orders of magnitude).
* PCA is optional; with the deep 128-D embedding it is usually unnecessary.
* Metrics are computed on a *held-out* split (stratified) and, when enough
  samples exist, with stratified k-fold cross-validation. Nothing is fabricated:
  if a class is too small to stratify, the report says so.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from . import config
from .pipeline import BuildResult

log = logging.getLogger("icfid.model")

MODEL_FORMAT_VERSION = 2
CLASSIFIERS = ("svc", "logistic", "rf", "knn")


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Everything required to reproduce a prediction."""

    classifier: object
    classes: List[str]
    scaler: object = None
    pca: object = None
    cfg: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    centroids: Dict[str, list] = field(default_factory=dict)   # per-class mean descriptor
    format_version: int = MODEL_FORMAT_VERSION

    # -- persistence ------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or config.DEFAULT_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Optional[Path] = None) -> "ModelBundle":
        path = Path(path or config.DEFAULT_MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"No trained model at {path}")
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} is not an INDIAN CRICKET FACE ID model bundle")
        return bundle

    # -- introspection ----------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def feature_dim(self) -> Optional[int]:
        try:
            return int(self.classifier.n_features_in_)
        except AttributeError:
            return None

    def describe(self) -> dict:
        return {
            "format_version": self.format_version,
            "classes": self.classes,
            "n_classes": self.n_classes,
            "feature_dim": self.feature_dim,
            "feature_extractor": self.cfg.get("feature_extractor"),
            "detector": self.cfg.get("detector"),
            "classifier": self.cfg.get("classifier"),
            "pca_components": self.cfg.get("pca_components", 0),
            "confidence_threshold": self.cfg.get("confidence_threshold"),
            "min_margin": self.cfg.get("min_margin"),
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


# --------------------------------------------------------------------------
# Classifier factory
# --------------------------------------------------------------------------
def make_classifier(name: str = "svc", **overrides):
    """Build a classifier. SVC is the default: it performs best on small,
    high-dimensional face-feature sets."""
    name = (name or "svc").lower()
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{name}'. Choose from {CLASSIFIERS}")

    params = {
        "svc": dict(kernel="rbf", C=10.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=0),
        "logistic": dict(max_iter=4000, C=5.0, class_weight="balanced",
                         n_jobs=None, random_state=0),
        "rf": dict(n_estimators=400, max_depth=None, min_samples_leaf=1,
                   class_weight="balanced_subsample", n_jobs=-1, random_state=0),
        "knn": dict(n_neighbors=3, weights="distance", metric="cosine"),
    }[name]
    params.update(overrides or {})
    params = {k: v for k, v in params.items() if v is not None}
    return {
        "svc": SVC, "logistic": LogisticRegression,
        "rf": RandomForestClassifier, "knn": KNeighborsClassifier,
    }[name](**params), name, params


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_model(build: BuildResult, cfg: Optional[config.PipelineConfig] = None,
                dataset_dir: Optional[Path] = None,
                progress=None) -> ModelBundle:
    """
    Train + evaluate. Returns a :class:`ModelBundle` (not yet saved).
    """
    cfg = cfg or config.PipelineConfig()
    log.info("Training on %s samples x %s features (%s classes)",
             build.X.shape[0], build.X.shape[1], len(build.classes))

    X, y = build.X, build.y
    n_samples, n_features = X.shape
    class_counts = {c: int((y == i).sum()) for i, c in enumerate(build.classes)}
    smallest = min(class_counts.values())

    # --- split (stratified when possible) ---------------------------------
    used_stratify = smallest >= 2
    test_size = cfg.test_size if n_samples * cfg.test_size >= len(build.classes) else 0.0
    if test_size > 0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=cfg.seed,
            stratify=y if used_stratify else None,
        )
    else:                                     # too few samples to hold anything out
        X_tr, y_tr = X, y
        X_te = y_te = None

    if progress:
        progress("Fitting transform + classifier", 55.0)

    # --- transform + fit ---------------------------------------------------
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    pca = None
    if cfg.pca_components and n_features > cfg.pca_components:
        max_components = min(cfg.pca_components, X_tr_s.shape[0] - 1, n_features)
        if max_components >= 2:
            pca = PCA(n_components=max_components, random_state=cfg.seed)
            X_tr_s = pca.fit_transform(X_tr_s)

    clf, clf_name, clf_params = make_classifier(cfg.classifier, **(cfg.classifier_params or {}))
    t0 = time.perf_counter()
    clf.fit(X_tr_s, y_tr)
    train_seconds = time.perf_counter() - t0

    def transform(mat: np.ndarray) -> np.ndarray:
        out = scaler.transform(mat)
        if pca is not None:
            out = pca.transform(out)
        return out

    # --- held-out evaluation ----------------------------------------------
    metrics: Dict[str, object] = {
        "training_samples": int(X_tr.shape[0]),
        "validation_samples": int(0 if X_te is None else X_te.shape[0]),
        "total_samples": int(n_samples),
        "feature_dim": int(n_features),
        "n_classes": len(build.classes),
        "class_counts": class_counts,
        "classifier": clf_name,
        "classifier_params": {k: str(v) for k, v in clf_params.items()},
        "feature_extractor": build.stats.get("feature_extractor"),
        "detector": build.stats.get("detector"),
        "training_seconds": round(train_seconds, 3),
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_dir or config.DATASET_DIR),
            "images_used": build.stats.get("images_valid", 0),
            "images_skipped_no_face": build.stats.get("images_without_face", 0),
            "augmented_variants": build.stats.get("augmented", 0),
        },
        "evaluation": {},
        "notes": [],
    }
    ev = metrics["evaluation"]

    if X_te is not None and len(X_te) > 0 and len(set(y_te.tolist())) > 0:
        y_pred = clf.predict(transform(X_te))
        labels = list(range(len(build.classes)))
        ev["accuracy"] = round(float(accuracy_score(y_te, y_pred)), 4)
        ev["precision_macro"] = round(float(precision_score(
            y_te, y_pred, average="macro", zero_division=0, labels=labels)), 4)
        ev["recall_macro"] = round(float(recall_score(
            y_te, y_pred, average="macro", zero_division=0, labels=labels)), 4)
        ev["f1_macro"] = round(float(f1_score(
            y_te, y_pred, average="macro", zero_division=0, labels=labels)), 4)
        ev["f1_weighted"] = round(float(f1_score(
            y_te, y_pred, average="weighted", zero_division=0, labels=labels)), 4)
        ev["confusion_matrix"] = confusion_matrix(y_te, y_pred, labels=labels).tolist()
        ev["per_class"] = _per_class_report(y_te, y_pred, build.classes, labels)
        ev["split"] = {
            "test_size": cfg.test_size,
            "stratified": used_stratify,
            "seed": cfg.seed,
            "samples": int(len(y_te)),
        }
        ev["confusion_matrix_labels"] = build.classes
        _add_ranking_metrics(ev, clf, transform(X_te), y_te, build.classes)
        if not used_stratify:
            metrics["notes"].append(
                "Some classes had fewer than 2 samples, so the split was not "
                "stratified - read per-class numbers with care.")
        metrics["notes"].append(
            "Held-out metrics come from the same image sources as training "
            "(augmented variants of a small photo set), so they OVERSTATE "
            "accuracy on unseen photographs.")
    else:
        ev["accuracy"] = None
        metrics["notes"].append(
            "Not enough samples for a held-out split - no accuracy is reported. "
            "Add more images per player (>= 8 recommended).")

    # --- cross-validation (optional, honest-ish estimate) ------------------
    if cfg.cross_validate and smallest >= 2 and n_samples >= 2 * len(build.classes) >= 4:
        try:
            if progress:
                progress("Cross-validating", 80.0)
            metrics["cross_validation"] = _cross_validate(X, y, build.classes, cfg, scaler, pca)
        except Exception as exc:  # pragma: no cover
            log.warning("Cross-validation skipped: %s", exc)

    centroids = _class_centroids(X, y, build.classes, build.records)
    fingerprint = _dataset_fingerprint(build)
    metrics["dataset"]["fingerprint"] = fingerprint
    metrics["class_samples"] = _class_sample_counts(build)

    bundle = ModelBundle(
        classifier=clf, classes=build.classes, scaler=scaler, pca=pca,
        cfg=cfg.to_dict(), metrics=metrics, centroids=centroids,
        metadata={
            "created_by": "train.py",
            "created_at": metrics["trained_at"],
            "model_version": f"v{MODEL_FORMAT_VERSION}-{fingerprint[:8]}",
            "dataset_fingerprint": fingerprint,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pipeline": "load -> validate -> detect -> align -> augment -> encode -> classify",
            "recognition_scope": (
                "This model recognises ONLY the players present in its training "
                "dataset. It is not a general Indian-cricket-player recogniser."
            ),
        },
    )
    return bundle


def _class_centroids(X: np.ndarray, y: np.ndarray, classes: List[str],
                     records: Optional[List[dict]] = None) -> Dict[str, list]:
    """
    Mean descriptor per class (original, non-augmented variants when known).

    Used at inference time for a *prototype similarity* score that is reported
    alongside - never instead of - the calibrated classifier probability.
    """
    mask = np.ones(len(y), dtype=bool)
    if records and len(records) == len(y):
        mask = np.array([r.get("variant") == "original" for r in records], dtype=bool)
        if mask.sum() < len(classes):          # never end up with an empty class
            mask = np.ones(len(y), dtype=bool)
    centroids: Dict[str, list] = {}
    for i, name in enumerate(classes):
        rows = X[mask & (y == i)]
        if rows.shape[0] == 0:
            rows = X[y == i]
        if rows.shape[0] == 0:
            continue
        centroids[name] = rows.mean(axis=0).astype(np.float32).tolist()
    return centroids


def _class_sample_counts(build: BuildResult) -> Dict[str, int]:
    originals: Dict[str, int] = {}
    augmented: Dict[str, int] = {}
    for row, label in zip(build.records, build.y):
        name = build.classes[int(label)]
        if row.get("variant") == "original":
            originals[name] = originals.get(name, 0) + 1
        else:
            augmented[name] = augmented.get(name, 0) + 1
    if not originals:                          # records unavailable
        for i, name in enumerate(build.classes):
            originals[name] = int((build.y == i).sum())
    return {
        name: {"images": originals.get(name, 0),
               "augmented": augmented.get(name, 0),
               "training_vectors": originals.get(name, 0) + augmented.get(name, 0)}
        for name in build.classes
    }


def _dataset_fingerprint(build: BuildResult) -> str:
    """Deterministic hash of file paths + labels + extractor (identifies the data
    a model was trained on, so the DB can record it as a model version)."""
    h = hashlib.sha256()
    h.update(build.stats.get("feature_extractor", "").encode())
    h.update(build.stats.get("detector", "").encode())
    for row in sorted(build.records, key=lambda r: (r.get("label", ""), r.get("file", ""),
                                                    r.get("variant", ""))):
        if row.get("variant") != "original":
            continue
        h.update(f"{row.get('label')}|{row.get('file')}".encode())
    return h.hexdigest()


def save_model(bundle: ModelBundle, path: Optional[Path] = None,
               write_metrics_json: bool = True) -> Path:
    """Persist model.pkl plus a human-readable metrics sidecar."""
    path = bundle.save(path)
    if write_metrics_json:
        try:
            sidecar = Path(path).with_name("metrics.json")
            payload = {
                "model_path": str(path),
                "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "metadata": bundle.metadata,
                "cfg": bundle.cfg,
                "classes": bundle.classes,
                "metrics": bundle.metrics,
            }
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            log.warning("Could not write metrics.json: %s", exc)
    return path


def load_metrics_json(path: Optional[Path] = None) -> Optional[dict]:
    path = Path(path or config.DEFAULT_MODEL_PATH).with_name("metrics.json")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _per_class_report(y_true, y_pred, classes, labels) -> List[dict]:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0,
                                   target_names=classes, output_dict=True)
    rows = []
    for name in classes:
        row = report.get(name, {})
        rows.append({
            "class": name,
            "precision": round(float(row.get("precision", 0.0)), 4),
            "recall": round(float(row.get("recall", 0.0)), 4),
            "f1": round(float(row.get("f1-score", 0.0)), 4),
            "support": int(row.get("support", 0)),
        })
    return rows


def _add_ranking_metrics(ev: dict, clf, X_te_t, y_te, classes) -> None:
    """Top-k accuracy / macro AUC when the classifier exposes probabilities."""
    if not hasattr(clf, "predict_proba"):
        return
    try:
        proba = clf.predict_proba(X_te_t)
        labels = np.arange(len(classes))
        k = min(3, len(classes) - 1)          # top-k is meaningless when k >= n_classes
        if k >= 2:
            ev[f"top{k}_accuracy"] = round(float(top_k_accuracy_score(
                y_te, proba, k=k, labels=labels)), 4)
        if len(classes) == 2:
            ev["roc_auc_macro"] = round(float(roc_auc_score(
                y_te, proba[:, 1])), 4)
        else:
            ev["roc_auc_macro"] = round(float(roc_auc_score(
                y_te, proba, multi_class="ovr", average="macro", labels=labels)), 4)
    except Exception as exc:  # pragma: no cover
        log.debug("Ranking metrics unavailable: %s", exc)


def _cross_validate(X, y, classes, cfg, scaler_proto, pca_proto) -> dict:
    """Stratified k-fold using the same transform+classifier recipe."""
    n_splits = int(min(5, min((y == c).sum() for c in set(y.tolist()))))
    if n_splits < 3:
        return {"available": False, "reason": "not enough samples per class"}
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    scores = []
    per_class_hits = {c: [0, 0] for c in classes}
    for train_idx, test_idx in folds.split(X, y):
        sc = StandardScaler().fit(X[train_idx])
        Xtr, Xte = sc.transform(X[train_idx]), sc.transform(X[test_idx])
        if pca_proto is not None:
            n_comp = min(pca_proto.n_components, Xtr.shape[0] - 1, Xtr.shape[1])
            p = PCA(n_components=n_comp, random_state=cfg.seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        clf, _, _ = make_classifier(cfg.classifier, **(cfg.classifier_params or {}))
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)
        scores.append(float(accuracy_score(y[test_idx], pred)))
        for true_c, pred_c in zip(y[test_idx], pred):
            name = classes[int(true_c)]
            per_class_hits[name][1] += 1
            if true_c == pred_c:
                per_class_hits[name][0] += 1
    return {
        "available": True,
        "folds": n_splits,
        "accuracy_mean": round(float(np.mean(scores)), 4),
        "accuracy_std": round(float(np.std(scores)), 4),
        "accuracy_folds": [round(s, 4) for s in scores],
        "per_class_recall": {
            name: round(hits / total, 4) if total else None
            for name, (hits, total) in per_class_hits.items()
        },
        "note": ("Cross-validation is still optimistic here: augmented variants of "
                 "the same photograph can land in different folds."),
    }
