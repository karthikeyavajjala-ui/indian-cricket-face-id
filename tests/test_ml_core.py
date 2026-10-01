"""
Tests for the ML core - fast, no network, no downloads.

    pytest -q                     # everything
    pytest -q -m "not slow"       # skip anything touching the demo dataset

The tests deliberately avoid asserting "accuracy ≥ X" style claims: they verify
that the pipeline is wired correctly (shapes, labels, persistence, decision
policy) rather than that a model is good.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config                                   # noqa: E402
from ml.face import HaarFaceDetector, align_face, make_detector   # noqa: E402
from ml.features import HogLbpExtractor, PixelExtractor, get_extractor, available_extractors  # noqa: E402
from ml.model import ModelBundle, make_classifier, save_model, train_model   # noqa: E402
from ml.pipeline import BuildResult, augment_face, scan_dataset, dataset_summary  # noqa: E402
from ml.predictor import FaceIdentifier, softmax, slugify   # noqa: E402

DEMO_DIR = config.DEMO_DATASET_DIR
needs_demo = pytest.mark.skipif(
    not (DEMO_DIR / "Demo_Persona_A").is_dir(),
    reason="demo dataset not built (run: python scripts/make_demo_dataset.py)")


# ── features ────────────────────────────────────────────────────────────────
def test_hog_lbp_is_deterministic_and_fixed_dim():
    extractor = HogLbpExtractor()
    rng = np.random.default_rng(0)
    image = (rng.random((112, 112, 3)) * 255).astype(np.uint8)
    a, b = extractor.extract(image), extractor.extract(image)
    assert a.shape == b.shape
    assert np.allclose(a, b), "feature extraction must be deterministic"
    assert a.dtype == np.float32


def test_different_images_give_different_features():
    extractor = HogLbpExtractor()
    rng = np.random.default_rng(1)
    first = extractor.extract((rng.random((112, 112, 3)) * 255).astype(np.uint8))
    second = extractor.extract((rng.random((112, 112, 3)) * 255).astype(np.uint8))
    assert not np.allclose(first, second)


def test_extractor_registry_and_auto_selection():
    names = {item["name"] for item in available_extractors()}
    assert {"hog_lbp", "pixel", "sface"} <= names
    auto = get_extractor("auto")
    assert auto.name in {"hog_lbp", "sface"}
    assert get_extractor("pixel").extract(np.zeros((112, 112, 3), np.uint8)).shape[0] == 576
    with pytest.raises(KeyError):
        get_extractor("does-not-exist")


def test_lbp_rejects_random_when_compared_to_face():
    """Sanity: the descriptor of a synthetic face is closer to itself than to noise."""
    extractor = HogLbpExtractor()
    face = np.zeros((112, 112, 3), np.uint8)
    face[30:80, 35:80] = 200                      # crude "face" blob
    noise = (np.random.default_rng(2).random((112, 112, 3)) * 255).astype(np.uint8)
    self_sim = float(np.dot(extractor.extract(face), extractor.extract(face)))
    other_sim = float(np.dot(extractor.extract(face), extractor.extract(noise)))
    assert self_sim > other_sim


# ── face detection / alignment ──────────────────────────────────────────────
def test_detectors_report_status_and_find_nothing_in_noise():
    status = detector_status_safe()
    assert "active" in status
    detector = make_detector("haar")
    noise = (np.random.default_rng(3).random((240, 240, 3)) * 255).astype(np.uint8)
    faces = detector.detect(noise)
    assert isinstance(faces, list)          # may be empty; must not raise


def detector_status_safe() -> dict:
    from ml.face import detector_status
    return detector_status()


def test_align_face_outputs_canonical_size():
    """Alignment must always return the canonical 112x112 patch, aligned or cropped."""
    import cv2
    detector = make_detector("auto")
    demo_image = DEMO_DIR / "Demo_Persona_A" / "00_portrait.jpg"
    if demo_image.is_file():
        image = cv2.imread(str(demo_image))
    else:                                                 # synthetic fallback
        image = np.full((320, 320, 3), 90, np.uint8)
        cv2.circle(image, (160, 150), 90, (190, 170, 150), -1)
        cv2.circle(image, (130, 130), 10, (40, 40, 40), -1)
        cv2.circle(image, (190, 130), 10, (40, 40, 40), -1)
    faces = detector.detect(image)
    if not faces:
        pytest.skip("detector found no face in the fallback image")
    aligned = align_face(image, faces[0], 112)
    assert aligned.shape == (112, 112, 3)
    assert float(aligned.std()) > 1.0, "aligned patch must not be flat"


def test_augmentation_is_label_preserving_shape():
    import random
    patch = (np.random.default_rng(4).random((112, 112, 3)) * 255).astype(np.uint8)
    out = augment_face(patch, random.Random(0), index=1)
    assert out.shape == patch.shape
    assert out.dtype == np.uint8


# ── training / model ────────────────────────────────────────────────────────
def _synthetic_build(n_classes=3, dim=24, seed=0,
                     images_per_class=4, augment=2) -> BuildResult:
    """Synthetic feature matrix shaped like the real one: `images_per_class` photos,
    each contributing one original descriptor plus `augment` augmented variants."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.0, size=(n_classes, dim))
    rows, labels, records = [], [], []
    for cls in range(n_classes):
        for image in range(images_per_class):
            base = centres[cls] + rng.normal(0, 0.35, size=dim)
            for variant in range(augment + 1):
                rows.append(base + rng.normal(0, 0.2, size=dim))
                labels.append(cls)
                records.append({"file": f"c{cls}_img{image}.jpg", "label": f"Player_{cls}",
                                "variant": "original" if variant == 0 else f"aug{variant}"})
    return BuildResult(
        X=np.asarray(rows, dtype=np.float32), y=np.asarray(labels, dtype=np.int64),
        classes=[f"Player_{c}" for c in range(n_classes)], records=records,
        stats={"feature_extractor": "hog_lbp", "detector": "haar",
               "images_valid": n_classes * images_per_class,
               "augmented": sum(1 for r in records if r["variant"] != "original"),
               "images_without_face": 0, "images_seen": n_classes * images_per_class,
               "faces_used": len(records)},
    )


@pytest.mark.parametrize("classifier", ["svc", "logistic", "knn"])
def test_train_and_persist_roundtrip(classifier, tmp_path):
    build = _synthetic_build()
    cfg = config.PipelineConfig(classifier=classifier, augment_per_image=0,
                                cross_validate=False, test_size=0.3, seed=1)
    bundle = train_model(build, cfg, dataset_dir=DEMO_DIR)
    assert isinstance(bundle, ModelBundle)
    assert bundle.classes == build.classes
    assert bundle.metrics["evaluation"]["accuracy"] is not None

    path = save_model(bundle, tmp_path / "model.pkl", write_metrics_json=True)
    assert path.is_file() and (tmp_path / "metrics.json").is_file()

    reloaded = ModelBundle.load(path)
    probe = _synthetic_build(images_per_class=2).X
    assert np.array_equal(reloaded.classifier.predict(reloaded.scaler.transform(probe)),
                          bundle.classifier.predict(bundle.scaler.transform(probe)))
    assert reloaded.centroids, "class centroids are needed for prototype similarity"


def test_metrics_payload_shape():
    build = _synthetic_build(images_per_class=8)
    cfg = config.PipelineConfig(cross_validate=True, test_size=0.3, seed=2)
    bundle = train_model(build, cfg)
    ev = bundle.metrics["evaluation"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"):
        assert key in ev, f"missing {key}"
    matrix = ev["confusion_matrix"]
    assert len(matrix) == len(bundle.classes)
    assert sum(sum(row) for row in matrix) == bundle.metrics["validation_samples"]
    assert bundle.metrics["class_samples"]["Player_0"]["images"] == 8
    assert bundle.metrics["class_samples"]["Player_0"]["training_vectors"] == 24


def test_unknown_classifier_rejected():
    with pytest.raises(ValueError):
        make_classifier("not-a-classifier")


# ── inference + decision policy ─────────────────────────────────────────────
def test_softmax_and_slug():
    probs = softmax(np.array([2.0, 1.0, 0.0]))
    assert pytest.approx(float(probs.sum()), rel=1e-6) == 1.0
    assert slugify("Virat Kohli") == "virat-kohli"
    assert slugify("K. L. Rahul") == "k.-l.-rahul"
    # folder names must produce the same slug as backend/catalog.py so profiles resolve
    from backend.catalog import slugify as catalog_slugify
    for folder in ("Virat_Kohli", "Rohit_Sharma", "Demo_Persona_A"):
        assert slugify(folder) == catalog_slugify(folder)


def test_missing_model_is_reported_not_faked(tmp_path):
    identifier = FaceIdentifier(model_path=tmp_path / "missing.pkl")
    assert identifier.ready is False
    result = identifier.identify(np.zeros((64, 64, 3), np.uint8))
    assert result["status"] == "model_missing"
    assert "message" in result and result["message"]
    assert result.get("player") is None            # never invents an identity
    assert result["engine"]["ready"] is False


def test_decision_policy_rejects_low_confidence_and_ambiguity(tmp_path):
    """The policy is the safety net: no confident, unambiguous winner -> Unknown."""
    build = _synthetic_build(images_per_class=10)
    cfg = config.PipelineConfig(classifier="svc", augment_per_image=0,
                                cross_validate=False, test_size=0.25, seed=3)
    bundle = train_model(build, cfg)
    path = save_model(bundle, tmp_path / "model.pkl", write_metrics_json=False)
    identifier = FaceIdentifier(model_path=path)
    assert identifier.ready

    class StubFace:
        x, y, w, h, score, landmarks = 4, 4, 40, 40, 1.0, None
        detector = "stub"
        box = [4, 4, 40, 40]
        area = 1600

    identifier._detector = type("D", (), {"name": "stub",
                                          "detect": lambda self, img, max_faces=5: [StubFace()]})()
    identifier._extractor = PixelExtractor()

    import backend.catalog  # noqa: F401  (optional catalog, must not break inference)

    # 1. near-uniform probabilities -> ambiguous or low confidence, never a name
    uniform = np.full((1, len(bundle.classes)), 1.0 / len(bundle.classes))
    decision = identifier._decide(StubFace(), np.zeros((112, 112, 3), np.uint8), uniform[0], None, 3)
    assert decision["status"] == "unknown"
    assert decision["player"] is None

    # 2. a clear winner above the threshold -> identified
    confident = np.zeros(len(bundle.classes))
    confident[0] = 0.97
    confident[1:] = 0.03 / (len(bundle.classes) - 1)
    decision = identifier._decide(StubFace(), np.zeros((112, 112, 3), np.uint8), confident, None, 3)
    assert decision["status"] == "identified"
    assert decision["player"]["name"] == bundle.classes[0].replace("_", " ")
    assert decision["player"]["id"] == slugify(bundle.classes[0])
    assert len(decision["top_predictions"]) == 3


# ── dataset scanning ────────────────────────────────────────────────────────
def test_scan_missing_dataset_is_not_fatal(tmp_path):
    samples, counts = scan_dataset(tmp_path / "nope")
    assert samples == [] and counts == {}
    summary = dataset_summary(tmp_path / "nope")
    assert summary["configured"] is False


@needs_demo
@pytest.mark.slow
def test_demo_dataset_scans_and_builds_matrix():
    summary = dataset_summary(DEMO_DIR)
    assert summary["configured"] and summary["num_classes"] >= 2
    assert summary["valid_images"] >= 6
    assert all(count >= 2 for count in summary["classes"].values())


@needs_demo
@pytest.mark.slow
def test_end_to_end_training_on_demo(tmp_path):
    """Full pipeline on the shipped fictional personas (fast: 30 small images)."""
    from ml.pipeline import build_feature_matrix
    cfg = config.PipelineConfig(extractor_params={}, augment_per_image=1,
                                cross_validate=False, test_size=0.3, seed=5)
    build = build_feature_matrix(DEMO_DIR, cfg)
    assert build.X.shape[0] >= len(build.classes) * 2
    assert build.X.shape[1] == get_extractor(cfg.feature_extractor).dim_hint \
        or build.X.shape[1] > 100
    assert set(np.unique(build.y)) == set(range(len(build.classes)))

    bundle = train_model(build, cfg, dataset_dir=DEMO_DIR)
    path = save_model(bundle, tmp_path / "demo.pkl", write_metrics_json=False)
    identifier = FaceIdentifier(model_path=path)
    assert identifier.ready

    images = sorted((DEMO_DIR / bundle.classes[0]).glob("*.jpg"))
    result = identifier.identify_image_bytes(images[0].read_bytes())
    assert result["status"] in {"identified", "unknown"}
    if result["status"] == "identified":
        assert result["player"]["name"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_predictions"]) >= 1
