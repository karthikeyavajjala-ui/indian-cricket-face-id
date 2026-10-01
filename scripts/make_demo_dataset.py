#!/usr/bin/env python3
"""
Build the DEMO dataset + (optionally) train the demo model.

The demo exists so the app can be explored end-to-end before you have any real
player photographs. It uses **AI-generated fictional people** that were produced
with an image model - nobody in it is a real person, so there is no likeness,
copyright or celebrity-rights problem, and it is explicitly NOT a stand-in for
your own dataset.

Source images live in ``dataset_demo/_sources/`` (folders starting with `_` are
ignored by the trainer). This script derives several plausible "different photos"
per persona (framing, zoom, rotation, perspective, exposure, JPEG quality,
mirroring) so the classifier has something to learn from, then optionally runs
the normal training pipeline into ``models/demo_model.pkl``.

    python scripts/make_demo_dataset.py --train
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config                                  # noqa: E402
from ml.face import align_face, make_detector          # noqa: E402
from ml.pipeline import build_feature_matrix           # noqa: E402
from ml.model import save_model, train_model           # noqa: E402

GREEN, YELLOW, DIM, BOLD, RESET = "\033[92m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"

DEMO_LABELS = {
    "persona_a": "Demo_Persona_A",
    "persona_b": "Demo_Persona_B",
    "persona_c": "Demo_Persona_C",
}
HOLDOUT_LABELS = {"stranger_d": "Unknown_Person_X"}


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def rephotograph(img: np.ndarray, face, rng: random.Random, index: int) -> np.ndarray:
    """Derive a plausible 'different photo' of the same (synthetic) person."""
    h, w = img.shape[:2]
    cx, cy = face.x + face.w / 2, face.y + face.h / 2

    # --- reframing: zoom + pan (mimics standing closer/further, different crop)
    zoom = rng.uniform(1.15, 1.85)
    cw = min(w, int(face.w * zoom * 2.6))
    ch = min(h, int(cw * rng.uniform(1.15, 1.45)))
    cw, ch = min(cw, w), min(ch, h)
    ox = int(np.clip(cx - cw / 2 + rng.uniform(-0.25, 0.25) * face.w, 0, w - cw))
    oy = int(np.clip(cy - ch / 2 + rng.uniform(-0.22, 0.22) * face.h, 0, h - ch))
    crop = img[oy:oy + ch, ox:ox + cw].copy()

    # --- small roll + perspective (hand-held camera feel)
    angle = rng.uniform(-7.0, 7.0)
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, rng.uniform(0.96, 1.05))
    crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.6:
        jitter = rng.uniform(0.006, 0.02)
        src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        dst = src + np.float32([[rng.uniform(-1, 1) * jitter * cw,
                                 rng.uniform(-1, 1) * jitter * ch] for _ in range(4)])
        crop = cv2.warpPerspective(crop, cv2.getPerspectiveTransform(src, dst), (cw, ch),
                                   borderMode=cv2.BORDER_REFLECT_101)

    # --- exposure / white balance / compression character
    crop = cv2.convertScaleAbs(crop, alpha=rng.uniform(0.85, 1.18),
                               beta=rng.uniform(-22, 22))
    b, g, r = cv2.split(crop.astype(np.float32))
    b *= rng.uniform(0.93, 1.07)
    g *= rng.uniform(0.96, 1.04)
    r *= rng.uniform(0.93, 1.07)
    crop = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        crop = cv2.GaussianBlur(crop, (3, 3), rng.uniform(0.4, 1.1))
    if rng.random() < 0.3:
        crop = np.clip(crop.astype(np.float32) +
                       np.random.normal(0, rng.uniform(2, 7), crop.shape), 0, 255).astype(np.uint8)

    # --- mirror (a real photo can be flipped by whoever published it)
    if index == 4:
        crop = cv2.flip(crop, 1)

    # --- export like a real photograph
    if max(crop.shape[:2]) > 720:
        s = 720 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)
    return crop


def build_dataset(sources_dir: Path, out_dir: Path, variants: int, seed: int) -> dict:
    rng = random.Random(seed)
    detector = make_detector("auto")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_in_out = sources_dir.resolve() == (out_dir / "_sources").resolve()

    # clean only generated class folders - never the source images themselves
    for child in out_dir.iterdir():
        if child.is_dir() and not child.name.startswith("_"):
            shutil.rmtree(child)
        elif child.name == "_holdout":
            shutil.rmtree(child)
    (out_dir / "_holdout").mkdir(exist_ok=True)
    if not sources_in_out:
        (out_dir / "_sources").mkdir(exist_ok=True)

    manifest: dict = {"detector": detector.name, "variants_per_persona": variants,
                      "classes": {}, "holdout": []}

    for source in sorted(sources_dir.glob("*")):
        if source.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        img = cv2.imread(str(source))
        if img is None:
            print(c(f"  ! cannot read {source.name}", YELLOW))
            continue
        stem = source.stem.lower()
        faces = detector.detect(img, max_faces=3)
        if not faces:
            print(c(f"  ! no face found in {source.name} - skipped", YELLOW))
            continue
        face = faces[0]

        if stem in HOLDOUT_LABELS:
            dest = out_dir / "_holdout" / f"{stem}.jpg"
            cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            manifest["holdout"].append(str(dest.relative_to(out_dir)))
            print(c(f"  holdout  {HOLDOUT_LABELS[stem]:<18} ← {source.name}", DIM))
            continue

        label = DEMO_LABELS.get(stem, f"Demo_{source.stem}")
        class_dir = out_dir / label
        class_dir.mkdir(exist_ok=True)
        written = []
        # variant 0: a clean face-aligned portrait (high quality reference)
        reference = align_face(img, face, 400)
        ref_path = class_dir / "00_portrait.jpg"
        cv2.imwrite(str(ref_path), reference, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written.append(ref_path.name)
        for i in range(1, variants + 1):
            variant = rephotograph(img, face, rng, i)
            path = class_dir / f"{i:02d}_photo.jpg"
            quality = rng.choice([78, 85, 90, 95])
            cv2.imwrite(str(path), variant, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            written.append(path.name)
        manifest["classes"][label] = written
        print(c(f"  built    {label:<18} {len(written):>2} images  ← {source.name}", GREEN))

    (out_dir / "README.md").write_text(
        "# Demo dataset (AI-generated fictional people)\n\n"
        "Every face in this folder was generated by a text-to-image model. Nobody\n"
        "here is a real person, so this data may be used freely for demos and tests,\n"
        "but it must NOT be presented as a real player dataset.\n\n"
        f"Source files (untouched): `_sources/`  ·  Rebuild: "
        "`python scripts/make_demo_dataset.py`\n\n"
        "`_holdout/` contains a person who is deliberately NOT in any class - use it\n"
        "to verify that the model answers \"Unknown\" for strangers.\n",
        encoding="utf-8")
    return manifest


def train_demo(dataset_dir: Path, model_out: Path, extractor: str, augment: int) -> int:
    cfg = config.PipelineConfig(feature_extractor=extractor, augment_per_image=augment,
                                classifier="svc", confidence_threshold=0.55,
                                min_margin=0.12, test_size=0.25)
    print(c("\nTraining the demo model …", BOLD))
    build = build_feature_matrix(dataset_dir, cfg)
    bundle = train_model(build, cfg, dataset_dir=dataset_dir)
    path = save_model(bundle, model_out)
    ev = bundle.metrics.get("evaluation", {})
    print(f"  extractor  : {build.stats['feature_extractor']}  dim={build.X.shape[1]}")
    print(f"  samples    : {build.X.shape[0]} ({build.stats['augmented']} augmented)")
    print(f"  validation accuracy: {ev.get('accuracy')}")
    print(f"  saved      : {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/make_demo_dataset.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sources", type=Path, default=config.DEMO_DATASET_DIR / "_sources")
    p.add_argument("--out", type=Path, default=config.DEMO_DATASET_DIR)
    p.add_argument("--variants", type=int, default=9,
                   help="derived photos generated per persona")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--extractor", default="auto")
    p.add_argument("--augment", type=int, default=4)
    p.add_argument("--train", action="store_true", help="also train models/demo_model.pkl")
    args = p.parse_args(argv)

    sources = [s for s in args.sources.glob("*") if s.suffix.lower() in config.IMAGE_EXTENSIONS]
    if not sources:
        print(c(f"No source images in {args.sources}.", YELLOW))
        print("This project ships with the fictional demo personas there; if the folder is "
              "empty, either restore it or skip the demo and use your own dataset/.")
        return 1

    print(c("\nBuilding demo dataset from AI-generated fictional personas", BOLD))
    print(c("  these people do not exist - the demo is not a player dataset\n", DIM))
    build_dataset(args.sources, args.out, args.variants, args.seed)

    if args.train:
        return train_demo(args.out, config.DEMO_MODEL_PATH, args.extractor, args.augment)

    print(c("\nNext:", BOLD))
    print(f"  python -m ml.train --dataset {args.out} --model-out {config.DEMO_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
