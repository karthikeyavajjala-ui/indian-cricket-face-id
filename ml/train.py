"""
Training CLI - the full offline pipeline.

    python -m ml.train --dataset dataset --classifier svc --extractor auto

Pipeline:  load dataset -> validate images -> detect faces -> align/crop ->
           augment -> extract features -> build X/y -> train -> validate ->
           metrics -> save models/model.pkl (+ models/metrics.json)

Everything that would change the meaning of the trained model is printed and
stored in the bundle, so a prediction can always be traced back to its model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from .features import available_extractors, get_extractor
from .face import detector_status, make_detector
from .model import CLASSIFIERS, save_model, train_model
from .pipeline import build_feature_matrix, dataset_summary

# --------------------------------------------------------------------------
# pretty printing helpers (no external deps)
# --------------------------------------------------------------------------
CYAN, GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if _supports_colour() else text


def banner(text: str, char: str = "─") -> None:
    line = char * 74
    print(_c(line, DIM))
    print(_c(f" {text}", BOLD))
    print(_c(line, DIM))


def bar(value: float, width: int = 28) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "░" * (width - filled)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ml.train",
        description="Train the INDIAN CRICKET FACE ID classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR,
                   help="dataset root containing one folder per player")
    p.add_argument("--model-out", type=Path, default=config.DEFAULT_MODEL_PATH,
                   help="where to write the trained model (.pkl)")
    p.add_argument("--classifier", default="svc", choices=CLASSIFIERS,
                   help="classifier family")
    p.add_argument("--extractor", default=config.PipelineConfig().feature_extractor,
                   help="feature extractor: auto | hog_lbp | sface | pixel")
    p.add_argument("--detector", default=config.PipelineConfig().detector,
                   help="face detector: auto | yunet | haar")
    p.add_argument("--augment", type=int, default=config.PipelineConfig().augment_per_image,
                   help="augmented variants generated per image (0 = off)")
    p.add_argument("--pca", type=int, default=config.PipelineConfig().pca_components,
                   help="PCA components applied before the classifier (0 = off)")
    p.add_argument("--test-size", type=float, default=config.PipelineConfig().test_size,
                   help="held-out validation fraction")
    p.add_argument("--threshold", type=float, default=config.PipelineConfig().confidence_threshold,
                   help="confidence threshold stored for inference-time rejection")
    p.add_argument("--margin", type=float, default=config.PipelineConfig().min_margin,
                   help="minimum top-2 margin stored for inference-time rejection")
    p.add_argument("--min-images", type=int, default=4,
                   help="warn when a player folder has fewer usable images than this")
    p.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    p.add_argument("--seed", type=int, default=config.PipelineConfig().seed)
    p.add_argument("--json-out", type=Path, default=None,
                   help="also write the full metrics report as JSON")
    p.add_argument("--list-extractors", action="store_true")
    p.add_argument("--scan-only", action="store_true",
                   help="validate the dataset and exit without training")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config.ensure_dirs()

    if args.list_extractors:
        for info in available_extractors():
            mark = "available " if info["available"] else "MISSING   "
            print(f" {mark} {info['name']:<10} dim={info.get('dim', '?'):<6} {info['description']}")
            if not info["available"]:
                print(f"    └─ unavailable: {info.get('reason', '')}")
        return 0

    banner("INDIAN CRICKET FACE ID · model training")
    print(f" dataset          : {args.dataset}")
    print(f" model output     : {args.model_out}")
    print(f" detector         : {args.detector}  {_c(str(detector_status()['active'] + ' active'), DIM)}")
    print(f" extractor        : {args.extractor}")
    print(f" classifier       : {args.classifier}")
    print(f" augmentation     : {args.augment} variants / image")
    print(f" decision policy  : confidence ≥ {args.threshold:.2f}, margin ≥ {args.margin:.2f}")

    # ---------------------------------------------------------------- scan
    banner("1 · dataset scan")
    summary = dataset_summary(args.dataset)
    if not summary["configured"]:
        print(_c("  Dataset not configured.", RED))
        print(f"  Expected: {args.dataset}/<Player_Name>/photo.jpg  (at least 2 folders)")
        print("  Add legally usable images, then re-run. See docs/DATASET.md")
        print("  Quick start:  python scripts/prepare_dataset.py --init")
        return 2

    for name, count in sorted(summary["classes"].items()):
        colour = GREEN if count >= args.min_images else YELLOW
        print(f"  {name:<24} {count:>4} images  {_c(bar(min(count / 24, 1.0)), colour)}")
    print(f"  total: {summary['valid_images']} valid image(s), "
          f"{summary['invalid_images']} rejected, {summary['num_classes']} players")
    for problem in summary["problems"]:
        print(_c(f"  ! {problem['file']}: {problem['reason']}", YELLOW))
    small = [n for n, c in summary["classes"].items() if c < args.min_images]
    if small:
        print(_c(f"  ! fewer than {args.min_images} images for: {', '.join(small)}", YELLOW))
        print(_c("    Accuracy for those players will be poor - add more photos.", DIM))

    if args.scan_only:
        print("\n scan-only mode: nothing was trained.")
        return 0

    # ------------------------------------------------------- feature matrix
    banner("2 · detect → align → encode")
    try:
        extractor = get_extractor(args.extractor)
    except Exception as exc:
        print(_c(f"  extractor error: {exc}", RED))
        return 3
    print(f"  detector active : {make_detector(args.detector).name}")
    print(f"  extractor active: {extractor.name}")

    last_line = [""]

    def progress(message: str, percent: float) -> None:
        text = f"  [{percent:5.1f}%] {message[:48]:<50}"
        sys.stdout.write("\r" + text)
        sys.stdout.flush()
        last_line[0] = text

    cfg = config.PipelineConfig(
        detector=args.detector, feature_extractor=args.extractor,
        classifier=args.classifier, classifier_params={},
        pca_components=args.pca, test_size=args.test_size, seed=args.seed,
        augment_per_image=args.augment, confidence_threshold=args.threshold,
        min_margin=args.margin, cross_validate=not args.no_cv,
    )
    t0 = time.perf_counter()
    try:
        build = build_feature_matrix(args.dataset, cfg, progress=progress)
    except ValueError as exc:
        sys.stdout.write("\r" + " " * len(last_line[0]) + "\r")
        print(_c(f"  {exc}", RED))
        return 4
    sys.stdout.write("\r" + " " * len(last_line[0]) + "\r")

    stats = build.stats
    print(f"  images scanned      : {stats['images_seen']}")
    print(f"  images without face : {stats['images_without_face']}  "
          f"{_c('(skipped - use clearer photos)', DIM) if stats['images_without_face'] else ''}")
    print(f"  face crops encoded  : {stats['faces_used']} "
          f"({stats['augmented']} augmented variants)")
    print(f"  feature dimension   : {build.X.shape[1]}")
    print(f"  X/y shape           : {build.X.shape} / {build.y.shape}")
    print(f"  encoding time       : {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------ training
    banner("3 · training")
    try:
        bundle = train_model(build, cfg, dataset_dir=args.dataset, progress=progress)
    except Exception as exc:
        sys.stdout.write("\r" + " " * len(last_line[0]) + "\r")
        print(_c(f"  training failed: {exc}", RED))
        return 5
    sys.stdout.write("\r" + " " * len(last_line[0]) + "\r")
    m = bundle.metrics
    print(f"  training samples    : {m['training_samples']}")
    print(f"  validation samples  : {m['validation_samples']}")
    print(f"  classes             : {m['n_classes']}")
    print(f"  fit time            : {m['training_seconds']}s")

    # ------------------------------------------------------------- metrics
    banner("4 · held-out validation")
    ev = m.get("evaluation", {})
    if ev.get("accuracy") is None:
        print(_c("  No held-out split was possible - accuracy NOT reported.", YELLOW))
    else:
        for key, label in (("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
                           ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"),
                           ("f1_weighted", "F1 (weighted)")):
            if ev.get(key) is not None:
                print(f"  {label:<18}: {ev[key]:.4f}  {bar(ev[key])}")
        if ev.get("top3_accuracy") is not None:
            print(f"  {'Top-3 accuracy':<18}: {ev['top3_accuracy']:.4f}  {bar(ev['top3_accuracy'])}")
        if ev.get("roc_auc_macro") is not None:
            print(f"  {'ROC-AUC (macro)':<18}: {ev['roc_auc_macro']:.4f}")

        print("\n  per-class:")
        print(f"  {'player':<24}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>5}")
        for row in ev.get("per_class", []):
            print(f"  {row['class']:<24}{row['precision']:>7.3f}{row['recall']:>7.3f}"
                  f"{row['f1']:>7.3f}{row['support']:>5}")

        cm = ev.get("confusion_matrix")
        if cm:
            names = ev.get("confusion_matrix_labels", bundle.classes)
            short = [n[:12] for n in names]
            width = max(6, max(len(s) for s in short) + 1)
            print("\n  confusion matrix (rows = true, cols = predicted):")
            print(" " * (width + 2) + "".join(f"{s:>{width}}" for s in short))
            for i, row in enumerate(cm):
                print(f"  {short[i]:<{width}}" + "".join(f"{v:>{width}}" for v in row))

    cv = m.get("cross_validation")
    if cv and cv.get("available"):
        print(f"\n  cross-validation ({cv['folds']}-fold): "
              f"accuracy {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
    for note in m.get("notes", []):
        print(_c(f"  note: {note}", YELLOW))

    # ---------------------------------------------------------------- save
    banner("5 · saving")
    path = save_model(bundle, args.model_out)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  model   : {path}  ({size_mb:.2f} MB)")
    print(f"  metrics : {path.with_name('metrics.json')}")
    print(f"  version : {bundle.metadata.get('model_version')}  "
          f"(dataset fingerprint {bundle.metadata.get('dataset_fingerprint', '')[:12]})")
    print(f"  classes : {', '.join(bundle.classes)}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "metadata": bundle.metadata, "cfg": bundle.cfg,
            "classes": bundle.classes, "metrics": bundle.metrics,
        }, indent=2), encoding="utf-8")
        print(f"  report  : {args.json_out}")

    print(_c("\n  done. restart the API (or POST /api/model/reload) to serve this model.", DIM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
