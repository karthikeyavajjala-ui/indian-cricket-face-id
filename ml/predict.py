"""
Prediction CLI - sanity-check a photo against the trained model.

    python -m ml.predict --image photo.jpg
    python -m ml.predict --dir samples/ --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .predictor import FaceIdentifier

CYAN, GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def print_result(result: dict, image_label: str) -> None:
    print(c(f"\n▌ {image_label}", BOLD))
    status = result.get("status")
    if status == "identified":
        player = result["player"]
        print(c(f"  PLAYER IDENTIFIED · {player['name']}", GREEN))
        print(f"  confidence        : {result['confidence'] * 100:.1f}%  "
              f"(threshold {result['threshold'] * 100:.0f}%)")
        if result.get("similarity") is not None:
            print(f"  prototype match   : {result['similarity'] * 100:.1f}%")
        print(f"  margin over #2    : {result['margin'] * 100:.1f} pts")
    elif status == "unknown":
        print(c(f"  UNKNOWN / LOW CONFIDENCE ({result.get('reason')})", YELLOW))
        if result.get("message"):
            print(f"  {result['message']}")
    elif status == "no_face":
        print(c("  No face detected in this image.", YELLOW))
    elif status == "model_missing":
        print(c(f"  {result.get('message')}", RED))
        print(f"  Train first: python -m ml.train --dataset dataset")
    else:
        print(c(f"  {status}: {result.get('message', '')}", YELLOW))

    if result.get("top_predictions"):
        print("  top predictions:")
        for i, p in enumerate(result["top_predictions"], 1):
            bar = "█" * int(round(p["confidence"] * 30)) + \
                  "░" * (30 - int(round(p["confidence"] * 30)))
            print(f"   {i}. {p['display_name']:<22} {p['confidence'] * 100:5.1f}%  {bar}")
    if result.get("faces_detected", 0) > 1:
        print(c(f"  {result['faces_detected']} faces detected - largest face was used.", DIM))
    if result.get("timings_ms"):
        t = result["timings_ms"]
        print(c(f"  timings: detect {t['detect']}ms · encode {t['encode']}ms · "
                f"classify {t['classify']}ms · total {t['total']}ms", DIM))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ml.predict",
                                description="Identify a player in an image",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="single image file")
    group.add_argument("--dir", type=Path, help="directory of images")
    p.add_argument("--model", type=Path, default=config.DEFAULT_MODEL_PATH)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None,
                   help="override the stored confidence threshold")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {} if args.threshold is None else {"confidence_threshold": args.threshold}
    # the catalog only improves display names (e.g. "K. L. Rahul" instead of "KL Rahul");
    # the ML core itself stays independent of the web app.
    catalog = {}
    try:
        from backend.catalog import catalog_keyed_by_slug
        catalog = catalog_keyed_by_slug()
    except Exception:
        pass
    identifier = FaceIdentifier(model_path=args.model, cfg_overrides=overrides,
                               player_catalog=catalog)

    if not identifier.ready:
        message = identifier.error
        if args.json:
            print(json.dumps({"status": "model_missing", "message": message}, indent=2))
        else:
            print(c(message, RED))
            print("Train first:  python -m ml.train --dataset dataset")
        return 2

    if args.image:
        files = [args.image]
    else:
        files = sorted(p for p in args.dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS)
        if not files:
            print(c(f"No images found in {args.dir}", RED))
            return 3

    results = {}
    for file in files:
        try:
            data = Path(file).read_bytes()
        except OSError as exc:
            print(c(f"  cannot read {file}: {exc}", RED))
            continue
        result = identifier.identify_image_bytes(data, top_k=args.top_k)
        results[str(file)] = result
        if not args.json:
            print_result(result, str(file))

    if args.json:
        print(json.dumps(results if len(results) > 1 else next(iter(results.values())),
                         indent=2))
    elif len(results) > 1:
        identified = sum(1 for r in results.values() if r.get("status") == "identified")
        print(c(f"\n{identified}/{len(results)} images identified.", BOLD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
