#!/usr/bin/env python3
"""
Dataset preparation / audit tool  -  scripts/prepare_dataset.py

The ML pipeline never downloads face images. You legally obtain photos of the
players you want to support and place them in:

    dataset/
        Virat_Kohli/  *.jpg *.jpeg *.png *.webp
        Rohit_Sharma/
        ...

This script helps you do the mechanical parts:

  --roster            show the player catalog with the exact folder names to use
  --init              create the dataset tree (empty folders + README + manifest)
  --init --players "Virat_Kohli,Rohit_Sharma"
  --audit             validate every image (size, blur, face detection, duplicates)
  --optimize          re-encode images (max side 1600px, JPEG q92, EXIF stripped)
  --min-images 8      audit warning threshold

Examples
--------
    python scripts/prepare_dataset.py --init
    python scripts/prepare_dataset.py --audit --dataset dataset
    python scripts/prepare_dataset.py --optimize --dataset dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config                      # noqa: E402
from ml.face import make_detector          # noqa: E402
from ml.pipeline import dataset_summary, scan_dataset   # noqa: E402

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


DATASET_README = """\
# Dataset folder

One sub-folder per supported player, folder name = player name with
underscores, e.g.:

    Virat_Kohli/
        01_face.jpg
        02_face.jpg
        ...

Rules
-----
* Only legally usable images. You are responsible for the licence/usage rights
  of every file you put here (CC-licensed photos with attribution, your own
  photographs, images you have explicit permission to use, ...).
* 10-30 clear, front-facing photos per player works far better than 3 photos.
  Vary lighting, background, expression, head angle and distance.
* This tool intentionally does NOT scrape or download photos.
* Folder names starting with `.` or `_` are ignored by the trainer.

Then run:

    python scripts/prepare_dataset.py --audit
    python -m ml.train --dataset dataset
"""


def load_roster() -> list[dict]:
    """Player catalog (real roster + clearly-labelled fictional demo personas)."""
    try:
        from backend.catalog import all_profiles
        return all_profiles()
    except Exception:                                        # pragma: no cover
        return []


def cmd_roster(args) -> int:
    roster = load_roster()
    print(c("\nPlayer catalog (folder name → display name)\n", BOLD))
    for entry in roster:
        tag = "DEMO (fictional)" if entry.get("fictional") else "real player"
        print(f"  {entry['folder']:<22} {entry['name']:<28} {c(tag, DIM)}")
        print(c(f"       role: {entry.get('role', '-')}", DIM))
    print(c("\nUse these folder names so profiles and dataset folders match.\n", DIM))
    return 0


def cmd_init(args) -> int:
    dataset_dir = Path(args.dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "README.md").write_text(DATASET_README, encoding="utf-8")
    (dataset_dir / ".gitkeep").touch()

    roster = load_roster()
    names = [n.strip() for n in (args.players.split(",") if args.players else []) if n.strip()]
    if not names:
        names = [e["folder"] for e in roster if not e.get("fictional")]
    created = []
    for name in names:
        folder = dataset_dir / name
        if not folder.exists():
            folder.mkdir(parents=True)
            (folder / "PUT_IMAGES_HERE.txt").write_text(
                f"Drop legally usable photos of {name.replace('_', ' ')} in this folder,\n"
                f"then delete this file and run:  python -m ml.train --dataset {dataset_dir}\n",
                encoding="utf-8")
            created.append(name)
    print(c(f"\nDataset initialised at {dataset_dir}", GREEN))
    print(f"  folders created: {', '.join(created) if created else '(already existed)'}")
    print(f"  next: add images, then run")
    print(c(f"    python scripts/prepare_dataset.py --audit --dataset {dataset_dir}", BOLD))
    print(c(f"    python -m ml.train --dataset {dataset_dir}", BOLD))
    return 0


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def cmd_audit(args) -> int:
    dataset_dir = Path(args.dataset)
    detector = make_detector(args.detector)
    summary = dataset_summary(dataset_dir)

    print(c(f"\nAudit · {dataset_dir}", BOLD))
    print(f"  detector           : {detector.name}")
    if not summary["configured"]:
        print(c("  Dataset not configured: need ≥ 2 player folders containing images.", RED))
        print(f"  Run:  python scripts/prepare_dataset.py --init --dataset {dataset_dir}")
        return 2

    samples, counts = scan_dataset(dataset_dir)
    ok_samples = [s for s in samples if s.ok]

    hashes: dict[str, list[str]] = defaultdict(list)
    no_face: list[str] = []
    multi_face: list[str] = []
    tiny: list[str] = []
    blurry: list[str] = []

    for i, sample in enumerate(ok_samples):
        img = cv2.imread(str(sample.path))
        if img is None:
            continue
        faces = detector.detect(img, max_faces=5)
        if not faces:
            no_face.append(str(sample.path.relative_to(dataset_dir)))
        elif len(faces) > 1:
            multi_face.append(f"{sample.path.relative_to(dataset_dir)} ({len(faces)})")
        if min(sample.shape) < 200:
            tiny.append(f"{sample.path.relative_to(dataset_dir)} ({sample.shape[0]}x{sample.shape[1]})")
        if sample.blur_variance < config.BLUR_VARIANCE_THRESHOLD:
            blurry.append(f"{sample.path.relative_to(dataset_dir)} (var={sample.blur_variance:.1f})")
        hashes[_file_hash(sample.path)].append(str(sample.path.relative_to(dataset_dir)))
        if (i + 1) % 10 == 0:
            sys.stdout.write(f"\r  analysed {i + 1}/{len(ok_samples)} images …")
            sys.stdout.flush()
    sys.stdout.write("\r" + " " * 60 + "\r")

    print(f"  images             : {len(samples)} files, {len(ok_samples)} valid, "
          f"{summary['invalid_images']} rejected")
    print(f"  players            : {summary['num_classes']}")
    print(f"  with a detectable face : {len(ok_samples) - len(no_face)}")
    print(f"  faces detected         : {len(ok_samples) - len(no_face)}")

    print(c("\n  per player:", BOLD))
    for name, count in sorted(counts.items()):
        colour = GREEN if count >= args.min_images else (YELLOW if count >= 2 else RED)
        note = "" if count >= args.min_images else f"  ← add more (target ≥ {args.min_images})"
        print(f"    {name:<24} {count:>4} images {c(note, colour)}")

    duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
    def _report(title: str, items: list[str], colour: str = YELLOW, limit: int = 12) -> None:
        if not items:
            return
        print(c(f"\n  {title} ({len(items)}):", colour))
        for item in items[:limit]:
            print(f"    - {item}")
        if len(items) > limit:
            print(c(f"    … and {len(items) - limit} more", DIM))

    _report("images with no detectable face (will be skipped in training)", no_face)
    _report("images with multiple faces (largest face is used)", multi_face, DIM)
    _report("very small images (<200px)", tiny)
    _report("possibly blurry images", blurry)
    if duplicates:
        print(c(f"\n  duplicate files by content ({len(duplicates)} groups):", YELLOW))
        for files in list(duplicates.values())[:10]:
            print(f"    - {' == '.join(files)}")

    problems = len(no_face) + len(tiny) + len(blurry) + len(duplicates)
    print()
    if problems == 0:
        print(c("  Dataset looks healthy. Train with:", GREEN))
    else:
        print(c(f"  {problems} item(s) worth cleaning up. Then train with:", YELLOW))
    print(c(f"    python -m ml.train --dataset {dataset_dir}", BOLD))

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "summary": summary, "no_face": no_face, "multi_face": multi_face,
            "tiny": tiny, "blurry": blurry, "duplicates": list(duplicates.values()),
        }, indent=2), encoding="utf-8")
        print(c(f"  report written to {args.json_out}", DIM))
    return 0


def cmd_optimize(args) -> int:
    """Re-encode images so training/inference are fast and consistent."""
    dataset_dir = Path(args.dataset)
    samples, _ = scan_dataset(dataset_dir)
    changed = 0
    for sample in samples:
        if not sample.ok:
            continue
        img = cv2.imread(str(sample.path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > args.max_side:
            scale = args.max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        target = sample.path.with_suffix(".jpg")
        ok = cv2.imwrite(str(target), img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if ok and target != sample.path:
            sample.path.unlink(missing_ok=True)
        changed += 1
    print(c(f"\nOptimised {changed} image(s) in {dataset_dir} "
            f"(max side {args.max_side}px, JPEG q{args.quality}, EXIF stripped)", GREEN))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/prepare_dataset.py",
        description="Initialise / audit the player image dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    p.add_argument("--detector", default="auto", help="detector used for the audit")
    p.add_argument("--min-images", type=int, default=8,
                   help="recommended images per player")
    p.add_argument("--roster", action="store_true", help="print the player catalog")
    p.add_argument("--init", action="store_true", help="create the dataset tree")
    p.add_argument("--players", default=None,
                   help="comma separated folder names for --init")
    p.add_argument("--audit", action="store_true", help="validate images")
    p.add_argument("--optimize", action="store_true", help="resize/re-encode images")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--json-out", type=Path, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.roster:
        return cmd_roster(args)
    if args.init:
        return cmd_init(args)
    if args.optimize:
        return cmd_optimize(args)
    if args.audit:
        return cmd_audit(args)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
