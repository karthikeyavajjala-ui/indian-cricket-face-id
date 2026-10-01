#!/usr/bin/env python3
"""
Fetch the OPTIONAL OpenCV-Zoo ONNX models used by the stronger pipeline.

  * YuNet  - face detector (better than Haar on pose/lighting/group photos)
  * SFace  - 128-D face embedding (MobileFaceNet trained with the SFace loss)

Both are Apache-2.0 licensed OpenCV Zoo models trained on public face datasets.
They contain no images of any player, so downloading them is not a dataset
download - but this script still asks for confirmation unless --yes is given.

    python scripts/fetch_models.py --yes

Without these models the app runs fully on OpenCV's bundled Haar cascade and the
classic HOG+LBP descriptor (identical API, slightly lower accuracy).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config                      # noqa: E402

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = {
    "yunet": {
        "url": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "dest": config.YUNET_MODEL_PATH,
        "size": 232_589,
        "desc": "YuNet face detector (OpenCV Zoo, Apache-2.0)",
        "sha256": None,
    },
    "sface": {
        "url": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "dest": config.SFACE_MODEL_PATH,
        "size": 38_696_353,
        "desc": "SFace face embedding, 128-D (OpenCV Zoo, Apache-2.0)",
        "sha256": None,
    },
}


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(c(f"  ↓ {url}", DIM))

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(1.0, blocks * block_size / total)
            bar = "█" * int(done * 30) + "░" * (30 - int(done * 30))
            sys.stdout.write(f"\r  [{bar}] {done * 100:5.1f}%")
            sys.stdout.flush()

    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                hook(read // (1 << 16), 1 << 16, total)
    sys.stdout.write("\r" + " " * 60 + "\r")
    tmp.replace(dest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python scripts/fetch_models.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--detector-only", action="store_true")
    p.add_argument("--embedding-only", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    wanted = ["yunet", "sface"]
    if args.detector_only:
        wanted = ["yunet"]
    elif args.embedding_only:
        wanted = ["sface"]

    print(c("\nOpenCV-Zoo ONNX models", BOLD))
    for key in wanted:
        info = MODELS[key]
        print(f"  • {info['desc']}")
        print(c(f"      {info['url']}", DIM))
        print(c(f"      → {info['dest']}  (~{info['size'] / 1e6:.1f} MB)", DIM))

    if not args.yes:
        try:
            answer = input("\nDownload now? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print(c("Aborted. The app still runs with the Haar + HOG/LBP pipeline.", YELLOW))
            return 1

    rc = 0
    for key in wanted:
        info = MODELS[key]
        dest: Path = info["dest"]
        if dest.is_file() and not args.force:
            print(c(f"\n{key}: already present ({sha256(dest)[:12]}…)", GREEN))
            continue
        print(c(f"\n{key}: downloading …", BOLD))
        try:
            download(info["url"], dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(c(f"  failed: {exc}", RED))
            print(c("  Check your connection, or download the file manually from the URL above.",
                    YELLOW))
            rc = 2
            continue
        digest = sha256(dest)
        print(c(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]}…)",
                GREEN))
        if key == "sface":
            try:
                import cv2
                if hasattr(cv2, "FaceRecognizerSF"):
                    cv2.FaceRecognizerSF.create(str(dest), "")
                    print(c("  verified: SFace loads in this OpenCV build.", GREEN))
                else:
                    print(c("  warning: this OpenCV build has no FaceRecognizerSF "
                            "(needs ≥ 4.5.4) - the sface extractor will stay unavailable.",
                            YELLOW))
            except Exception as exc:                     # pragma: no cover
                print(c(f"  warning: verification failed ({exc})", YELLOW))

    print(c("\nNow retrain so the model records the new engine:\n"
            "    python -m ml.train --dataset dataset\n", BOLD))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
